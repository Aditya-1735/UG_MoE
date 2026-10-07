#!/usr/bin/env python3
"""
kaggle_train.py - Training script for Graph-Aware MoE on Kaggle
Usage: python kaggle_train.py [--experiment track_b_full] [--seeds 42,123] [--epochs 50]
"""
import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Train Graph-Aware MoE on Kaggle')
    parser.add_argument('--experiment', type=str, default='track_b_full',
                        choices=['track_b_full'],
                        help='Experiment type (only track_b_full implemented)')
    parser.add_argument('--seeds', type=str, default='42,123',
                        help='Comma-separated seeds (e.g., 42,123,456)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of train data for validation (for checkpoint selection)')
    return parser.parse_args()


def run_track_b_full(args):
    """Run Track B full experiment with given seeds"""
    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Val split: {args.val_split}")
    
    # Load data
    data_dir = Path('preprocessing/muad_compat/output')
    if not Path('preprocessing/muad_compat/output/chunk_train.pkl').exists():
        print("ERROR: Data not found at preprocessing/muad_compat/output/")
        print("Please ensure git lfs pull completed or data is available")
        return
    
    train_data, node_num, edges = load_data(
        'preprocessing/muad_compat/output/chunk_train.pkl',
        'preprocessing/muad_compat/output/metadata.json'
    )
    test_data, _, _ = load_data(
        'preprocessing/muad_compat/output/chunk_test.pkl',
        'preprocessing/muad_compat/output/metadata.json'
    )
    
    # Split train into train/val for checkpoint selection
    train_keys = list(train_data.keys())
    n_val = int(len(train_keys) * args.val_split)
    val_keys = train_keys[:n_val]
    train_keys = train_keys[n_val:]
    
    train_dataset = ChunkDataset(train_data, train_keys, node_num, edges)
    val_dataset = ChunkDataset(train_data, val_keys, node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Config
    config = DEFAULT_CONFIG.copy()
    config['log_dim'] = 14
    model_config = {k: v for k, v in config.items() if k != 'epochs'}
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"track_b_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"RUN seed={seed}")
        print(f"{'='*60}")
        
        seed_everything(seed)
        
        model = GraphAwareMoE(**{k:v for k,v in config.items() if k != 'epochs'}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        best_epoch = 0
        best_model_state = None
        
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss = 0
            for graph, labels in train_loader:
                graph, labels = graph.to(device), labels.to(device)
                binary_labels = (labels >= 1).long()
                
                optimizer.zero_grad()
                outputs = model(graph, binary_labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluate every 5 epochs on validation set
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for graph, labels in val_loader:
                        graph, labels = graph.to(device), labels.to(device)
                        binary_labels = (labels >= 1).long()
                        outputs = model(graph, binary_labels)
                        loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                        test_loss += loss.item()
                        preds = outputs["y_pred"]
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(binary_labels.cpu().tolist())
                
                avg_test_loss = test_loss / len(val_loader)
                from sklearn.metrics import f1_score, precision_score, recall_score
                f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
                precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
                recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
                
                print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f} | "
                      f"Val Loss={avg_test_loss:.4f} | F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
                
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    # Save best checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_f1': best_val_f1,
                        'seed': seed,
                    }, output_dir / f"best_seed{seed}.pth")
            else:
                # Quick eval without full metrics
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for graph, labels in test_loader:
                        graph, labels = graph.to(device), labels.to(device)
                        binary_labels = (labels >= 1).long()
                        outputs = model(graph, binary_labels)
                        preds = outputs["y_pred"]
                        correct += (preds == binary_labels).sum().item()
                        total += labels.size(0)
                
                test_acc = correct / total
                print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f} | Test Acc={test_acc:.4f}")
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for graph, labels in test_loader:
                graph, labels = graph.to(device), labels.to(device)
                binary_labels = (labels >= 1).long()
                outputs = model(graph, binary_labels)
                preds = outputs["y_pred"]
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(binary_labels.cpu().tolist())
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        final_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        final_pre = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        final_rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        
        result = {
            'seed': seed,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'final_f1': final_f1,
            'final_precision': final_pre,
            'final_recall': final_rec
        }
        
        # Save individual result immediately
        with open(output_dir / f"result_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print(f"  Final Test F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
        
        results.append(result)
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    if successful:
        f1s = [r['best_val_f1'] for r in successful]
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Successful runs: {len(successful)}/{len(seeds)}")
        print(f"Best Val F1s: {[f'{f:.4f}' for f in f1s]}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Max F1: {np.max(f1s):.4f}, Min F1: {np.min(f1s):.4f}")
    
    # Save final summary
    summary = {
        "experiment": "track_b_full",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "hidden_dim": 64,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": 1e-5,
            "event_num": 14,
            "seeds": seeds,
            "router_type": "uncertainty_guided",
            "fusion_type": "confidence",
            "uncertainty_type": "variational",
            "val_split": args.val_split
        },
        "results": results
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    args = parse_args()
    
    if args.experiment == 'track_b_full':
        run_track_b_full(args)
    else:
        print(f"Experiment '{args.experiment}' not implemented. Only 'track_b_full' is available.")
        sys.exit(1)


if __name__ == '__main__':
    main()