#!/usr/bin/env python3
"""
kaggle_train.py - Minimal training script for Kaggle
Usage: python kaggle_train.py [--experiment ablation|track_b_full|track_a_full] [--seeds 42,123] [--epochs 50]
"""
import argparse
import json
import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import project modules
from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Train Graph-Aware MoE on Kaggle')
    parser.add_argument('--experiment', type=str, default='track_b_full',
                        choices=['ablation', 'track_b_full', 'track_a_full'],
                        help='Experiment type')
    parser.add_argument('--seeds', type=str, default='42,123',
                        help='Comma-separated seeds (e.g., 42,123,456)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    return parser.parse_args()


def run_track_b_full(args):
    """Run Track B full experiment with given seeds"""
    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    
    # Load data
    data_dir = Path('project/preprocessing/muad_compat/output')
    if not Path('project/preprocessing/muad_compat/output/chunk_train.pkl').exists():
        print("ERROR: Data not found at project/preprocessing/muad_compat/output/")
        print("Please upload data as Kaggle Dataset or ensure it's in the repo")
        return
    
    train_data, node_num, edges = load_data(
        'project/preprocessing/muad_compat/output/chunk_train.pkl',
        'project/preprocessing/muad_compat/output/metadata.json'
    )
    test_data, _, _ = load_data(
        'project/preprocessing/muad_compat/output/chunk_test.pkl',
        'project/preprocessing/muad_compat/output/metadata.json'
    )
    
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Config
    config = DEFAULT_CONFIG.copy()
    config['log_dim'] = 14
    model_config = {k: v for k, v in config.items() if k != 'epochs'}
    
    results = []
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"RUN seed={seed}")
        print(f"{'='*60}")
        
        seed_everything(seed)
        
        model = GraphAwareMoE(**{k:v for k,v in config.items() if k != 'epochs'}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        best_epoch = 0
        
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
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                all_preds = []
                all_labels = []
                test_loss = 0
                with torch.no_grad():
                    for graph, labels in test_loader:
                        graph, labels = graph.to(device), labels.to(device)
                        binary_labels = (labels >= 1).long()
                        outputs = model(graph, binary_labels)
                        loss = outputs.get("loss", criterion(outputs["MMlogit"], binary_labels))
                        test_loss += loss.item()
                        preds = outputs["y_pred"]
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(binary_labels.cpu().tolist())
                
                avg_test_loss = test_loss / len(test_loader)
                from sklearn.metrics import f1_score, precision_score, recall_score
                f1 = f1_score(all_labels, all_preds, average='binary')
                precision = precision_score(all_labels, all_preds, average='binary')
                recall = recall_score(all_labels, all_preds, average='binary')
                
                print(f"Epoch {epoch:2d}: Train Loss={train_loss/len(train_loader):.4f} | "
                      f"Test F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch
        
        # Final eval
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
        
        final_f1 = f1_score(all_labels, all_preds, average='binary')
        final_pre = precision_score(all_labels, all_preds, average='binary')
        final_rec = recall_score(all_labels, all_preds, average='binary')
        
        results.append({
            'seed': seed,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'final_f1': final_f1,
            'final_precision': final_pre,
            'final_recall': final_rec
        })
        
        print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
        print(f"  Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    if successful:
        f1s = [r['best_f1'] for r in successful]
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Successful runs: {len(successful)}/{len(args.seeds)}")
        print(f"Best F1s: {[f'{f:.4f}' for f in f1s]}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Max F1: {np.max(f1s):.4f}, Min F1: {np.min(f1s):.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"runs/track_b_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}/summary.json", 'w') as f:
        json.dump({
            'experiment': 'track_b_full',
            'config': vars(args),
            'results': results
        }, f, indent=2)


def main():
    args = parse_args()
    
    if args.experiment == 'track_b_full':
        run_track_b_full(args)
    else:
        print(f"Experiment '{args.experiment}' not implemented yet")
        sys.exit(1)


if __name__ == '__main__':
    from datetime import datetime
    main()