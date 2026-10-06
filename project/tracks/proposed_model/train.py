"""
Training script for Graph-Aware Mixture-of-Experts (Track B).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG, create_graph_aware_moe


def load_muad_baseline(device, model_path=None):
    """Load MUAD baseline model for encoder reuse."""
    from vendor.muad.models.main_model import MainModel
    from vendor.muad.training.base_model import BaseModel
    
    if model_path and Path(model_path).exists():
        # Load from checkpoint
        baseline = BaseModel(
            device, lr=0.001, epochs=50, patience=15,
            result_dir='./tmp', hash_id='baseline',
            hidden_dim=[64, 64], event_num=14
        )
        baseline.load_state_dict(torch.load(model_path, map_location=device))
        return baseline.model
    return None


def main():
    parser = argparse.ArgumentParser(description="Train Graph-Aware MoE")
    parser.add_argument("--config", type=str, default=None, help="Config JSON file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--muad-checkpoint", type=str, default=None, help="MUAD baseline checkpoint to reuse encoders")
    parser.add_argument("--output-dir", type=str, default="./runs/proposed_model", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Override with CLI args
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    
    # Load data
    print("Loading data...")
    data_dir = Path("project/preprocessing/muad_compat/output")
    
    train_data, node_num, edges = load_data(
        str(data_dir / "chunk_train.pkl"),
        str(data_dir / "metadata.json")
    )
    test_data, _, _ = load_data(
        str(data_dir / "chunk_test.pkl"),
        str(data_dir / "metadata.json")
    )
    
    print(f"Train chunks: {len(train_data)}, Test chunks: {len(test_data)}")
    print(f"Nodes: {node_num}, Edges: {len(edges[0])}")
    
    # Create datasets
    train_dataset = ChunkDataset(train_data, list(train_data.keys()), node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load MUAD baseline for encoder reuse
    print("Loading MUAD baseline...")
    muad_baseline = load_muad_baseline(device, args.muad_checkpoint)
    
    # Create model
    print("Creating Graph-Aware MoE...")
    model = create_graph_aware_moe(config, muad_baseline).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (graph, labels) in enumerate(train_loader):
            graph = graph.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(graph, labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["y_pred"]
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for graph, labels in test_loader:
                graph = graph.to(device)
                labels = labels.to(device)
                
                outputs = model(graph, labels)
                loss = outputs.get("loss", criterion(outputs["MMlogit"], labels))
                
                test_loss += loss.item()
                preds = outputs["y_pred"]
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        
        # Compute F1
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}, "
              f"F1: {f1:.4f}, Pre: {precision:.4f}, Rec: {recall:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  -> New best F1: {best_f1:.4f} at epoch {best_epoch}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
            }, output_dir / f"checkpoint_epoch_{epoch}.pth")
    
    print(f"\nTraining complete!")
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()