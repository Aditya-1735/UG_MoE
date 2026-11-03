#!/usr/bin/env python3
"""
HDFS Training Script for Graph-Aware MoE (log_only_mode).
Runs 5 seeds × 50 epochs on HDFS dataset.
"""
import sys
import json
import torch
import dgl
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/muad"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vendor.muad.data.utils import load_data, create_dataloader
from vendor.muad.data.dataset import ChunkDataset
from vendor.muad.utils.general_utils import seed_everything
from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
from preprocessing.hdfs.hdfs_dataset import HDFSLogDataset, create_hdfs_dataloaders


def run_single_experiment(run_id: int, seed: int, output_base: Path, config: dict):
    """Run single HDFS experiment with given seed."""
    print(f"\n{'='*60}")
    print(f"RUN {run_id}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    seed_everything(seed)
    
    # Load HDFS data
    data_dir = Path(config["data_dir"])
    train_sequences = data_dir / "train_full.pkl"
    test_sequences = data_dir / "test_full.pkl"
    template_vocab = data_dir / "template_vocab.json"
    log_graph_dir = Path(config["log_graph_dir"])
    
    # Create dataloaders
    train_loader, test_loader = create_hdfs_dataloaders(
        train_sequences_pkl=str(train_sequences),
        test_sequences_pkl=str(test_sequences),
        template_vocab_json=str(template_vocab),
        log_graph_dir=str(config["log_graph_dir"]),
        batch_size=config["batch_size"],
        max_seq_len=config["max_seq_len"],
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model_config = config["model_config"].copy()
    model_config['log_dim'] = config.get("template_vocab_size", 50)
    model = GraphAwareMoE(**model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_epoch = 0
    patience = config.get("patience", 10)
    patience_counter = 0
    epoch_results = []
    
    print(f"  Training for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            template_ids = batch["template_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            # For log_only_mode, we need to pass the right inputs
            # The model expects a graph with ndata['logs'] containing template_ids
            # We'll create a dummy graph structure
            batch_size = template_ids.shape[0]
            seq_len = template_ids.shape[1]
            
            # Create a dummy graph for log-only mode
            # The model expects graph.ndata['logs'] with template_ids
            # Create a simple graph with batch_size * 1 nodes (one per sample)
            from dgl import batch as dgl_batch
            import dgl
            
            # Create individual graphs and batch them
            graphs = []
            for i in range(batch_size):
                g = dgl.graph(([], []), num_nodes=1)
                g.ndata['logs'] = template_ids[i:i+1].view(-1)
                graphs.append(g)
            
            batched_graph = dgl.batch(graphs).to(device)
            
            outputs = model(batched_graph, labels.to(device))
            loss = outputs["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["y_pred"]
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    template_ids = batch["template_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = template_ids.shape[0]
                    
                    # Create batched graph
                    import dgl
                    graphs = []
                    for i in range(batch_size):
                        g = dgl.graph(([], []), num_nodes=1)
                        g.ndata['logs'] = template_ids[i:i+1].view(-1)
                        graphs.append(g)
                    batched_graph = dgl.batch(graphs).to(device)
                    
                    outputs = model(batched_graph, labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], labels))
                    
                    test_loss += loss.item()
                    preds = outputs["y_pred"]
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            
            print(f"  Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
            
            epoch_results.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_f1,
                    'seed': seed,
                }, output_base / f"run_{run_id}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            template_ids = batch["template_ids"].to(device)
            labels = batch["label"].to(device)
            batch_size = template_ids.shape[0]
            seq_len = template_ids.shape[1]
            
            # Create batched graph for log-only mode
            g = dgl.graph(([], []), num_nodes=template_ids.numel(), device=device)
            g.ndata['logs'] = template_ids.view(-1)
            g.set_batch_num_nodes([seq_len] * template_ids.shape[0])
            batched_graph = g  # Already on device
            
            outputs = model(batched_graph, labels)
            preds = outputs["y_pred"]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "epoch_results": epoch_results
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"  Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Graph-Aware MoE on HDFS")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="./runs/hdfs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override with CLI args
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["device"] = args.device
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(config["data_dir"])
    train_sequences = data_dir / "train_full.pkl"
    test_sequences = data_dir / "test_full.pkl"
    template_vocab = data_dir / "template_vocab.json"
    log_graph_dir = Path(config["log_graph_dir"])
    
    # Create dataloaders
    from preprocessing.hdfs.hdfs_dataset import create_hdfs_dataloaders
    
    train_loader, test_loader = create_hdfs_dataloaders(
        train_sequences_pkl=str(train_sequences),
        test_sequences_pkl=str(test_sequences),
        template_vocab_json=str(template_vocab),
        log_graph_dir=str(log_graph_dir),
        batch_size=args.batch_size,
        max_seq_len=config.get("max_seq_len", 100),
        num_workers=0
    )
    
    # Create model
    model_config = config["model_config"].copy()
    model_config['log_dim'] = config.get("template_vocab_size", 50)
    model = GraphAwareMoE(**model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"hdfs_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    epoch_results = []
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            template_ids = batch["template_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Create batched graph
            import dgl
            batch_size = template_ids.shape[0]
            graphs = []
            for i in range(batch_size):
                g = dgl.graph(([], []), num_nodes=1)
                g.ndata['logs'] = template_ids[i:i+1].view(-1)
                graphs.append(g)
            batched_graph = dgl.batch(graphs).to(device)
            
            outputs = model(batched_graph, labels)
            loss = outputs["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["y_pred"]
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    template_ids = batch["template_ids"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = template_ids.shape[0]
                    
                    graphs = []
                    for i in range(batch_size):
                        g = dgl.graph(([], []), num_nodes=1)
                        g.ndata['logs'] = template_ids[i:i+1].view(-1)
                        graphs.append(g)
                    batched_graph = dgl.batch(graphs).to(device)
                    
                    outputs = model(batched_graph, labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], labels))
                    
                    test_loss += loss.item()
                    preds = outputs["y_pred"]
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            
            print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
            
            epoch_results.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_f1,
                    'seed': seed,
                }, output_base / f"run_{run_id}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            template_ids = batch["template_ids"].to(device)
            labels = batch["label"].to(device)
            batch_size = template_ids.shape[0]
            
            graphs = []
            for i in range(batch_size):
                g = dgl.graph(([], []), num_nodes=1)
                g.ndata['logs'] = template_ids[i:i+1].view(-1)
                graphs.append(g)
            batched_graph = dgl.batch(graphs).to(device)
            
            outputs = model(batched_graph, labels)
            preds = outputs["y_pred"]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    
    results = {
        "run_id": run_id,
        "seed": seed,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "epoch_results": epoch_results
    }
    
    with open(output_base / f"run_{run_id}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Graph-Aware MoE on HDFS")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="./runs/hdfs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override with CLI args
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["device"] = args.device
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(config["data_dir"])
    train_sequences = data_dir / "train_full.pkl"
    test_sequences = data_dir / "test_full.pkl"
    template_vocab = data_dir / "template_vocab.json"
    log_graph_dir = Path(config["log_graph_dir"])
    
    # Create dataloaders
    from preprocessing.hdfs.hdfs_dataset import create_hdfs_dataloaders
    
    train_loader, test_loader = create_hdfs_dataloaders(
        train_sequences_pkl=str(train_sequences),
        test_sequences_pkl=str(test_sequences),
        template_vocab_json=str(template_vocab),
        log_graph_dir=str(log_graph_dir),
        batch_size=args.batch_size,
        max_seq_len=config.get("max_seq_len", 100),
        num_workers=0
    )
    
    # Create model
    model_config = config["model_config"].copy()
    model_config['log_dim'] = config.get("template_vocab_size", 50)
    model = GraphAwareMoE(**model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"hdfs_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    epoch_results = []
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            template_ids = batch["template_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # The HDFSLogDataset returns template_ids as [B, seq_len]
            # Model expects graph with B*seq_len nodes and ndata['logs'] as flat [B*seq_len]
            batch_size = template_ids.shape[0]
            seq_len = template_ids.shape[1]
            
            # Create graph with B*seq_len nodes
            total_nodes = template_ids.shape[0] * template_ids.shape[1]
            g = dgl.graph(([], []), num_nodes=template_ids.numel(), device=device)
            g.ndata['logs'] = template_ids.reshape(-1)
            # Set batch info: each sample has seq_len nodes
            g.set_batch_num_nodes([seq_len] * batch_size)
            batched_graph = g  # Already on device
            
            outputs = model(batched_graph, labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["y_pred"]
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    template_ids = batch["template_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = template_ids.shape[0]
                    seq_len = template_ids.shape[1]
                    
                    # Create batched graph for log-only mode
                    g = dgl.graph(([], []), num_nodes=template_ids.numel(), device=device)
                    g.ndata['logs'] = template_ids.reshape(-1)
                    g.set_batch_num_nodes([seq_len] * batch_size)
                    batched_graph = g  # Already on device
                    
                    outputs = model(batched_graph, labels)
                    loss = outputs.get("loss", criterion(outputs["MMlogit"], labels))
                    
                    test_loss += loss.item()
                    preds = outputs["y_pred"]
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            
            print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}, "
                  f"F1={f1:.4f}, Pre={precision:.4f}, Rec={recall:.4f}")
            
            epoch_results.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), output_base / "best_model.pth")
                print(f"  -> New best F1: {best_f1:.4f} at epoch {best_epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            template_ids = batch["template_ids"].to(device)
            labels = batch["label"].to(device)
            batch_size = template_ids.shape[0]
            seq_len = template_ids.shape[1]
            
            # Create batched graph for log-only mode
            g = dgl.graph(([], []), num_nodes=template_ids.numel(), device=device)
            g.ndata['logs'] = template_ids.view(-1)
            g.set_batch_num_nodes([seq_len] * template_ids.shape[0])
            batched_graph = g  # Already on device
            
            outputs = model(batched_graph, labels)
            preds = outputs["y_pred"]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    final_f1 = f1_score(all_labels, all_preds, average='binary')
    final_pre = precision_score(all_labels, all_preds, average='binary')
    final_rec = recall_score(all_labels, all_preds, average='binary')
    
    results = {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_f1": final_f1,
        "final_precision": final_pre,
        "final_recall": final_rec,
        "epoch_results": epoch_results
    }
    
    with open(output_base / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Final F1: {final_f1:.4f}, Pre: {final_pre:.4f}, Rec: {final_rec:.4f}")
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    main()