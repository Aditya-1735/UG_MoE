"""
Shared experiment protocol: one train/val/test split, one training loop, one set of metrics.

Every experiment script should go through this module so that all models are compared under
the same protocol (implementation guide section 6, step 14, and section 13):
  * a validation set is carved out of the training chunks (stratified by culprit, fixed
    ``split_seed`` shared by every model and every training seed);
  * the checkpoint is selected on VALIDATION metrics only;
  * the test set is evaluated exactly once, with the selected checkpoint.

Label convention (inherited from MUAD, vendor/muad/models/main_model.py): ``culprit == 0`` is
normal, ``culprit >= 1`` is anomalous. Models receive the raw culprit labels and binarize them
internally; metrics are computed on the binarized labels.
"""
import copy
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "vendor" / "muad", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vendor.muad.data.dataset import ChunkDataset  # noqa: E402
from vendor.muad.data.utils import create_dataloader, load_data  # noqa: E402
from vendor.muad.utils.general_utils import seed_everything  # noqa: E402

DEFAULT_DATA_DIR = PROJECT_ROOT / "recovered_artifact"
DEFAULT_SPLIT_SEED = 2026
DEFAULT_VAL_RATIO = 0.15


def configure_torch() -> None:
    """Disable flash / mem-efficient SDPA (known NaN gradients in Transformer attention)."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def to_binary(labels: torch.Tensor) -> torch.Tensor:
    return (labels >= 1).long()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = 50,
    val_ratio: float = DEFAULT_VAL_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_out: Optional[Path] = None,
) -> Dict:
    """
    Build train/val/test loaders. The val set is a stratified (by culprit) subset of
    chunk_train.pkl; chunk_test.pkl is untouched and used only for the final evaluation.

    Note: chunk ids carry no timestamps, so a time-ordered split is not possible with this
    artifact; the split is stratified-random at chunk (window) level.
    """
    data_dir = Path(data_dir)
    train_all, node_num, edges, metadata = load_data(
        str(data_dir / "chunk_train.pkl"), str(data_dir / "metadata.json")
    )
    test_data, _, _, _ = load_data(str(data_dir / "chunk_test.pkl"), str(data_dir / "metadata.json"))

    all_ids = sorted(train_all.keys())
    culprits = [train_all[k]["culprit"] for k in all_ids]
    train_ids, val_ids = train_test_split(
        all_ids, test_size=val_ratio, random_state=split_seed, stratify=culprits
    )
    test_ids = sorted(test_data.keys())

    if split_out is not None:
        split_out = Path(split_out)
        split_out.parent.mkdir(parents=True, exist_ok=True)
        with open(split_out, "w") as f:
            json.dump(
                {
                    "data_dir": str(data_dir),
                    "split_seed": split_seed,
                    "val_ratio": val_ratio,
                    "train_ids": sorted(train_ids),
                    "val_ids": sorted(val_ids),
                    "test_ids": test_ids,
                },
                f,
            )

    train_ds = ChunkDataset(train_all, train_ids, node_num, edges)
    val_ds = ChunkDataset(train_all, val_ids, node_num, edges)
    test_ds = ChunkDataset(test_data, test_ids, node_num, edges)

    def _dist(chunks, ids):
        vals, counts = np.unique([chunks[k]["culprit"] for k in ids], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    return {
        "train": create_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        "val": create_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        "test": create_dataloader(test_ds, batch_size=batch_size, shuffle=False),
        "metadata": metadata,
        "sizes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "culprit_dist": {
            "train": _dist(train_all, train_ids),
            "val": _dist(train_all, val_ids),
            "test": _dist(test_data, test_ids),
        },
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: List[int], y_pred: List[int], p_anomaly: List[float]) -> Dict[str, float]:
    """Anomaly = positive class. Also reports normal-class and macro F1 (test set is ~83% anomalous)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_anomaly = np.asarray(p_anomaly)
    single_class = len(np.unique(y_true)) < 2
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, p_anomaly)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, p_anomaly)),
    }


def trivial_baseline_metrics(loader) -> Dict[str, float]:
    """Metrics of a constant 'always anomalous' predictor, as a floor for every table."""
    y = []
    for _, labels in loader:
        y.extend(to_binary(labels).tolist())
    return compute_metrics(y, [1] * len(y), [1.0] * len(y))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
# A forward function maps (model, graph, raw_labels) -> (logits [B, 2], loss scalar or None).
ForwardFn = Callable[[torch.nn.Module, object, torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]


def default_forward(model, graph, labels):
    """Works for GraphAwareMoE (tracks/proposed_model) and MUAD MainModel: both return
    {'MMlogit', 'loss'} and binarize raw culprit labels internally."""
    out = model(graph, labels)
    return out["MMlogit"], out.get("loss")


@torch.no_grad()
def evaluate(model, loader, device, forward_fn: ForwardFn = default_forward) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, p_anom = [], [], []
    total_loss, n_batches = 0.0, 0
    for graph, labels in loader:
        graph, labels = graph.to(device), labels.to(device)
        logits, loss = forward_fn(model, graph, labels)
        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        y_true.extend(to_binary(labels).cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        p_anom.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, p_anom)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def _selection_key(m: Dict[str, float]) -> Tuple[float, float]:
    """Model selection on validation: F1 (anomaly class), ties broken by PR-AUC."""
    pr = m["pr_auc"] if np.isfinite(m["pr_auc"]) else 0.0
    return (m["f1"], pr)


def train_and_select(
    model: torch.nn.Module,
    loaders: Dict,
    device,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    patience: Optional[int] = 15,
    ckpt_path: Optional[Path] = None,
    forward_fn: ForwardFn = default_forward,
    log_prefix: str = "",
) -> Dict:
    """
    Train on loaders['train'], select the epoch with the best validation score, restore it,
    and evaluate loaders['test'] once. Test metrics are never looked at during training.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_key, best_state, best_epoch = None, None, 0
    history, epochs_since_best = [], 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_ok, n_skipped = 0.0, 0, 0
        for graph, labels in loaders["train"]:
            graph, labels = graph.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, loss = forward_fn(model, graph, labels)
            if not torch.isfinite(logits).all() or loss is None or not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item()
            n_ok += 1

        val = evaluate(model, loaders["val"], device, forward_fn)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_ok, 1),
            "skipped_batches": n_skipped,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(record)
        print(
            f"{log_prefix}Epoch {epoch:3d}: train_loss={record['train_loss']:.4f} "
            f"val_loss={val['loss']:.4f} val_F1={val['f1']:.4f} val_macroF1={val['macro_f1']:.4f} "
            f"val_PR-AUC={val['pr_auc']:.4f}" + (f" [skipped {n_skipped} NaN batches]" if n_skipped else "")
        )

        key = _selection_key(val)
        if best_key is None or key > best_key:
            best_key, best_epoch, epochs_since_best = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                print(f"{log_prefix}Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)

    val_best = evaluate(model, loaders["val"], device, forward_fn)
    test = evaluate(model, loaders["test"], device, forward_fn)
    print(
        f"{log_prefix}Selected epoch {best_epoch}: val_F1={val_best['f1']:.4f} | "
        f"TEST F1={test['f1']:.4f} macroF1={test['macro_f1']:.4f} F1_normal={test['f1_normal']:.4f} "
        f"PR-AUC={test['pr_auc']:.4f} ROC-AUC={test['roc_auc']:.4f}"
    )
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val": val_best,
        "test": test,
        "history": history,
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
    }


def summarize(results: List[Dict], split: str = "test") -> Dict[str, Dict[str, float]]:
    """Mean/std over seeds for every metric of the given split."""
    if not results:
        return {}
    keys = results[0][split].keys()
    out = {}
    for k in keys:
        vals = np.array([r[split][k] for r in results], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "values": vals.tolist()}
    return out


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_DATA_DIR",
    "configure_torch",
    "seed_everything",
    "load_splits",
    "compute_metrics",
    "trivial_baseline_metrics",
    "evaluate",
    "train_and_select",
    "summarize",
    "default_forward",
]
