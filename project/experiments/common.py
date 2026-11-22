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

# Flush every line even when stdout is redirected to a file (nohup / log files on the server),
# so progress is visible with `tail -f` instead of appearing in large buffered blocks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except AttributeError:  # non-standard stream (e.g. some notebooks)
        pass

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
MODALITIES = ("metrics", "traces", "logs")


def fit_normalizer(chunks: Dict, ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-channel (last axis) mean/std of each modality over the given TRAINING chunks,
    pooled over samples, services and time steps. Computed in float64: the log features in
    recovered_artifact are ~6e-10 in magnitude and would lose precision in float32."""
    stats = {}
    for m in MODALITIES:
        x = np.stack([np.asarray(chunks[k][m], dtype=np.float64) for k in ids])
        x = x.reshape(-1, x.shape[-1])
        mean, std = x.mean(0), x.std(0)
        std = np.where(std > 0, std, 1.0)  # constant channel -> leave centred, unscaled
        stats[m] = {"mean": mean, "std": std}
    return stats


def apply_normalizer(chunks: Dict, ids: List[str], stats) -> Dict:
    """Return a new {id: chunk} dict with z-scored modalities (float32); labels untouched."""
    out = {}
    for k in ids:
        c = dict(chunks[k])
        for m in MODALITIES:
            x = np.asarray(c[m], dtype=np.float64)
            c[m] = ((x - stats[m]["mean"]) / stats[m]["std"]).astype(np.float32)
        out[k] = c
    return out


def _chunk_key(a) -> bytes:
    return np.round(np.asarray(a, dtype=np.float64), 5).tobytes()


def reconstruct_runs(chunks: Dict) -> List[List[str]]:
    """
    Recover the temporal order of the chunks. The artifact was produced by Eadro's preprocessing
    (vendor/eadro/codes/preprocess/align.py), which cuts each fault-injection run into 10-step
    windows with STRIDE 1, so window t+1's first 9 metric steps equal window t's last 9 steps.
    Chaining on that identity yields one ordered list of chunk ids per run (4 runs for
    recovered_artifact, verified unambiguous on 2026-11-17). Runs are returned in a
    deterministic order (sorted by their first chunk id); the true wall-clock order between runs
    is not recoverable and does not matter for a run-level split.
    """
    head = {}
    for k, c in chunks.items():
        head.setdefault(_chunk_key(c["metrics"][:, :-1, :]), []).append(k)
    nxt = {}
    for k, c in chunks.items():
        cand = head.get(_chunk_key(c["metrics"][:, 1:, :]), [])
        if len(cand) > 1:
            raise RuntimeError(f"Ambiguous successor for chunk {k}; cannot reconstruct runs")
        if cand:
            nxt[k] = cand[0]
    has_pred = set(nxt.values())
    runs = []
    for start in sorted(k for k in chunks if k not in has_pred):
        run, seen = [start], {start}
        while run[-1] in nxt and nxt[run[-1]] not in seen:
            run.append(nxt[run[-1]])
            seen.add(run[-1])
        runs.append(run)
    if sum(len(r) for r in runs) != len(chunks):
        raise RuntimeError("Run reconstruction did not cover every chunk (cycle or broken chain)")
    return runs


NORMAL_SEGMENTS = (("train", 0.0, 0.6), ("val", 0.6, 0.8), ("test", 0.8, 1.0))


def assign_pipeline_runs(runs: List[Dict], frac_eval: float = 0.2) -> Dict[str, List[int]]:
    """
    Chronological run-level split for data built by data_pipeline.build_eadro:
      fault runs sorted by start time -> the last ceil(frac_eval * n) are TEST, the
      ceil(frac_eval * n) before them are VAL, the rest TRAIN (no future information in training);
      SN (4 fault runs): 2 / 1 / 1, TT (9): 5 / 2 / 2.
      Fault-free runs (recorded days later): every split must contain purely-normal operation.
      With >= 3 of them (SN) they are assigned whole, round-robin train, test, val. With fewer (TT
      has 2) each one is listed under "segmented" and cut chronologically into train / val / test
      segments (NORMAL_SEGMENTS, see segment_normal_runs).
    """
    fault = sorted((r for r in runs if r["kind"] == "fault"), key=lambda r: r["t0"])
    normal = sorted((r for r in runs if r["kind"] == "normal"), key=lambda r: r["t0"])
    n = len(fault)
    k = max(1, int(np.ceil(frac_eval * n)))
    if n < 3:
        raise ValueError(f"need >= 3 fault runs for a train/val/test run split, got {n}")
    if n - 2 * k < 1:
        k = (n - 1) // 2
    out = {"train": [r["index"] for r in fault[: n - 2 * k]],
           "val": [r["index"] for r in fault[n - 2 * k: n - k]],
           "test": [r["index"] for r in fault[n - k:]], "segmented": []}
    if len(normal) >= 3:
        for i, r in enumerate(normal):
            out[("train", "test", "val")[i % 3]].append(r["index"])
    else:
        out["segmented"] = [r["index"] for r in normal]
    return out


def segment_normal_runs(chunks: Dict, runs: List[Dict], run_ids: List[int], window: int) -> Dict[str, List[str]]:
    """
    Cut each fault-free run in run_ids chronologically at 60% / 80% of its duration into
    train / val / test segments. A window is kept only if it lies entirely inside one segment;
    windows straddling a cut are dropped so no second of data is shared between splits.
    """
    out = {s: [] for s, _, _ in NORMAL_SEGMENTS}
    for ri in run_ids:
        r = runs[ri]
        for k, c in chunks.items():
            if c["run_index"] != ri:
                continue
            a = c["t0"] - r["t0"]
            for s, lo, hi in NORMAL_SEGMENTS:
                if a >= lo * r["seconds"] and a + window <= hi * r["seconds"]:
                    out[s].append(k)
                    break
    return out


def _load_pipeline_splits(data_dir: Path, batch_size: int, split_out: Optional[Path], normalize: str) -> Dict:
    """Load rebuilt Eadro or GAIA chunks with their dataset-specific split."""
    chunks, node_num, edges, metadata = load_data(str(data_dir / "chunks.pkl"), str(data_dir / "metadata.json"))
    if metadata.get("format") == "gaia_micross_v1":
        manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
        ids = {split: manifest.pop(f"{split}_ids") for split in ("train", "val", "test")}
        all_ids = ids["train"] + ids["val"] + ids["test"]
        if any(not subset for subset in ids.values()) or len(all_ids) != len(chunks) or set(all_ids) != chunks.keys():
            raise ValueError("GAIA split manifest must assign every chunk to exactly one nonempty partition")
        split_info = {"data_dir": str(data_dir), "system": metadata.get("system"), **manifest}
    else:
        assignment = assign_pipeline_runs(metadata["runs"])
        ids = {s: sorted(k for k, c in chunks.items() if c["run_index"] in set(assignment[s]))
               for s in ("train", "val", "test")}
        if assignment["segmented"]:
            seg = segment_normal_runs(chunks, metadata["runs"], assignment["segmented"], int(metadata["chunk_lenth"]))
            ids = {s: sorted(ids[s] + seg[s]) for s in ids}
        split_info = {"data_dir": str(data_dir), "split": "pipeline_runs", "system": metadata.get("system"),
                      "runs": {s: [metadata["runs"][i]["name"] + f" ({metadata['runs'][i]['kind']})"
                                   for i in assignment[s]] for s in assignment},
                      "normal_segments": {s: [lo, hi] for s, lo, hi in NORMAL_SEGMENTS}
                      if assignment["segmented"] else None}
    if split_out is not None:
        split_out = Path(split_out)
        split_out.parent.mkdir(parents=True, exist_ok=True)
        with open(split_out, "w") as f:
            json.dump({**split_info, **{f"{s}_ids": v for s, v in ids.items()}}, f)
    if normalize == "train_zscore":
        stats = fit_normalizer(chunks, ids["train"])
        src = apply_normalizer(chunks, ids["train"] + ids["val"] + ids["test"], stats)
        if split_out is not None:
            with open(Path(split_out).with_name("normalization.json"), "w") as f:
                json.dump({"method": normalize,
                           "stats": {m: {k: v.tolist() for k, v in st.items()} for m, st in stats.items()}},
                          f, indent=2)
    else:
        src = chunks
    loaders = {}
    for s in ("train", "val", "test"):
        ds = ChunkDataset(src, ids[s], node_num, edges)
        loaders[s] = create_dataloader(ds, batch_size=batch_size, shuffle=(s == "train"))

    def _dist(keys):
        vals, counts = np.unique([chunks[k]["culprit"] for k in keys], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    return {**loaders, "metadata": metadata, "sizes": {s: len(v) for s, v in ids.items()},
            "normalization": {"method": normalize}, "split": split_info,
            "culprit_dist": {s: _dist(v) for s, v in ids.items()}}


def load_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = 50,
    val_ratio: float = DEFAULT_VAL_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_out: Optional[Path] = None,
    normalize: str = "train_zscore",
    split: str = "run",
    test_run: int = 3,
    val_run: int = 2,
) -> Dict:
    """
    Build train/val/test loaders.

    split:
      "run" (default) - leak-free run-level split. All chunks (chunk_train.pkl + chunk_test.pkl)
                        are re-ordered into their original fault-injection runs
                        (reconstruct_runs); run `test_run` is the test set, run `val_run` the
                        validation set, the remaining runs are training. No window overlaps
                        across splits, and the test run's faults hit services unseen in
                        training. Required by implementation guide section 13.
      "random"        - the artifact's own split (chunk_test.pkl as test) with a stratified
                        `val_ratio` validation subset of chunk_train.pkl. LEAKY: windows have
                        stride 1, so 83.9% of test chunks have a 1-second-shifted (90% identical)
                        neighbour in train. Kept only to reproduce runs before 2026-11-17 night.

    normalize:
      "train_zscore" (default) - per-channel z-score of metrics / traces / logs with statistics
                                 fitted on the TRAIN split only (implementation guide step 5 and
                                 section 13), applied identically to train / val / test and to
                                 every model. Needed because the raw log features are ~6e-10 and
                                 numerically invisible to the networks (PROJECT_STATUS.md 3.5).
      "none"                   - raw artifact values.

    Labels: culprit 0 = normal, 1 = cpu_load, 4 = network_delay, 5 = network_loss (decoded from
    the raw SN fault files); models use culprit >= 1 as "anomalous".
    """
    if normalize not in ("train_zscore", "none"):
        raise ValueError(f"normalize must be 'train_zscore' or 'none', got {normalize!r}")
    if split not in ("run", "random"):
        raise ValueError(f"split must be 'run' or 'random', got {split!r}")
    data_dir = Path(data_dir)
    if (data_dir / "chunks.pkl").exists():
        return _load_pipeline_splits(data_dir, batch_size, split_out, normalize)
    train_all, node_num, edges, metadata = load_data(
        str(data_dir / "chunk_train.pkl"), str(data_dir / "metadata.json")
    )
    test_data, _, _, _ = load_data(str(data_dir / "chunk_test.pkl"), str(data_dir / "metadata.json"))
    pool = {**train_all, **test_data}
    if len(pool) != len(train_all) + len(test_data):
        raise RuntimeError("chunk ids collide between chunk_train.pkl and chunk_test.pkl")

    split_info = {"data_dir": str(data_dir), "split": split}
    if split == "run":
        runs = reconstruct_runs(pool)
        n = len(runs)
        if not (0 <= test_run < n and 0 <= val_run < n and test_run != val_run):
            raise ValueError(f"need distinct test_run/val_run in [0, {n}), got {test_run}/{val_run}")
        test_ids = list(runs[test_run])
        val_ids = list(runs[val_run])
        train_ids = [k for i, r in enumerate(runs) if i not in (test_run, val_run) for k in r]
        split_info.update(n_runs=n, run_lengths=[len(r) for r in runs], test_run=test_run,
                          val_run=val_run, runs=runs)
    else:
        all_ids = sorted(train_all.keys())
        culprits = [train_all[k]["culprit"] for k in all_ids]
        train_ids, val_ids = train_test_split(
            all_ids, test_size=val_ratio, random_state=split_seed, stratify=culprits
        )
        test_ids = sorted(test_data.keys())
        split_info.update(split_seed=split_seed, val_ratio=val_ratio)

    if split_out is not None:
        split_out = Path(split_out)
        split_out.parent.mkdir(parents=True, exist_ok=True)
        with open(split_out, "w") as f:
            json.dump({**split_info, "train_ids": sorted(train_ids), "val_ids": sorted(val_ids),
                       "test_ids": sorted(test_ids)}, f)

    norm_info = {"method": normalize}
    if normalize == "train_zscore":
        stats = fit_normalizer(pool, train_ids)  # train split only; val/test never seen
        src = apply_normalizer(pool, train_ids + val_ids + test_ids, stats)
        norm_info["stats"] = {m: {k: v.tolist() for k, v in st.items()} for m, st in stats.items()}
        if split_out is not None:
            with open(Path(split_out).with_name("normalization.json"), "w") as f:
                json.dump(norm_info, f, indent=2)
    else:
        src = pool
    train_src = test_src = src

    train_ds = ChunkDataset(train_src, train_ids, node_num, edges)
    val_ds = ChunkDataset(train_src, val_ids, node_num, edges)
    test_ds = ChunkDataset(test_src, test_ids, node_num, edges)

    def _dist(chunks, ids):
        vals, counts = np.unique([chunks[k]["culprit"] for k in ids], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    return {
        "train": create_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        "val": create_dataloader(val_ds, batch_size=batch_size, shuffle=False),
        "test": create_dataloader(test_ds, batch_size=batch_size, shuffle=False),
        "metadata": metadata,
        "sizes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "normalization": {"method": normalize},
        "split": {k: v for k, v in split_info.items() if k != "runs"},
        "culprit_dist": {
            "train": _dist(pool, train_ids),
            "val": _dist(pool, val_ids),
            "test": _dist(pool, test_ids),
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
    y_true, y_pred, p_anom, culprit = [], [], [], []
    total_loss, n_batches = 0.0, 0
    for graph, labels in loader:
        graph, labels = graph.to(device), labels.to(device)
        logits, loss = forward_fn(model, graph, labels)
        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        culprit.extend(labels.long().cpu().tolist())
        y_true.extend(to_binary(labels).cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        p_anom.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, p_anom)
    metrics["loss"] = total_loss / max(n_batches, 1)
    # Per-fault-type detection rate (recall_type<c>, c = raw culprit id >= 1) and the false-positive
    # rate on normal windows: shows which fault types a model misses.
    culprit, y_pred = np.asarray(culprit), np.asarray(y_pred)
    for c in sorted(set(culprit.tolist()) - {0}):
        metrics[f"recall_type{c}"] = float(np.mean(y_pred[culprit == c] == 1))
    if (culprit == 0).any():
        metrics["fpr_normal"] = float(np.mean(y_pred[culprit == 0] == 1))
    return metrics


def _selection_key(m: Dict[str, float]) -> Tuple[float, float]:
    """
    Model selection on validation: macro-F1, ties broken by PR-AUC (guide §: "best checkpoint based
    on validation F1 or PR-AUC"). Not anomaly-class F1: on an anomaly-heavy validation split the
    trivial all-anomalous predictor scores F1 ~0.9 and wins selection (round 5 TT, PROJECT_STATUS),
    whereas its macro-F1 is <= 0.5.
    """
    pr = m["pr_auc"] if np.isfinite(m["pr_auc"]) else 0.0
    return (m["macro_f1"], pr)


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
