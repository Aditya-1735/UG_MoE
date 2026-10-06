#!/usr/bin/env python3
"""
Environment Validation Script for MUAD/Graph-Aware MoE Project

Verifies that all imports used by the project actually work.
Reports: PASS / WARNING / FAIL for each import.
"""

import sys
import traceback

class ValidationResult:
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"

results = []

def check(name, import_func, required=True, description=""):
    """Check an import and record result."""
    try:
        import_func()
        status = ValidationResult.PASS
        msg = f"[PASS] {name}"
    except ImportError as e:
        status = ValidationResult.FAIL if required else ValidationResult.WARNING
        msg = f"{'[FAIL]' if required else '[WARN]'} {name}: {e}"
    except Exception as e:
        status = ValidationResult.FAIL if required else ValidationResult.WARNING
        msg = f"{'[FAIL]' if required else '[WARN]'} {name}: {type(e).__name__}: {e}"
    
    results.append((status, name, msg, description))
    print(msg)
    return status == ValidationResult.PASS

def check_callable(name, func, required=True, description=""):
    """Check a callable (e.g., module.function) works."""
    try:
        func()
        status = ValidationResult.PASS
        msg = f"[PASS] {name}"
    except Exception as e:
        status = ValidationResult.FAIL if required else ValidationResult.WARNING
        msg = f"{'[FAIL]' if required else '[WARN]'} {name}: {type(e).__name__}: {e}"
    
    results.append((status, name, msg, description))
    print(msg)
    return status == ValidationResult.PASS

def main():
    print("=" * 70)
    print("ENVIRONMENT VALIDATION - MUAD/Graph-Aware MoE")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print()

    # ============================================================
    # CORE ML FRAMEWORK
    # ============================================================
    print("--- Core ML Framework ---")
    check("torch", lambda: __import__("torch"), required=True, 
          description="Primary tensor computation framework")
    
    check("torch.cuda", lambda: __import__("torch").cuda, required=False,
          description="CUDA support (optional but recommended)")
    
    check("dgl", lambda: __import__("dgl"), required=True,
          description="Graph neural network library")
    
    check("dgl.nn.pytorch", lambda: __import__("dgl.nn.pytorch", fromlist=["GATv2Conv", "GlobalAttentionPooling"]), required=True,
          description="DGL PyTorch modules (GATv2Conv, GlobalAttentionPooling)")
    
    check("dgl.dataloading", lambda: __import__("dgl.dataloading"), required=False,
          description="DGL dataloading utilities")

    # ============================================================
    # DATA SCIENCE STACK
    # ============================================================
    print("\n--- Data Science Stack ---")
    check("numpy", lambda: __import__("numpy"), required=True,
          description="Numerical computation")
    
    check("pandas", lambda: __import__("pandas"), required=True,
          description="Data processing (preprocessing scripts)")
    
    check("scipy", lambda: __import__("scipy"), required=True,
          description="Scientific computing")
    
    check("sklearn", lambda: __import__("sklearn"), required=True,
          description="Machine learning utilities (metrics, etc.)")
    
    check("sklearn.metrics", lambda: __import__("sklearn.metrics", fromlist=["f1_score", "precision_score", "recall_score"]), required=True,
          description="Classification metrics")

    # ============================================================
    # UTILITIES
    # ============================================================
    print("\n--- Utilities ---")
    check("tqdm", lambda: __import__("tqdm"), required=True,
          description="Progress bars")
    
    check("yaml", lambda: __import__("yaml"), required=True,
          description="YAML config parsing (PyYAML)")
    
    check("networkx", lambda: __import__("networkx"), required=True,
          description="Graph utilities (vendor/eadro)")

    # ============================================================
    # PREPROCESSING DEPENDENCIES
    # ============================================================
    print("\n--- Preprocessing Dependencies ---")
    check("drain3", lambda: __import__("drain3"), required=True,
          description="Log template mining (convert_logs.py)")
    
    check("drain3.TemplateMiner", lambda: __import__("drain3", fromlist=["TemplateMiner"]), required=True,
          description="Drain3 TemplateMiner class")
    
    check("drain3.file_persistence", lambda: __import__("drain3.file_persistence", fromlist=["FilePersistence"]), required=True,
          description="Drain3 file persistence")
    
    check("tick", lambda: __import__("tick"), required=True,
          description="Hawkes process for log features (convert_traces.py via Eadro)")
    
    check("tick.hawkes", lambda: __import__("tick.hawkes", fromlist=["HawkesADM4"]), required=True,
          description="HawkesADM4 for log Hawkes process")

    # ============================================================
    # PROJECT MODULES
    # ============================================================
    print("\n--- Project Modules (src/) ---")
    
    # Add project root to path
    import os
    # Check common locations
    possible_roots = [
        "/content/project",  # Colab
        "/kaggle/working/project",  # Kaggle
        "project",  # Local: running from repo root
        ".",  # Local: running from project directory
    ]
    project_root = None
    for root in possible_roots:
        if os.path.exists(os.path.join(root, "src", "experts")):
            project_root = root
            break
    if project_root is None:
        project_root = "project"  # Default assumption
    
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "vendor/muad"))
    sys.path.insert(0, os.path.join(project_root, "src"))

    check("src.experts", lambda: __import__("src.experts"), required=True,
          description="Specialized experts package")
    
    check("src.experts.temporal_expert", lambda: __import__("src.experts.temporal_expert", fromlist=["TemporalExpert", "TemporalExpertGraph"]), required=True,
          description="Temporal expert (GRU/LSTM/Transformer)")
    
    check("src.experts.semantic_expert", lambda: __import__("src.experts.semantic_expert", fromlist=["SemanticExpert", "LogSemanticExpert"]), required=True,
          description="Semantic expert (log/event semantics)")
    
    check("src.experts.dependency_expert", lambda: __import__("src.experts.dependency_expert", fromlist=["DependencyExpert", "GATDependencyExpert"]), required=True,
          description="Dependency expert (GAT/Graph Transformer)")
    
    check("src.experts.cross_modal_expert", lambda: __import__("src.experts.cross_modal_expert", fromlist=["CrossModalExpert", "CrossModalExpertGraph"]), required=True,
          description="Cross-modal expert (multi-modal fusion)")

    check("src.routing", lambda: __import__("src.routing"), required=True,
          description="Expert routing package")
    
    check("src.routing.router", lambda: __import__("src.routing.router", fromlist=["ExpertRouter", "create_router"]), required=True,
          description="Expert router (graph-aware, uncertainty-guided)")

    check("src.uncertainty", lambda: __import__("src.uncertainty"), required=True,
          description="Uncertainty modeling package")
    
    check("src.uncertainty.uncertainty", lambda: __import__("src.uncertainty.uncertainty", fromlist=["UncertaintyBlock", "VariationalUncertaintyBlock", "compute_predictive_uncertainty"]), required=True,
          description="Uncertainty blocks (variational, KL divergence)")

    check("src.fusion", lambda: __import__("src.fusion"), required=True,
          description="Adaptive fusion package")
    
    check("src.fusion.fusion", lambda: __import__("src.fusion.fusion", fromlist=["AdaptiveFusion", "DynamicFusion", "ModalityDropout"]), required=True,
          description="Fusion modules (confidence-weighted, dynamic)")

    # ============================================================
    # VENDOR MUAD MODULES
    # ============================================================
    print("\n--- Vendor MUAD Modules ---")
    check("vendor.muad.data", lambda: __import__("vendor.muad.data"), required=True,
          description="MUAD data utilities")
    
    check("vendor.muad.data.utils", lambda: __import__("vendor.muad.data.utils", fromlist=["load_data", "create_dataloader", "collate"]), required=True,
          description="MUAD data loading and collation")
    
    check("vendor.muad.data.dataset", lambda: __import__("vendor.muad.data.dataset", fromlist=["ChunkDataset"]), required=True,
          description="MUAD ChunkDataset")
    
    check("vendor.muad.models", lambda: __import__("vendor.muad.models"), required=True,
          description="MUAD models package")
    
    check("vendor.muad.models.main_model", lambda: __import__("vendor.muad.models.main_model", fromlist=["MainModel"]), required=True,
          description="MUAD MainModel (baseline)")
    
    check("vendor.muad.models.encoders", lambda: __import__("vendor.muad.models.encoders", fromlist=["MetricEncoder", "TraceEncoder", "LogEncoder"]), required=True,
          description="MUAD modality encoders")
    
    check("vendor.muad.models.graph_model", lambda: __import__("vendor.muad.models.graph_model", fromlist=["GraphModel1"]), required=True,
          description="MUAD graph model (GAT)")
    
    check("vendor.muad.training.base_model", lambda: __import__("vendor.muad.training.base_model", fromlist=["BaseModel"]), required=True,
          description="MUAD BaseModel trainer")
    
    check("vendor.muad.utils.general_utils", lambda: __import__("vendor.muad.utils.general_utils", fromlist=["seed_everything"]), required=True,
          description="MUAD utilities (seed_everything)")
    
    check("vendor.muad.active_learning.strategies", lambda: __import__("vendor.muad.active_learning.strategies", fromlist=["hybrid_selection", "entropy_based_selection", "confidence_based_selection"]), required=False,
          description="Active learning strategies (Track A only)")

    # ============================================================
    # TRACK B MODEL
    # ============================================================
    print("\n--- Track B: Graph-Aware MoE ---")
    check("tracks.proposed_model", lambda: __import__("tracks.proposed_model"), required=True,
          description="Proposed model package")
    
    check("tracks.proposed_model.model", lambda: __import__("tracks.proposed_model.model", fromlist=["GraphAwareMoE", "create_graph_aware_moe", "DEFAULT_CONFIG"]), required=True,
          description="GraphAwareMoE model and factory")

    # ============================================================
    # FUNCTIONAL TESTS
    # ============================================================
    print("\n--- Functional Tests ---")
    
    # Test torch tensor creation
    check_callable("torch.tensor creation", lambda: __import__("torch").tensor([1,2,3]), required=True,
                   description="Basic tensor creation")
    
    # Test torch.nn
    check("torch.nn.Module class", lambda: getattr(__import__("torch.nn", fromlist=["Module"]), "Module"), required=True,
                  description="Neural network module base class")
    
    # Test DGL graph creation
    check_callable("dgl.graph", lambda: __import__("dgl").graph(([0,1], [1,2])), required=True,
                   description="DGL graph creation")
    
    # Test DGL batching
    check_callable("dgl.batch", lambda: __import__("dgl").batch([__import__("dgl").graph(([0],[1])), __import__("dgl").graph(([0],[1]))]), required=True,
                   description="DGL graph batching")
    
    # Test Model instantiation (lightweight)
    try:
        from tracks.proposed_model import GraphAwareMoE, DEFAULT_CONFIG
        model = GraphAwareMoE(**{k:v for k,v in DEFAULT_CONFIG.items() if k != 'epochs'})
        param_count = sum(p.numel() for p in model.parameters())
        check_callable(f"GraphAwareMoE instantiation ({param_count:,} params)", lambda: None, required=True,
                       description="Full model instantiation test")
    except Exception as e:
        results.append((ValidationResult.FAIL, "GraphAwareMoE instantiation", f"[FAIL] GraphAwareMoE instantiation: {e}", "Full model test"))
        print(f"[FAIL] GraphAwareMoE instantiation: {e}")

    # Test MUAD baseline instantiation
    try:
        from vendor.muad.training.base_model import BaseModel
        import torch
        device = torch.device("cpu")
        baseline = BaseModel(device, lr=0.001, epochs=1, patience=1, result_dir="/tmp", hash_id="test", hidden_dim=[64,64], event_num=14)
        check_callable("MUAD BaseModel instantiation", lambda: None, required=True,
                       description="MUAD baseline model instantiation")
    except Exception as e:
        results.append((ValidationResult.FAIL, "MUAD BaseModel instantiation", f"[FAIL] MUAD BaseModel: {e}", "MUAD baseline test"))
        print(f"[FAIL] MUAD BaseModel instantiation: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    pass_count = sum(1 for r in results if r[0] == ValidationResult.PASS)
    warn_count = sum(1 for r in results if r[0] == ValidationResult.WARNING)
    fail_count = sum(1 for r in results if r[0] == ValidationResult.FAIL)
    
    print(f"Total checks: {len(results)}")
    print(f"[PASS] PASS:  {pass_count}")
    print(f"[WARN] WARN:  {warn_count}")
    print(f"[FAIL] FAIL:  {fail_count}")
    
    if fail_count > 0:
        print("\n--- FAILED CHECKS ---")
        for status, name, msg, desc in results:
            if status == ValidationResult.FAIL:
                print(f"  [FAIL] {name}")
                if desc:
                    print(f"      Purpose: {desc}")
    
    if warn_count > 0:
        print("\n--- WARNINGS ---")
        for status, name, msg, desc in results:
            if status == ValidationResult.WARNING:
                print(f"  [WARN] {name}")
                if desc:
                    print(f"      Purpose: {desc}")
    
    print("\n" + "=" * 70)
    if fail_count == 0:
        print("[OK] ALL REQUIRED CHECKS PASSED - Environment ready!")
        return 0
    else:
        print(f"[FAIL] {fail_count} REQUIRED CHECK(S) FAILED - Fix before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())