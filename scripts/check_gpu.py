#!/usr/bin/env python3
"""
GPU Diagnostic Script for MUAD/Graph-Aware MoE Project

Reports:
- Python version
- OS
- PyTorch version
- CUDA availability
- CUDA version
- GPU name
- GPU count
- GPU memory
- DGL version
- Other relevant framework versions

Fails gracefully if PyTorch/TensorFlow/JAX is not installed.
"""

import sys
import platform
import subprocess

def run_cmd(cmd):
    """Run command and return output or None on failure."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def check_pytorch():
    """Check PyTorch installation and CUDA."""
    print("\n=== PyTorch ===")
    try:
        import torch
        print(f"Version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / (1024**3)
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Total memory: {mem_total:.2f} GB")
                print(f"    Allocated: {mem_allocated:.2f} GB")
                print(f"    Reserved: {mem_reserved:.2f} GB")
        else:
            print("CUDA: NOT AVAILABLE (CPU only)")
            
        # Test tensor operations
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        print(f"Tensor ops test: PASS (matmul {x.shape} @ {y.shape} = {z.shape})")
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            z = torch.mm(x, y)
            print(f"CUDA tensor ops test: PASS")
            
        return True
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"PyTorch error: {e}")
        return False

def check_dgl():
    """Check DGL installation."""
    print("\n=== DGL (Deep Graph Library) ===")
    try:
        import dgl
        print(f"Version: {dgl.__version__}")
        print(f"Backend: {dgl.backend.backend_name}")
        
        # Test basic graph operations
        import torch
        g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        g.ndata['feat'] = torch.randn(4, 16)
        print(f"Graph ops test: PASS (nodes={g.num_nodes()}, edges={g.num_edges()})")
        
        if torch.cuda.is_available():
            g = g.to('cuda')
            print(f"GPU graph test: PASS")
            
        return True
    except ImportError:
        print("DGL: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"DGL error: {e}")
        return False

def check_numpy_scipy():
    """Check NumPy and SciPy."""
    print("\n=== NumPy / SciPy ===")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: NOT INSTALLED")
        
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except ImportError:
        print("SciPy: NOT INSTALLED")

def check_sklearn():
    """Check scikit-learn."""
    print("\n=== scikit-learn ===")
    try:
        import sklearn
        print(f"Version: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn: NOT INSTALLED")

def check_preprocessing_deps():
    """Check preprocessing dependencies."""
    print("\n=== Preprocessing Dependencies ===")
    
    for pkg in ['drain3', 'tick', 'pandas', 'pyyaml', 'networkx', 'tqdm']:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{pkg}: {version}")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

def check_nvidia_smi():
    """Run nvidia-smi for detailed GPU info."""
    print("\n=== nvidia-smi ===")
    output = run_cmd("nvidia-smi")
    if output:
        print(output)
    else:
        print("nvidia-smi not available or no GPU")

def check_nvcc():
    """Check CUDA compiler version."""
    print("\n=== NVCC ===")
    output = run_cmd("nvcc --version")
    if output:
        print(output)
    else:
        print("nvcc not available")

def main():
    print("=" * 60)
    print("GPU DIAGNOSTIC - MUAD/Graph-Aware MoE")
    print("=" * 60)
    
    # System info
    print(f"\nPython: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Core ML frameworks
    pytorch_ok = check_pytorch()
    dgl_ok = check_dgl()
    check_numpy_scipy()
    check_sklearn()
    check_preprocessing_deps()
    
    # Hardware info
    check_nvidia_smi()
    check_nvcc()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pytorch_ok and dgl_ok:
        print("[OK] Core ML stack: READY")
    else:
        print("[FAIL] Core ML stack: INCOMPLETE")
        if not pytorch_ok:
            print("   - PyTorch missing or broken")
        if not dgl_ok:
            print("   - DGL missing or broken")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU acceleration: AVAILABLE ({torch.cuda.device_count()} GPU(s))")
        else:
            print("[WARN] GPU acceleration: NOT AVAILABLE (CPU only)")
    except:
        print("[FAIL] GPU acceleration: UNKNOWN")
    
    print("\nFor this project:")
    print("- GPU is OPTIONAL but recommended for faster training")
    print("- Minimum: 4GB VRAM (8GB+ recommended)")
    print("- PyTorch 2.2.x + DGL 2.2.x with matching CUDA required")

if __name__ == "__main__":
    main()