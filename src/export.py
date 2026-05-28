"""
Model export and compilation module.
1. Exports best PyTorch checkpoint to ONNX format.
2. Runs onnx-simplifier on the ONNX graph to clean up structure and fold constants.
3. Compiles the simplified ONNX model to a TensorRT FP16 engine (requires TensorRT/trtexec).
4. Run a latency benchmark comparing PyTorch, ONNX CPU, and ONNX GPU backends.
"""
import os
import yaml
import argparse
import time
import subprocess
from pathlib import Path
import numpy as np

import torch
from src.models.classifier import AccessoryClassifier

def export_onnx(model_path, onnx_output_path):
    model = AccessoryClassifier()
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Extract state dict if saved inside a checkpoint dictionary
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting model to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print("ONNX export complete. Running simplification...")

    # Run onnx-simplifier
    try:
        import onnx
        from onnxsim import simplify
        
        onnx_model = onnx.load(onnx_output_path)
        simplified_model, check = simplify(onnx_model)
        
        if check:
            onnx.save(simplified_model, onnx_output_path)
            print("✓ ONNX model simplified successfully.")
        else:
            print("⚠️ ONNX simplification verification failed. Keeping standard export.")
    except Exception as e:
        print(f"⚠️ Failed to simplify ONNX model: {e}")

def compile_tensorrt(onnx_path, engine_output_path):
    """Compiles ONNX to a TensorRT engine using the CLI utility trtexec."""
    print(f"Compiling ONNX to TensorRT engine: {engine_output_path}...")
    
    # trtexec configuration flags
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_output_path}",
        "--fp16",  # Enable FP16 precision
        "--workspace=512", # Workspace allocation in MB
        "--minShapes=image:1x3x224x224",
        "--optShapes=image:8x3x224x224",
        "--maxShapes=image:32x3x224x224",
    ]
    
    try:
        # Check if trtexec CLI is available in PATH
        subprocess.run(["trtexec", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ TensorRT compilation complete.")
        print(result.stdout[-500:])  # Print summary lines from stdout log
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n[SKIP] TensorRT compilation skipped. 'trtexec' command not found in PATH.")
        print("To compile for GPU deployment, run this on a GPU system with TensorRT installed:")
        print(" " + " ".join(cmd))

def run_benchmark(model_path, onnx_path, n_runs=300, batch_size=1):
    print(f"\nBenchmarking inference latency (batch_size={batch_size}, {n_runs} runs)...")
    dummy_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    results = {}

    # PyTorch CPU
    model = AccessoryClassifier()
    state_dict = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            model(dummy_pt)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(dummy_pt)
        results["PyTorch CPU"] = (time.perf_counter() - t0) / n_runs * 1000

    # ONNX Runtime CPU
    try:
        import onnxruntime as ort
        sess_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        for _ in range(15):
            sess_cpu.run(None, {"image": dummy_np})
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess_cpu.run(None, {"image": dummy_np})
        results["ONNX Runtime CPU"] = (time.perf_counter() - t0) / n_runs * 1000
    except Exception as e:
        print(f"ONNX CPU benchmark failed: {e}")

    # ONNX Runtime GPU (CUDA)
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            sess_gpu = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
            for _ in range(15):
                sess_gpu.run(None, {"image": dummy_np})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                sess_gpu.run(None, {"image": dummy_np})
            results["ONNX Runtime GPU"] = (time.perf_counter() - t0) / n_runs * 1000
        else:
            results["ONNX Runtime GPU (Unavailable)"] = 0.0
    except Exception as e:
        print(f"ONNX GPU benchmark failed: {e}")

    # Print results table
    print("\n" + "=" * 42)
    print(f"{'Inference Backend':<25} | {'Latency (ms)':>12}")
    print("=" * 42)
    for backend, ms in results.items():
        if ms == 0.0:
            print(f"{backend:<25} | {'N/A':>12}")
        else:
            print(f"{backend:<25} | {ms:>10.2f} ms")
    print("=" * 42)
    print("Target performance check:")
    cpu_lat = results.get("ONNX Runtime CPU", 999.0)
    gpu_lat = results.get("ONNX Runtime GPU", 999.0)
    
    if cpu_lat <= 18.0:
        print("  ✓ ONNX CPU meets ≤18ms latency target.")
    else:
        print(f"  ⚠️ ONNX CPU does not meet ≤18ms target (current: {cpu_lat:.2f}ms).")
        
    if gpu_lat != 999.0:
        if gpu_lat <= 6.0:
            print("  ✓ ONNX GPU meets ≤6ms latency target.")
        else:
            print(f"  ⚠️ ONNX GPU does not meet ≤6ms target (current: {gpu_lat:.2f}ms).")

def main():
    parser = argparse.ArgumentParser(description="Export and compile trained model")
    parser.add_argument("--model", default="outputs/checkpoints/best.pth", help="Path to PyTorch best.pth checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        output_dir = Path(cfg.get("output_dir", "outputs"))
    else:
        output_dir = Path("outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "accessor_classifier.onnx"
    engine_path = output_dir / "accessor_classifier.engine"

    if not Path(args.model).exists():
        print(f"[ERROR] PyTorch model checkpoint not found at {args.model}. Run training script first.")
        return

    export_onnx(args.model, str(onnx_path))
    compile_tensorrt(str(onnx_path), str(engine_path))
    run_benchmark(args.model, str(onnx_path))

if __name__ == "__main__":
    main()
