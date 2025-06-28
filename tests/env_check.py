import os
import sys
import platform
import torch
import time

'''
From Root:> python -m tests.env_check
'''

class Benchmarks:
    NONE = -1
    GPU_ONLY = 1
    CPU_ONLY = 2
    GPU_CPU = 3

def parse_benchmark_args(args):
    if not args:
        return Benchmarks.NONE
    normalized = [a.lower() for a in args]
    if any(a in ["no", "none"] for a in normalized):
        return Benchmarks.NONE
    if "both" in normalized or "gpu_cpu" in normalized or ("gpu" in normalized and "cpu" in normalized):
        return Benchmarks.GPU_CPU
    if "cpu" in normalized:
        return Benchmarks.CPU_ONLY
    if "gpu" in normalized:
        return Benchmarks.GPU_ONLY
    return Benchmarks.NONE

def divider(title):
    print("\n" + "-" * 40)
    print(f"{title}")
    print("-" * 40)

def check_python():
    divider("🐍 Python")
    print("Python executable:", sys.executable)
    print("Python version:   ", platform.python_version())

def check_gpu():
    divider("🖥️ GPU (via PyTorch)")
    print("PyTorch version:   ", torch.__version__)
    print("CUDA available?    ", torch.cuda.is_available())
    print("CUDA version:      ", torch.version.cuda)
    print("cuDNN version:     ", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("Device count:      ", torch.cuda.device_count())
        for gpu_index in range(torch.cuda.device_count()):
            print(f" - GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
            print("Total memory (MB):", torch.cuda.get_device_properties(gpu_index).total_memory // (1024 ** 2))
            print("Allocated memory (MB):", torch.cuda.memory_allocated(gpu_index) // (1024 ** 2))
            print("Reserved memory (MB):", torch.cuda.memory_reserved(gpu_index) // (1024 ** 2))
            print("Max memory allocated (MB):", torch.cuda.max_memory_allocated(gpu_index) // (1024 ** 2))
            print("Max memory reserved (MB):", torch.cuda.max_memory_reserved(gpu_index) // (1024 ** 2))
        return True
    else:
        return False

def check_env():
    divider("🌱 Environment")
    print("Current working dir:", os.getcwd())
    print("Virtual env active: ", sys.prefix != sys.base_prefix)
    print("PYTHONPATH:         ", os.environ.get("PYTHONPATH", "(not set)"))

def check_repo():
    divider("📁 Repo Files (Top Level)")
    for item in os.listdir("."):
        if os.path.isdir(item):
            print(f"[DIR]  {item}")
        else:
            print(f"       {item}")

def benchmark( test_what ):
    match test_what:
        case Benchmarks.NONE:
            return False
        case Benchmarks.GPU_ONLY:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                print("Cannot test GPU, exiting")
                return False
        case Benchmarks.CPU_ONLY:
            device = "cpu"
        case Benchmarks.GPU_CPU:
            benchmark( test_what=Benchmarks.CPU_ONLY )
            return benchmark( test_what=Benchmarks.GPU_ONLY )
        
    divider(f"🚀 {device.upper()} Benchmark (Matrix Multiplication)")
    size = 8192
    runs = 20

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Size:               {size} x {size}")
    print(f"Iterations:         {runs}")
    print(f"Total time:         {end - start:.4f} seconds")
    print(f"Matmuls/sec:        {runs / (end - start):.2f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    benchmarks_to_do = parse_benchmark_args( args )
    check_python()
    gpu_bool = check_gpu()
    check_env()
    check_repo()
    benchmark( benchmarks_to_do )

    divider("✅ All checks done.")
    print("Ready to run tests or training scripts!")
