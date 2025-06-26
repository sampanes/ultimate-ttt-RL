import os
import sys
import platform
import torch

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
        for i in range(torch.cuda.device_count()):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

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

if __name__ == "__main__":
    check_python()
    check_gpu()
    check_env()
    check_repo()

    divider("✅ All checks done.")
    print("Ready to run tests or training scripts!")
