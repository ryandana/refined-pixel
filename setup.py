#!/usr/bin/env python3
"""
Cross-platform setup script for AI Image Processor
Handles installation of dependencies and model downloads
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json
import hashlib
import time

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.ENDC):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def get_system_info():
    """Get system information"""
    system = platform.system().lower()
    architecture = platform.machine().lower()
    python_version = sys.version_info
    
    return {
        'system': system,
        'architecture': architecture,
        'python_version': python_version,
        'is_windows': system == 'windows',
        'is_macos': system == 'darwin',
        'is_linux': system == 'linux'
    }

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print_colored("❌ Python 3.8 or higher is required!", Colors.FAIL)
        print_colored(f"Current version: {sys.version}", Colors.WARNING)
        return False
    
    print_colored(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected", Colors.OKGREEN)
    return True

def run_command(command, description, check=True, show_output=False):
    """Run a system command with error handling"""
    print_colored(f"🔄 {description}...", Colors.OKCYAN)
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=False, 
                                 capture_output=True, text=True, timeout=300)
        else:
            result = subprocess.run(command, check=False, 
                                 capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print_colored(f"✅ {description} completed successfully", Colors.OKGREEN)
            if show_output and result.stdout:
                print_colored(f"Output: {result.stdout.strip()}", Colors.OKBLUE)
            return True
        else:
            print_colored(f"❌ {description} failed (exit code: {result.returncode})", Colors.FAIL)
            if result.stderr:
                print_colored(f"Error: {result.stderr.strip()}", Colors.FAIL)
            if result.stdout:
                print_colored(f"Output: {result.stdout.strip()}", Colors.WARNING)
            
            if not check:
                print_colored(f"⚠️  Continuing despite error in {description}", Colors.WARNING)
                return True
            return False
            
    except subprocess.TimeoutExpired:
        print_colored(f"❌ {description} timed out after 5 minutes", Colors.FAIL)
        return False
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ {description} failed with error: {e}", Colors.FAIL)
        return False
    except Exception as e:
        print_colored(f"❌ Unexpected error during {description}: {e}", Colors.FAIL)
        return False

def create_virtual_environment():
    """Create and setup virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_colored("📁 Virtual environment already exists", Colors.WARNING)
        response = input(f"{Colors.OKCYAN}Do you want to recreate it? (y/n): {Colors.ENDC}").lower().strip()
        if response in ['y', 'yes']:
            print_colored("🗑️  Removing existing virtual environment...", Colors.WARNING)
            shutil.rmtree(venv_path)
        else:
            return True
    
    print_colored("🏗️  Creating virtual environment...", Colors.OKCYAN)
    
    # Create venv
    if not run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment"):
        return False
    
    print_colored("✅ Virtual environment created successfully", Colors.OKGREEN)
    return True

def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    system_info = get_system_info()
    
    if system_info['is_windows']:
        return Path("venv") / "Scripts" / "python.exe"
    else:
        return Path("venv") / "bin" / "python"

def get_venv_pip():
    """Get the path to the virtual environment pip executable"""
    system_info = get_system_info()
    
    if system_info['is_windows']:
        return Path("venv") / "Scripts" / "pip.exe"
    else:
        return Path("venv") / "bin" / "pip"

def upgrade_pip_alternative():
    """Alternative method to upgrade pip on Windows"""
    python_path = get_venv_python()
    
    print_colored("🔧 Trying alternative pip upgrade method...", Colors.OKCYAN)
    
    # Method 1: Use python -m pip
    if run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                   "Upgrading pip (method 1)", check=False):
        return True
    
    # Method 2: Use ensurepip and then upgrade
    print_colored("🔧 Trying ensurepip method...", Colors.OKCYAN)
    run_command([str(python_path), "-m", "ensurepip", "--upgrade"], 
                "Ensuring pip is installed", check=False)
    
    if run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                   "Upgrading pip (method 2)", check=False):
        return True
    
    # Method 3: Download and install pip manually
    print_colored("🔧 Trying manual pip installation...", Colors.OKCYAN)
    try:
        import urllib.request
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_path = Path("get-pip.py")
        
        urllib.request.urlretrieve(get_pip_url, get_pip_path)
        
        result = run_command([str(python_path), str(get_pip_path)], 
                           "Installing pip manually", check=False)
        
        get_pip_path.unlink(missing_ok=True)  # Clean up
        
        if result:
            return True
    except Exception as e:
        print_colored(f"Manual pip installation failed: {e}", Colors.WARNING)
    
    print_colored("⚠️  Pip upgrade failed, but continuing with existing pip version", Colors.WARNING)
    return True  # Continue anyway

def install_dependencies(use_cuda=False):
    """Install required Python dependencies"""
    pip_path = get_venv_pip()
    python_path = get_venv_python()
    
    # Try to upgrade pip with multiple methods
    if not run_command([str(pip_path), "install", "--upgrade", "pip"], "Upgrading pip", check=False):
        if not upgrade_pip_alternative():
            print_colored("⚠️  Continuing with current pip version", Colors.WARNING)
    
    # Verify pip is working
    if not run_command([str(python_path), "-m", "pip", "--version"], "Checking pip version", show_output=True):
        print_colored("❌ Pip is not working properly", Colors.FAIL)
        return False
    
    # Base dependencies
    base_deps = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "gdown>=4.7.0"
        
    ]
    
    # ONNX Runtime (crucial for AI models)
    onnx_deps = [
        "onnxruntime-gpu>=1.16.0" if use_cuda else "onnxruntime>=1.16.0",
        "onnx>=1.14.0"
    ]
    
    # Install base dependencies one by one
    for dep in base_deps:
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            if run_command([str(python_path), "-m", "pip", "install", dep], 
                          f"Installing {dep} (attempt {retry_count + 1})"):
                break
            
            retry_count += 1
            if retry_count < max_retries:
                print_colored(f"⏳ Retrying in 2 seconds...", Colors.WARNING)
                time.sleep(2)
        
        if retry_count == max_retries:
            print_colored(f"❌ Failed to install {dep} after {max_retries} attempts", Colors.FAIL)
            return False
    
    # Install ONNX Runtime (critical for AI models)
    print_colored("🧠 Installing ONNX Runtime...", Colors.OKCYAN)
    for dep in onnx_deps:
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            if run_command([str(python_path), "-m", "pip", "install", dep], 
                          f"Installing {dep} (attempt {retry_count + 1})"):
                break
            
            retry_count += 1
            if retry_count < max_retries:
                print_colored(f"⏳ Retrying in 2 seconds...", Colors.WARNING)
                time.sleep(2)
        
        if retry_count == max_retries:
            print_colored(f"❌ Failed to install {dep} after {max_retries} attempts", Colors.FAIL)
            # Try CPU version as fallback if GPU version fails
            if "onnxruntime-gpu" in dep:
                print_colored("🔄 Trying CPU version as fallback...", Colors.WARNING)
                if run_command([str(python_path), "-m", "pip", "install", "onnxruntime>=1.16.0"], 
                              "Installing ONNX Runtime (CPU fallback)"):
                    print_colored("✅ ONNX Runtime (CPU) installed successfully", Colors.OKGREEN)
                    break
            return False
    
    # PyTorch installation with compatible versions
    if use_cuda:
        print_colored("🔥 Installing PyTorch with CUDA support...", Colors.OKCYAN)
        torch_cmd = [
            str(python_path), "-m", "pip", "install", 
            "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        print_colored("💻 Installing PyTorch (CPU version)...", Colors.OKCYAN)
        torch_cmd = [
            str(python_path), "-m", "pip", "install",
            "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    if not run_command(torch_cmd, "Installing PyTorch"):
        print_colored("❌ PyTorch installation failed. Trying alternative approach...", Colors.FAIL)
        
        # Fallback: Install without version constraints
        if use_cuda:
            fallback_cmd = [
                str(python_path), "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            fallback_cmd = [
                str(python_path), "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ]
        
        if not run_command(fallback_cmd, "Installing PyTorch (fallback)"):
            print_colored("❌ PyTorch installation failed completely.", Colors.FAIL)
            return False
        else:
            print_colored("⚠️  PyTorch installed with latest versions - may have compatibility issues", Colors.WARNING)
    
    # AI model dependencies with compatible versions
    ai_deps = [
        "rembg>=2.0.50",
        "basicsr>=1.4.2",
        "facexlib>=0.3.0",
        "gfpgan>=1.3.8",
        "realesrgan>=0.3.0"
    ]
    
    for dep in ai_deps:
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            if run_command([str(python_path), "-m", "pip", "install", dep], 
                          f"Installing {dep} (attempt {retry_count + 1})"):
                break
            
            retry_count += 1
            if retry_count < max_retries:
                print_colored(f"⏳ Retrying in 2 seconds...", Colors.WARNING)
                time.sleep(2)
        
        if retry_count == max_retries:
            print_colored(f"❌ Failed to install {dep} after {max_retries} attempts", Colors.FAIL)
            print_colored(f"⚠️  You can try installing {dep} manually later", Colors.WARNING)
    
    return True

def download_file(url, destination, description="file"):
    """Download a file with progress bar"""
    print_colored(f"📥 Downloading {description}...", Colors.OKCYAN)
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r📥 Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # New line after progress
        print_colored(f"✅ Downloaded {description} successfully", Colors.OKGREEN)
        return True
        
    except Exception as e:
        print()  # New line after progress 
        print_colored(f"❌ Failed to download {description}: {e}", Colors.FAIL)
        return False

def verify_file_hash(file_path, expected_hash):
    """Verify file integrity using SHA256 hash"""
    if not Path(file_path).exists():
        return False
    
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash.lower() == expected_hash.lower()
    except Exception:
        return False

def download_models():
    """Download required AI models"""
    models_dir = Path("models")
    upscaler_dir = models_dir / "upscaler"
    remover_dir = models_dir / "remover"
    
    # Create model directories
    upscaler_dir.mkdir(parents=True, exist_ok=True)
    remover_dir.mkdir(parents=True, exist_ok=True)
    
    # Model download configurations
    upscaler_models = {
        "RealESRGAN_x4plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "hash": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
        },
        "RealESRGAN_x2plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 
            "hash": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
        },
        "RealESRGAN_x4plus_anime_6B.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "hash": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
        }
    }
    
    # Download upscaler models
    for model_name, model_info in upscaler_models.items():
        model_path = upscaler_dir / model_name
        
        if model_path.exists() and verify_file_hash(model_path, model_info["hash"]):
            print_colored(f"✅ {model_name} already exists and verified", Colors.OKGREEN)
            continue
        
        if not download_file(model_info["url"], model_path, f"upscaler model {model_name}"):
            print_colored(f"⚠️  Skipping {model_name} - you can download it manually later", Colors.WARNING)
            continue
        
        # Verify downloaded file
        if not verify_file_hash(model_path, model_info["hash"]):
            print_colored(f"❌ Hash verification failed for {model_name}", Colors.FAIL)
            model_path.unlink(missing_ok=True)
            print_colored(f"⚠️  {model_name} verification failed - you may need to download it manually", Colors.WARNING)
    
    print_colored("✅ Model download process completed", Colors.OKGREEN)
    return True

def create_run_scripts():
    """Create platform-specific run scripts"""
    system_info = get_system_info()
    
    # Windows batch file
    if system_info['is_windows']:
        with open("run.bat", "w") as f:
            f.write("""@echo off
echo Starting AI Image Processor...
call venv\\Scripts\\activate.bat
python main.py
pause
""")
        
        with open("run_cuda.bat", "w") as f:
            f.write("""@echo off
echo Starting AI Image Processor (CUDA)...
call venv\\Scripts\\activate.bat
set CUDA_VISIBLE_DEVICES=0
python main.py
pause
""")
    
    # Unix shell scripts (Linux/macOS)
    else:
        with open("run.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Starting AI Image Processor..."
source venv/bin/activate
python main.py
""")
        
        with open("run_cuda.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Starting AI Image Processor (CUDA)..."
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python main.py
""")
        
        # Make shell scripts executable
        os.chmod("run.sh", 0o755)
        os.chmod("run_cuda.sh", 0o755)
    
    print_colored("✅ Run scripts created successfully", Colors.OKGREEN)

def check_cuda_availability():
    """Check if CUDA is available on the system"""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_colored("🔥 NVIDIA GPU detected!", Colors.OKGREEN)
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    print_colored("💻 No CUDA-capable GPU detected, using CPU", Colors.WARNING)
    return False

def test_installation():
    """Test if the installation was successful"""
    print_colored("🧪 Testing installation...", Colors.OKCYAN)
    python_path = get_venv_python()
    
    # Test importing key packages with version checks
    test_imports = [
        ("import flask", "Flask"),
        ("import cv2", "OpenCV"),
        ("import PIL", "Pillow"),
        ("import numpy", "NumPy"),
        ("import torch; print(f'PyTorch: {torch.__version__}')", "PyTorch"),
        ("import torchvision; print(f'TorchVision: {torchvision.__version__}')", "TorchVision"),
        ("import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')", "ONNX Runtime")
    ]
    
    for import_test, module_name in test_imports:
        if not run_command([str(python_path), "-c", import_test], 
                          f"Testing {module_name} import", check=False, show_output=True):
            print_colored(f"⚠️  {module_name} import failed - you may need to install it manually", Colors.WARNING)
    
    print_colored("✅ Installation test completed", Colors.OKGREEN)

def main():
    """Main setup function"""
    print_colored("🚀 AI Image Processor Setup", Colors.HEADER)
    print_colored("=" * 50, Colors.HEADER)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Get system information
    system_info = get_system_info()
    print_colored(f"🖥️  System: {system_info['system'].title()} {system_info['architecture']}", Colors.OKBLUE)
    
    # Check for CUDA
    has_cuda = check_cuda_availability()
    
    # Ask user about CUDA installation
    if has_cuda:
        while True:
            choice = input(f"{Colors.OKCYAN}Do you want to install CUDA support? (y/n): {Colors.ENDC}").lower().strip()
            if choice in ['y', 'yes']:
                use_cuda = True
                break
            elif choice in ['n', 'no']:
                use_cuda = False
                break
            else:
                print_colored("Please enter 'y' or 'n'", Colors.WARNING)
    else:
        use_cuda = False
    
    # Create virtual environment
    if not create_virtual_environment():
        print_colored("❌ Failed to create virtual environment", Colors.FAIL)
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(use_cuda):
        print_colored("❌ Failed to install some dependencies", Colors.FAIL)
        print_colored("⚠️  You may need to install missing packages manually", Colors.WARNING)
        
        # Ask if user wants to continue
        while True:
            choice = input(f"{Colors.OKCYAN}Continue with setup anyway? (y/n): {Colors.ENDC}").lower().strip()
            if choice in ['y', 'yes']:
                break
            elif choice in ['n', 'no']:
                sys.exit(1)
            else:
                print_colored("Please enter 'y' or 'n'", Colors.WARNING)
    
    # Download models
    if not download_models():
        print_colored("⚠️  Some models may not have downloaded correctly", Colors.WARNING)
        print_colored("⚠️  The application may still work with reduced functionality", Colors.WARNING)
    
    # Create run scripts
    create_run_scripts()
    
    # Test installation
    test_installation()
    
    # Final success message
    print_colored("\n" + "=" * 50, Colors.OKGREEN)
    print_colored("🎉 Setup completed!", Colors.OKGREEN)
    print_colored("=" * 50, Colors.OKGREEN)
    
    # Instructions
    print_colored("\n📋 Next steps:", Colors.HEADER)
    if system_info['is_windows']:
        print_colored("   • Run 'run.bat' to start the application", Colors.OKBLUE)
        if use_cuda:
            print_colored("   • Run 'run_cuda.bat' for CUDA acceleration", Colors.OKBLUE)
    else:
        print_colored("   • Run './run.sh' to start the application", Colors.OKBLUE)
        if use_cuda:
            print_colored("   • Run './run_cuda.sh' for CUDA acceleration", Colors.OKBLUE)
    
    print_colored("   • The application will open in your default browser", Colors.OKBLUE)
    print_colored("   • Use Ctrl+C to stop the server", Colors.OKBLUE)
    
    print_colored("\n🔧 Troubleshooting:", Colors.HEADER)
    print_colored("   • If you encounter errors, try running the script as administrator", Colors.OKBLUE)
    print_colored("   • Check that your antivirus isn't blocking the installation", Colors.OKBLUE)
    print_colored("   • Ensure you have a stable internet connection", Colors.OKBLUE)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n❌ Setup interrupted by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ Unexpected error during setup: {e}", Colors.FAIL)
        print_colored("Try running the script as administrator or check your antivirus settings", Colors.WARNING)
        sys.exit(1)