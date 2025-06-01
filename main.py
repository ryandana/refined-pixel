#!/usr/bin/env python3
"""
Enhanced Background Remover & Image Upscaler
Offline web application with Apple-inspired design and progress tracking
Fixed version with CPU optimization and proper model management
"""

import os
import sys
import json
import platform
from flask import Flask, request, render_template, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import uuid
import logging
import webbrowser
import threading
import time
import socket
import zipfile
import shutil
from tqdm import tqdm
import threading
import queue
import subprocess
from pathlib import Path
import requests
from urllib.parse import urlparse

# Third-party imports (will be installed)
try:
    from rembg import remove, new_session as rembg_session
    import cv2
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please run the setup script first.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
PROCESSED_FOLDER = BASE_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REMOVER_MODELS_DIR = MODELS_DIR / 'remover'
UPSCALER_MODELS_DIR = MODELS_DIR / 'upscaler'

# Create directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODELS_DIR, REMOVER_MODELS_DIR, UPSCALER_MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Model configurations
BG_REMOVER_MODELS = {
    'u2net': {'name': 'General Purpose', 'description': 'Balanced speed and quality'},
    'isnet-general-use': {'name': 'High Accuracy', 'description': 'Best for complex images'},
    'silueta': {'name': 'Objects & Products', 'description': 'Perfect for items'}
}

UPSCALER_MODELS = {
    'RealESRGAN_x4plus': {
        'name': 'Photo Enhancement 4x',
        'description': 'Best for realistic photos',
        'scale': 4,
        'model_path': 'RealESRGAN_x4plus.pth'
    },
    'RealESRGAN_x4plus_anime_6B': {
        'name': 'Anime Enhancement 4x',
        'description': 'Perfect for illustrations',
        'scale': 4,
        'model_path': 'RealESRGAN_x4plus_anime_6B.pth'
    },
    'RealESRGAN_x2plus': {
        'name': 'Quick Enhancement 2x',
        'description': 'Faster processing',
        'scale': 2,
        'model_path': 'RealESRGAN_x2plus.pth'
    }
}

# Global model instances and progress tracking
bg_remover_sessions = {}
upscaler_models = {}
progress_status = {}
progress_queue = queue.Queue()
terminal_progress_bars = {}
system_monitor_thread = None
monitor_active = False

def get_system_info():
    """Get system information for optimal model selection"""
    system = platform.system().lower()
    has_cuda = False
    
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass
    
    return {
        'system': system,
        'has_cuda': has_cuda,
        'cpu_count': os.cpu_count()
    }

def update_progress(task_id, progress, message):
    """Update progress for a specific task"""
    progress_status[task_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }
    
    # Also update terminal progress bar
    if task_id not in terminal_progress_bars:
        terminal_progress_bars[task_id] = tqdm(total=100, desc=f"[{task_id}]", position=len(terminal_progress_bars), leave=False)
    
    bar = terminal_progress_bars[task_id]
    bar.n = progress
    bar.set_description(f"[{task_id}] {message}")
    bar.refresh()

    if progress >= 100:
        bar.close()
        if task_id in terminal_progress_bars:
            del terminal_progress_bars[task_id]

def download_bg_remover_model(model_name, task_id=None):
    """Download and cache background remover models locally"""
    model_dir = REMOVER_MODELS_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    
    # Check if model already exists (basic check)
    model_files = list(model_dir.glob('*'))
    if model_files:
        logger.info(f"Background remover model {model_name} already exists locally")
        return True
    
    try:
        if task_id:
            update_progress(task_id, 10, f"Downloading {model_name} model...")
        
        # Set environment variable to use local models directory
        os.environ['U2NET_HOME'] = str(REMOVER_MODELS_DIR)
        
        # Initialize session (this will download the model to our local directory)
        logger.info(f"Downloading background remover model: {model_name}")
        session = rembg_session(model_name)
        
        if task_id:
            update_progress(task_id, 30, f"Model {model_name} downloaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download bg remover model {model_name}: {e}")
        if task_id:
            update_progress(task_id, 0, f"Error downloading model: {str(e)}")
        return False

def initialize_bg_remover_session(model_name, task_id=None):
    """Initialize background remover session with progress tracking"""
    if model_name not in bg_remover_sessions:
        try:
            # Ensure model is downloaded first
            if not download_bg_remover_model(model_name, task_id):
                raise Exception(f"Failed to download model {model_name}")
            
            if task_id:
                update_progress(task_id, 35, f"Loading {BG_REMOVER_MODELS[model_name]['name']} model...")
            
            # Set environment variable to use local models directory
            os.environ['U2NET_HOME'] = str(REMOVER_MODELS_DIR)
            
            logger.info(f"Initializing background remover model: {model_name}")
            bg_remover_sessions[model_name] = rembg_session(model_name)
            
            if task_id:
                update_progress(task_id, 45, "Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize bg remover model {model_name}: {e}")
            if task_id:
                update_progress(task_id, 0, f"Error loading model: {str(e)}")
            raise
    return bg_remover_sessions[model_name]

def initialize_upscaler_model(model_name, task_id=None):
    """Initialize upscaler model with progress tracking and proper model configuration"""
    if model_name not in upscaler_models:
        try:
            model_config = UPSCALER_MODELS[model_name]
            model_path = UPSCALER_MODELS_DIR / model_config['model_path']

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if task_id:
                update_progress(task_id, 20, f"Loading {model_config['name']} model...")

            logger.info(f"Initializing upscaler model: {model_name}")

            # FIXED: Proper model architecture configuration
            netscale = model_config['scale']
            
            if 'anime' in model_name.lower():
                # Anime model uses 6 blocks
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=6,  # Anime models use 6 blocks
                    num_grow_ch=32, 
                    scale=netscale
                )
            else:
                # Photo models use 23 blocks
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23,  # Photo models use 23 blocks
                    num_grow_ch=32, 
                    scale=netscale
                )

            system_info = get_system_info()
            device = 'cuda' if system_info['has_cuda'] else 'cpu'

            if task_id:
                update_progress(task_id, 40, f"Initializing on {device.upper()}...")

            # FIXED: Better tile size configuration for quality
            if device == 'cpu':
                # Larger tile size for better quality, even on CPU
                tile_size = 512  # Increased from 256
                tile_pad = 20    # Increased padding
                pre_pad = 10     # Added pre-padding
                cv2.setNumThreads(os.cpu_count()) # Utilize all CPU cores for OpenCV operations
                use_half = False # Use full precision on CPU for better quality
            else: # GPU settings
                tile_size = 1024  # Larger tiles for GPU
                tile_pad = 32
                pre_pad = 10
                use_half = True # Can use half precision on GPU for speed if supported

            logger.info(f"Using device: {device}, tile_size: {tile_size}, half_precision: {use_half}")

            # FIXED: Proper RealESRGANer configuration
            upscaler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                model=model,
                tile=tile_size,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=use_half,
                device=device
            )

            upscaler_models[model_name] = upscaler

            if task_id:
                update_progress(task_id, 60, "Model ready for processing")

        except Exception as e:
            logger.error(f"Failed to initialize upscaler model {model_name}: {e}")
            if task_id:
                update_progress(task_id, 0, f"Error loading model: {str(e)}")
            raise

    return upscaler_models[model_name]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    return jsonify({
        'bgRemover': BG_REMOVER_MODELS,
        'upscaler': UPSCALER_MODELS
    })

@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    if task_id in progress_status:
        return jsonify(progress_status[task_id])
    return jsonify({'progress': 0, 'message': 'Task not found'})

@app.route('/api/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    operation = request.form.get('operation', 'bg-remove')
    task_id = request.form.get('task_id', str(uuid.uuid4()))
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Generate unique filename
        filename = str(uuid.uuid4())
        original_ext = os.path.splitext(file.filename)[1].lower()
        
        if original_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return jsonify({'error': 'Invalid file type'}), 400

        input_filename = filename + original_ext
        input_path = UPLOAD_FOLDER / input_filename
        
        # Save uploaded file
        update_progress(task_id, 10, 'Uploading image...')
        file.save(str(input_path))
        logger.info(f"Saved uploaded file: {input_path}")

        # Process based on operation
        if operation == 'bg-remove':
            bg_model = request.form.get('bg_model', 'isnet-general-use')
            output_filename = f"{filename}_no_bg.png"
            output_path = PROCESSED_FOLDER / output_filename
            
            # Remove background with progress tracking
            update_progress(task_id, 15, 'Preparing background removal...')
            session = initialize_bg_remover_session(bg_model, task_id)
            
            update_progress(task_id, 60, 'Removing background...')
            input_image = Image.open(input_path)
            output_image = remove(input_image, session=session)
            
            update_progress(task_id, 90, 'Saving processed image...')
            output_image.save(str(output_path))
            
        elif operation == 'upscale':
            upscaler_model = request.form.get('upscaler_model', 'RealESRGAN_x4plus')
            scale = UPSCALER_MODELS[upscaler_model]['scale']
            output_filename = f"{filename}_upscaled_{scale}x.png"
            output_path = PROCESSED_FOLDER / output_filename
            
            # Upscale image with progress tracking
            update_progress(task_id, 15, 'Preparing image enhancement...')
            upscaler = initialize_upscaler_model(upscaler_model, task_id)
            
            update_progress(task_id, 70, f'Enhancing image {scale}x...')
            input_image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            
            output, _ = upscaler.enhance(input_image, outscale=scale)
            
            update_progress(task_id, 90, 'Saving enhanced image...')
            cv2.imwrite(str(output_path), output)
            
        else:
            return jsonify({'error': 'Invalid operation'}), 400

        # Clean up input file
        try:
            input_path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove input file: {e}")

        update_progress(task_id, 100, 'Processing complete!')
        
        return jsonify({
            'processed_image_url': f'/api/processed/{output_filename}',
            'download_filename': output_filename
        })

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        update_progress(task_id, 0, f'Error: {str(e)}')
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/processed/<filename>')
def send_processed_file(filename):
    file_path = PROCESSED_FOLDER / filename
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path), mimetype='image/png')

def find_free_port(start_port=5000, max_attempts=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")

def wait_for_server(host='localhost', port=5000, timeout=30):
    """Wait for server to start"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def cleanup_old_files():
    """Clean up old processed files to save space"""
    try:
        current_time = time.time()
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for file_path in folder.glob('*'):
                if file_path.is_file():
                    # Remove files older than 1 hour
                    if current_time - file_path.stat().st_mtime > 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def run_server():
    """Run Flask server silently and start monitoring threads"""
    port = find_free_port()

    # Disable Werkzeug access log
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    logger.info(f"Starting server on port {port}")

    # Start periodic cleanup
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    # Start Flask server
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)


def periodic_cleanup():
    """Periodically clean up old files"""
    while True:
        time.sleep(1800)  # Clean every 30 minutes
        cleanup_old_files()
        
        

# Terminal progress and system monitoring
progress_bars = {}

def update_progress(task_id, progress, message):
    """Update task progress and show terminal bar"""
    progress_status[task_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }

    if task_id not in progress_bars:
        progress_bars[task_id] = tqdm(total=100, desc=f"[{task_id}]", position=len(progress_bars), leave=False)
    
    bar = progress_bars[task_id]
    bar.n = progress
    bar.set_description(f"[{task_id}] {message}")
    bar.refresh()

    if progress >= 100:
        bar.close()
        del progress_bars[task_id]

if __name__ == '__main__':
    try:
        # Check if models exist
        missing_models = []
        for model_name, config in UPSCALER_MODELS.items():
            model_path = UPSCALER_MODELS_DIR / config['model_path']
            if not model_path.exists():
                missing_models.append(config['model_path'])
        
        if missing_models:
            print("\n⚠️  Warning: Missing upscaler model files:")
            for model in missing_models:
                print(f"   - {model}")
            print("\nBackground removal will work, but upscaling may fail.")
            print("Please download the required model files to the models/upscaler/ directory.")
            print("\nContinuing in 5 seconds...")
            time.sleep(5)
        
        # Find available port
        port = find_free_port()
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start and open browser
        if wait_for_server(port=port):
            url = f"http://127.0.0.1:{port}"
            logger.info(f"✅ Server started successfully!")
            logger.info(f"🌐 Opening browser to {url}")
            webbrowser.open(url)
        else:
            logger.warning("Server did not start in time")
        
        print(f"\n🚀 Refined Pixel is running at: http://127.0.0.1:{port}")
        print("📱 The application is now completely offline!")
        print("❌ Press Ctrl+C to stop the server\n")
        
        # Keep main thread alive
        try:
            server_thread.join()
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down gracefully...")
            cleanup_old_files()  # Final cleanup
            print("👋 Thank you for using Refined Pixel!")
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the error message above and try again.")
        sys.exit(1)