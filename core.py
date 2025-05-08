"""
core.py
Core module for DeOldify Video Colorizer.
Contains all the video processing logic, configuration, and model handling.
"""

import os
import sys
import json
import subprocess
import threading
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Callable

import logging
import requests
from tqdm import tqdm

import torch
from functools import partial
# Allow DeOldify weight loading to use `partial` in torch.load
try:
    torch.serialization.add_safe_globals([partial])
except AttributeError:
    # Older PyTorch may not support adding safe globals
    pass

# Logger for core module
logger = logging.getLogger("deoldify_core")
logger.setLevel(logging.INFO)
# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

# Progress reporting support for CLI
progress_callback_func = None

def report_progress(frame_num: int, total_frames: int, message: str = "Processing"):
    """
    Report progress to any registered callback functions.
    Call this from within the video processing loop.
    """
    global progress_callback_func
    if progress_callback_func:
        try:
            progress_callback_func(frame_num, total_frames, message)
        except Exception as e:
            logger.error(f"Error in progress callback: {e}", file=sys.stderr)

# Override torch.load to always use weights_only=False when supported
try:
    import inspect as _inspect
    sig = _inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        _orig_torch_load = torch.load
        def _torch_load_override(f, *args, **kwargs):
            # Only add weights_only if not already specified
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(f, *args, **kwargs)
        torch.load = _torch_load_override
except Exception as e:
    # If signature inspection or override fails, proceed without patch
    logger.warning(f"Failed to patch torch.load: {e}")
    pass

from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect

# OpenCV support
try:
    import cv2
    # Suppress OpenCV warnings for missing watermark image
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.ERROR)
    except AttributeError:
        pass
except ImportError:
    logger.error("OpenCV (cv2) module not found. Please install 'opencv-python' via pip or 'opencv' via conda.")
    sys.exit(1)

# Allow overriding APP_DIR and WORK_DIR via environment variables
# Base application directory (can be overridden via env var)
APP_DIR = Path(os.environ.get("DEOLDIFY_APP_DIR", str(Path.home() / ".deoldify_colorizer")))
REPO_DIR = APP_DIR / "DeOldify"
MODELS_DIR = REPO_DIR / "models"
# Working directory for temporary files (override with DEOLDIFY_WORK_DIR)
WORK_DIR = Path(os.environ.get("DEOLDIFY_WORK_DIR", str(APP_DIR / "work")))
CONFIG_FILE = APP_DIR / "config.json"

# Timeouts for subprocess calls
SUBPROCESS_TIMEOUT = 120  # seconds

# Check for required external tools
def ensure_tools() -> bool:
    """
    Ensure required external tools are on PATH.
    Returns True if all required tools are found, False otherwise.
    """
    missing_tools = []
    for tool in ("git", "ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"The following required tools are missing: {', '.join(missing_tools)}. Please install them.")
        return False
    return True

# Pretrained model URLs
MODEL_URLS = {
    "ColorizeArtistic_gen.pth": "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
    "ColorizeStable_gen.pth":   "https://data.deepai.org/deoldify/ColorizeStable_gen.pth"
}

# Global flags
process_running = False
should_cancel = False

# Default configuration
DEFAULT_CONFIG = {
    "render_factor": 35,
    "artistic_mode": True,
    "output_dir": str(Path.home() / "Videos" / "Colorized"),
    "recent_files": [],
    "theme": "dark",
    "auto_clean": True,
    "preview_quality": "medium",
    "use_gpu": True,
    "max_workers": None  # None means use CPU count
}

def ensure_dirs():
    """Ensure necessary directories exist."""
    for dir_path in [APP_DIR, WORK_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def load_config() -> Dict[str, Any]:
    """Load configuration from file or create default."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            updated = False
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
                    updated = True
            if updated:
                save_config(config)
            return config
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading config: {e}")
    # Create default if missing or corrupted
    return create_default_config()

def create_default_config() -> Dict[str, Any]:
    """Create default configuration if it doesn't exist."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        logger.info("Creating default configuration...")
        save_config(DEFAULT_CONFIG)
        logger.info("Default configuration created")
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Error saving config: {e}")
        return False

def update_recent_files(filepath: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add a file to recent files list."""
    if config is None:
        config = load_config()
    path_str = str(filepath)
    recent = config.get("recent_files", [])
    if path_str in recent:
        recent.remove(path_str)
    recent.insert(0, path_str)
    config["recent_files"] = recent[:10]
    save_config(config)
    return config

def run_subprocess(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run a subprocess with timeout and proper error handling.
    Raises subprocess.TimeoutExpired on timeout.
    """
    timeout = kwargs.pop('timeout', SUBPROCESS_TIMEOUT)
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise

def ensure_conda() -> bool:
    """Ensure 'deoldify' conda env exists; do not auto-install."""
    try:
        # Check that conda is available
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Conda not found! Please install Conda and rerun setup.")
        return False

    # Check if 'deoldify' environment exists
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, check=True
        )
        envs = json.loads(result.stdout).get("envs", [])
        if any(Path(env).stem == "deoldify" for env in envs):
            return True
        else:
            logger.error("Conda environment 'deoldify' not found. Please run initial_setup.sh to install dependencies.")
            return False
    except Exception as e:
        logger.error(f"Error checking conda environments: {e}")
        return False

def clone_repository() -> bool:
    """Clone the DeOldify repository if missing."""
    if REPO_DIR.exists():
        return True
    try:
        logger.info("Cloning DeOldify repository...")
        run_subprocess(
            ["git", "clone", "https://github.com/jantic/DeOldify.git", str(REPO_DIR)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to clone repository: {e}")
        return False

def download_model_weights() -> bool:
    """Download pretrained model weights if not present, with robust error handling."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not ensure_conda():
        return False
    
    success = True
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        if dest.exists():
            continue
            
        logger.info(f"Downloading {name}...")
        try:
            with requests.get(url, stream=True, timeout=30) as resp:
                if resp.status_code != 200:
                    logger.error(f"Failed to download {name}: HTTP {resp.status_code}. Skipping this file.")
                    success = False
                    continue
                    
                total = int(resp.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                
                # Validate file content is not HTML (sometimes a 200 response is an error page)
                with open(dest, "rb") as vf:
                    header = vf.read(15)
                if header.lstrip().startswith(b"<"):
                    logger.error(f"Downloaded {name} appears to be HTML. Removing file.")
                    dest.unlink(missing_ok=True)
                    success = False
                    continue
        except Exception as e:
            logger.error(f"Error downloading {name}: {e}. Removing file if exists.")
            try:
                dest.unlink(missing_ok=True)
            except Exception as unlink_exc:
                logger.error(f"Failed to remove incomplete file {dest}: {unlink_exc}")
            success = False
            continue
    
    # Check that all required models exist
    for name in MODEL_URLS:
        if not (MODELS_DIR / name).exists():
            logger.error(f"Model file {name} missing after download attempts.")
            success = False
    
    return success

def setup_deoldify() -> bool:
    """Run full setup: directories, conda, clone, download weights."""
    ensure_dirs()
    # Check for required external tools
    if not ensure_tools():
        return False
    if not ensure_conda():
        return False
    if not clone_repository():
        return False
    if not download_model_weights():
        return False
    logger.info("DeOldify setup complete")
    return True

def check_setup() -> bool:
    """Return True if the repository and models directory exist with required files."""
    if not REPO_DIR.exists() or not MODELS_DIR.exists():
        return False
        
    # Check for essential model files
    for model_file in MODEL_URLS.keys():
        if not (MODELS_DIR / model_file).exists():
            return False
    
    return True

def get_best_device() -> torch.device:
    """
    Get the best available PyTorch device (CUDA GPU, MPS, or CPU).
    Returns torch.device for the selected device.
    """
    config = load_config()
    if not config.get("use_gpu", True):
        logger.info("GPU disabled in config, using CPU")
        return torch.device("cpu")
        
    # Try CUDA first
    if torch.cuda.is_available():
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    # Try MPS (Apple Silicon) next
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS")
        return torch.device("mps")
    
    # Fall back to CPU
    logger.info("No GPU available, using CPU")
    return torch.device("cpu")

def get_colorizer(artistic: bool = True, for_video: bool = False):
    """
    Initialize DeOldify colorizer; for_video selects video pipeline.
    Restores working directory after import.
    Also selects the appropriate PyTorch device.
    """
    device = get_best_device()
    logger.info(f"Using device: {device}")

    if str(REPO_DIR) not in sys.path:
        sys.path.append(str(REPO_DIR))
    previous_cwd = os.getcwd()
    try:
        os.chdir(REPO_DIR)
        from deoldify.visualize import get_image_colorizer, get_video_colorizer
        # Determine which creator to call and which arguments it accepts
        creator = get_video_colorizer if for_video else get_image_colorizer
        sig = inspect.signature(creator)
        kwargs = {}
        if "artistic" in sig.parameters:
            kwargs["artistic"] = artistic
        if "device" in sig.parameters:
            kwargs["device"] = device
        
        colorizer = creator(**kwargs)
        
        # Move model to selected device
        try:
            colorizer.to(device)
        except Exception as e:
            logger.warning(f"Failed to move model to device {device}: {e}")
            
        return colorizer
    except ImportError as e:
        logger.error(f"Failed to import DeOldify: {e}")
        logger.error("Please ensure DeOldify is properly installed")
        raise
    finally:
        os.chdir(previous_cwd)

def get_video_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Get video frame count, FPS, duration, width, and height."""
    try:
        result = run_subprocess([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json", str(file_path)
        ], capture_output=True, text=True, check=True)
        
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        
        # Handle r_frame_rate
        try:
            num, den = stream["r_frame_rate"].split("/")
            fps = float(num) / float(den) if den != "0" else float(num)
        except (ValueError, ZeroDivisionError):
            logger.warning(f"Invalid frame rate: {stream.get('r_frame_rate')}, using 30fps")
            fps = 30.0
        
        # Get duration from stream or format section
        duration = float(stream.get("duration") or info.get("format", {}).get("duration", 0))
        if duration <= 0:
            logger.warning("Invalid duration, estimating from frame count")
            # Try to get frame count directly
            frame_count_result = run_subprocess([
                "ffprobe", "-v", "error",
                "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "json", str(file_path)
            ], capture_output=True, text=True, check=False)
            
            try:
                frame_count = int(json.loads(frame_count_result.stdout)["streams"][0].get("nb_read_frames", 0))
                duration = frame_count / fps if frame_count > 0 and fps > 0 else 0
            except (ValueError, KeyError, json.JSONDecodeError):
                logger.warning("Could not determine frame count, using 0")
                frame_count = 0
        else:
            frame_count = int(duration * fps)
        
        width = int(stream["width"])
        height = int(stream["height"])
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count
        }
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error getting video info: {e}")
        return None

def has_audio_stream(file_path: Union[str, Path]) -> bool:
    """Check if a video file has an audio stream."""
    try:
        result = run_subprocess([
            "ffprobe", "-i", str(file_path),
            "-show_streams", "-select_streams", "a",
            "-loglevel", "error"
        ], capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())
    except subprocess.SubprocessError:
        logger.warning(f"Failed to check for audio in {file_path}")
        return False

def extract_frames(input_file: Union[str, Path], output_dir: Path) -> bool:
    """Extract frames from video to the output directory."""
    input_file_str = str(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%08d.png"
    
    try:
        proc = subprocess.Popen([
            "ffmpeg", "-i", input_file_str,
            "-qscale:v", "1", str(pattern)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        while proc.poll() is None:
            if should_cancel:
                proc.terminate()
                logger.info("Frame extraction cancelled")
                return False
            time.sleep(0.1)
        
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            logger.error(f"ffmpeg error: {stderr}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return False

def extract_audio(input_file: Union[str, Path], output_file: Path) -> bool:
    """Extract audio from video if present."""
    if not has_audio_stream(input_file):
        logger.info("No audio stream found in video")
        return False
        
    try:
        run_subprocess([
            "ffmpeg", "-y", "-i", str(input_file),
            "-q:a", "0", "-map", "a", str(output_file)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if not output_file.exists() or output_file.stat().st_size == 0:
            logger.warning("Audio extraction produced empty file")
            return False
            
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Error extracting audio: {e}")
        return False

def colorize_frame(colorizer, input_path: Path, output_path: Path, render_factor: int) -> bool:
    """Colorize a single frame."""
    try:
        # Choose the appropriate transform function
        if hasattr(colorizer, 'vis'):
            func = colorizer.vis.get_transformed_image
        elif hasattr(colorizer, 'get_transformed_image'):
            func = colorizer.get_transformed_image
        else:
            logger.error(f"No method to transform image on {type(colorizer)}")
            return False
            
        # Disable watermark if supported
        sig = inspect.signature(func)
        if 'watermarked' in sig.parameters:
            img = func(str(input_path), render_factor, watermarked=False)
        else:
            img = func(str(input_path), render_factor)
            
        img.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error colorizing frame {input_path.name}: {e}")
        return False

def colorize_frames(
    colorizer,
    frames_dir: Path,
    output_dir: Path,
    render_factor: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> bool:
    """Colorize all frames in parallel with progress reporting."""
    global progress_callback_func
    progress_callback_func = progress_callback
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.png"))
    
    if not frames:
        logger.error(f"No frames found in {frames_dir}")
        return False
        
    config = load_config()
    max_workers = config.get("max_workers") or os.cpu_count() or 4
    
    logger.info(f"Colorizing {len(frames)} frames using {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(colorize_frame, colorizer, frame, output_dir / frame.name, render_factor):
            i for i, frame in enumerate(frames)
        }
        
        with tqdm(total=len(futures), desc="Colorizing frames", file=sys.stdout) as pbar:
            for i, future in enumerate(as_completed(futures)):
                if should_cancel:
                    executor.shutdown(cancel_futures=True)
                    logger.info("Colorization cancelled")
                    return False
                    
                success = future.result()
                if not success:
                    logger.error(f"Failed to colorize frame {i+1}/{len(frames)}")
                    return False
                    
                pbar.update(1)
                
                # Report progress
                if progress_callback:
                    try:
                        progress_callback(i+1, len(frames), "Colorizing frames")
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")
    
    return True

def assemble_video(
    frames_dir: Path,
    audio_file: Optional[Path],
    output_file: Path,
    fps: float,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> bool:
    """Assemble colorized frames into final video."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    temp_video = WORK_DIR / "temp_output.mp4"
    
    if progress_callback:
        progress_callback(0, 2, "Assembling video")
    
    try:
        # First create video from frames
        proc = subprocess.Popen([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "18",
            str(temp_video)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        while proc.poll() is None:
            if should_cancel:
                proc.terminate()
                logger.info("Video assembly cancelled")
                return False
            time.sleep(0.1)
        
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            logger.error(f"ffmpeg error while creating video: {stderr}")
            return False
        
        if progress_callback:
            progress_callback(1, 2, "Adding audio")
        
        # Add audio if available
        if audio_file and audio_file.exists() and audio_file.stat().st_size > 0:
            proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-i", str(temp_video),
                "-i", str(audio_file),
                "-c:v", "copy", "-c:a", "aac", "-q:a", "2",
                str(output_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while proc.poll() is None:
                if should_cancel:
                    proc.terminate()
                    logger.info("Audio addition cancelled")
                    return False
                time.sleep(0.1)
                
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
                logger.error(f"ffmpeg error while adding audio: {stderr}")
                return False
        else:
            logger.info("No audio file found or empty audio, using video without audio")
            shutil.copy(temp_video, str(output_file))
        
        if progress_callback:
            progress_callback(2, 2, "Video assembly complete")
            
        return True
    except Exception as e:
        logger.error(f"Error assembling video: {e}")
        return False
    finally:
        if temp_video.exists():
            try:
                temp_video.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temporary video: {e}")

def format_time(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "00:00"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

def clean_workspace(keep_models: bool = True):
    """Clean temporary files to save space."""
    if keep_models:
        # Only clean work directory
        if WORK_DIR.exists():
            for item in WORK_DIR.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to clean {item}: {e}")
    else:
        # Clean all app directories
        if APP_DIR.exists():
            try:
                shutil.rmtree(APP_DIR, ignore_errors=True)
                logger.info(f"Removed {APP_DIR}")
            except Exception as e:
                logger.error(f"Failed to clean app directory: {e}")
        ensure_dirs()

def process_video(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    render_factor: int,
    artistic_mode: bool,
    auto_clean: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    completion_callback: Optional[Callable[[bool, str, Optional[str]], None]] = None
):
    """Colorize a full video with progress and completion callbacks."""
    global process_running, should_cancel
    process_running = True
    should_cancel = False
    
    if not completion_callback:
        # Default no-op callback
        completion_callback = lambda success, message, output_file=None: None
    
    try:
        # Ensure paths are Path objects
        input_file = Path(input_file) if not isinstance(input_file, Path) else input_file
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video metadata
        info = get_video_info(input_file)
        if not info:
            completion_callback(False, "Failed to get video info")
            return
            
        fps = info["fps"]
        frame_count = info["frame_count"]
        
        if progress_callback:
            progress_callback(0, 4, "Starting video processing")
        
        # Set up temporary directories
        temp_frames = WORK_DIR / "frames"
        temp_colored = WORK_DIR / "colored"
        audio_file = WORK_DIR / "audio.aac"
        
        # Step 1: Extract frames
        if progress_callback:
            progress_callback(0, 4, "Extracting frames")
            
        has_audio = has_audio_stream(input_file)
        if has_audio:
            # Extract audio in parallel
            audio_thread = threading.Thread(
                target=extract_audio,
                args=(input_file, audio_file)
            )
            audio_thread.start()
        else:
            audio_file = None
        
        if not extract_frames(input_file, temp_frames):
            completion_callback(False, "Frame extraction failed")
            return
            
        if has_audio:
            audio_thread.join()
        
        # Step 2: Initialize colorizer
        if progress_callback:
            progress_callback(1, 4, "Initializing colorizer")
            
        try:
            colorizer = get_colorizer(artistic=artistic_mode, for_video=True)
        except Exception as e:
            completion_callback(False, f"Failed to initialize colorizer: {e}")
            return
        
        # Step 3: Colorize frames
        if progress_callback:
            progress_callback(2, 4, "Colorizing frames")
            
        if not colorize_frames(
            colorizer,
            temp_frames,
            temp_colored,
            render_factor,
            progress_callback=lambda current, total, _: progress_callback(
                2 + (current / total),
                4,
                f"Colorizing frame {current}/{total}"
            ) if progress_callback else None
        ):
            completion_callback(False, "Frame colorization failed")
            return
        
        # Step 4: Assemble final video
        if progress_callback:
            progress_callback(3, 4, "Assembling video")
            
        output_file = output_dir / f"{input_file.stem}_colorized.mp4"
        if not assemble_video(
            temp_colored,
            audio_file,
            output_file,
            fps,
            progress_callback=lambda current, total, msg: progress_callback(
                3 + (current / total),
                4,
                msg
            ) if progress_callback else None
        ):
            completion_callback(False, "Video assembly failed")
            return
        
        # Clean up if requested
        if auto_clean:
            if progress_callback:
                progress_callback(4, 4, "Cleaning up temporary files")
            clean_workspace(True)
        
        # Update recent files list
        update_recent_files(str(output_file))
        
        # Report success
        completion_callback(True, "Video colorized successfully", str(output_file))
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        completion_callback(False, f"Error processing video: {e}")
    finally:
        process_running = False
        
class VideoProcessingMonitor:
    """Helper class to monitor and control video processing."""
    
    def __init__(self):
        self.processing = False
        self.progress = 0.0
        self.status_message = ""
        self.last_update_time = 0
    
    def start(self):
        """Mark processing as started."""
        self.processing = True
        self.progress = 0.0
        self.status_message = "Starting..."
    
    def update(self, progress: float, message: str):
        """Update progress and status message."""
        self.progress = progress
        self.status_message = message
        self.last_update_time = time.time()
    
    def stop(self, success: bool = True):
        """Mark processing as stopped."""
        self.processing = False
        if success:
            self.progress = 1.0
            self.status_message = "Complete"
        else:
            self.status_message = "Failed"
    
    def cancel(self):
        """Request cancellation of processing."""
        global should_cancel
        if self.processing:
            should_cancel = True
            self.status_message = "Cancelling..."
            return True
        return False

def get_supported_video_formats() -> List[str]:
    """Return a list of supported video file extensions."""
    return [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"]

def is_video_file(file_path: Union[str, Path]) -> bool:
    """Check if a file is a supported video based on extension."""
    ext = str(Path(file_path).suffix).lower()
    return ext in get_supported_video_formats()

def validate_video_file(file_path: Union[str, Path]) -> bool:
    """
    Validate that the file exists and can be read by ffprobe.
    Returns True if the file is a valid video, False otherwise.
    """
    try:
        if not Path(file_path).is_file():
            logger.error(f"File not found: {file_path}")
            return False
            
        # Try to get video info as validation
        info = get_video_info(file_path)
        if not info:
            logger.error(f"Not a valid video file: {file_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        return False

def get_version() -> str:
    """Return the DeOldify core version."""
    return "1.1.0"  # Update this when making significant changes
