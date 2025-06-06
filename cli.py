#!/usr/bin/env python3
"""
ColorRevive
Command Line Interface for DeOldify Video Colorizer.
A beautiful and functional CLI that works with the core.py module for video processing.
"""

import os
import sys
import time
import threading
import argparse
import shutil
from pathlib import Path

import re
import traceback

import logging
import subprocess
import warnings
# Suppress watermark-related warnings
warnings.filterwarnings("ignore", ".*watermark.*")
# Suppress additional irrelevant warnings
warnings.filterwarnings("ignore", ".*validation set.*")
warnings.simplefilter("ignore")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Regex to strip ANSI codes
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

# Validate required external dependencies
try:
    import yaml
except ImportError:
    logger.error("PyYAML is required. Install with 'pip install pyyaml'")
    sys.exit(1)
try:
    import matplotlib
except ImportError:
    logger.error("matplotlib is required for DeOldify visualization. Install with 'pip install matplotlib'")
    sys.exit(1)

# Import core module for actual DeOldify functionality
try:
    import core
except ModuleNotFoundError:
    logger.error("Missing core module. Ensure 'core.py' is present and accessible.")
    sys.exit(1)

# Terminal dimensions
try:
    TERM_WIDTH = shutil.get_terminal_size().columns
    TERM_HEIGHT = shutil.get_terminal_size().lines
except Exception:
    TERM_WIDTH, TERM_HEIGHT = 80, 24
CLEAR_LINE = "\033[2K\r"  # ANSI code to clear the current line
CURSOR_UP = "\033[1A"     # ANSI code to move cursor up one line

# Colors and styling - Modern color palette
class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Primary colors - using a more muted/modern palette
    PRIMARY = "\033[38;5;61m"    # Moderate purple instead of cyan
    SECONDARY = "\033[38;5;134m"  # Medium purple
    ACCENT = "\033[38;5;168m"     # Soft pink/magenta
    
    # Status colors
    SUCCESS = "\033[38;5;108m"    # Soft green
    ERROR = "\033[38;5;174m"      # Soft red
    WARNING = "\033[38;5;179m"    # Soft yellow/orange
    INFO = "\033[38;5;74m"        # Soft blue
    
    # UI elements
    HEADING = "\033[38;5;146m"    # Light lavender for headings
    TEXT = "\033[38;5;252m"       # Light grey for text
    SUBTLE = "\033[38;5;244m"     # Medium grey for subtle text
    
    # Background colors
    BG_PRIMARY = "\033[48;5;61m"
    BG_SECONDARY = "\033[48;5;134m"
    BG_DARK = "\033[48;5;236m"
    BG_LIGHT = "\033[48;5;252m"

# ASCII art logo with new color scheme
LOGO = f"""
{Colors.PRIMARY}{Colors.BOLD}
    {Colors.ACCENT}ColorRevive{Colors.RESET}
"""

# Progress bar function with improved styling
def progress_bar(progress, total, prefix="", suffix="", length=30, fill='█', empty='░'):
    """Create a text-based progress bar with modern styling"""
    # Ensure we have valid progress and total values
    progress = max(0, min(progress, total))
    total = max(1, total)  # Prevent division by zero

    percent = min(100, int(100 * (progress / float(total))))
    filled_length = int(length * progress // total)
    bar = fill * filled_length + empty * (length - filled_length)

    # Choose color based on progress
    if percent < 30:
        color = Colors.ERROR
    elif percent < 70:
        color = Colors.WARNING
    else:
        color = Colors.SUCCESS

    return f"{prefix} {Colors.SUBTLE}[{color}{bar}{Colors.SUBTLE}] {percent}%{Colors.RESET} {suffix}"

# Status message styles
def status_info(message):
    """Format an info message"""
    return f"{Colors.INFO}ℹ {Colors.RESET}{message}"

def status_success(message):
    """Format a success message"""
    return f"{Colors.SUCCESS}✓ {Colors.RESET}{message}"

def status_error(message):
    """Format an error message"""
    return f"{Colors.ERROR}✗ {Colors.RESET}{message}"

def status_warning(message):
    """Format a warning message"""
    return f"{Colors.WARNING}⚠ {Colors.RESET}{message}"

def status_processing(message):
    """Format a processing message"""
    return f"{Colors.PRIMARY}➤ {Colors.RESET}{message}"

# Helper functions for UI
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text, color=None):
    """Print text centered in the terminal"""
    if color:
        text = f"{color}{text}{Colors.RESET}"
    # Strip ANSI codes for accurate width calculation
    clean_text = ANSI_ESCAPE.sub('', text)
    padding = max(0, (TERM_WIDTH - len(clean_text)) // 2)
    print(" " * padding + text)

def print_header():
    """Print application header"""
    clear_screen()
    print_centered(LOGO)
    print_centered("Transform black & white videos into colorized masterpieces", Colors.SUBTLE)
    print("\n" + "═" * TERM_WIDTH + "\n")  # Using a more distinct separator

def print_separator():
    """Print a separator line"""
    print(f"\n{Colors.SUBTLE}{'─' * TERM_WIDTH}{Colors.RESET}\n")

def print_footer():
    """Print application footer"""
    print_separator()
    print(f"{Colors.DIM}ColorRevive CLI v1.1 - Press Ctrl+C to quit at any time{Colors.RESET}")

def format_time(seconds):
    """Format seconds into a readable time string"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{s:.1f}s"

def print_video_info(info):
    """Print video information in a nice format"""
    if not info:
        print(status_error("No video information available"))
        return
        
    duration = core.format_time(info.get("duration", 0))
    frames = info.get("frame_count", 0)
    fps = info.get("fps", 0)
    width = info.get("width", 0)
    height = info.get("height", 0)
    
    print(f"{Colors.HEADING}Video Information:{Colors.RESET}")
    print(f"  {Colors.SUBTLE}Duration:{Colors.RESET}   {duration}")
    print(f"  {Colors.SUBTLE}Frames:{Colors.RESET}     {frames}")
    print(f"  {Colors.SUBTLE}FPS:{Colors.RESET}        {fps:.2f}")
    print(f"  {Colors.SUBTLE}Resolution:{Colors.RESET} {width}x{height}")

# Setup environment
def setup_environment():
    """Setup the DeOldify environment"""
    print_header()
    print(status_info("Setting up DeOldify environment...\n"))
    try:
        # Create directories
        print(status_processing("Creating directories..."))
        try:
            core.ensure_dirs()
        except Exception as e:
            print(status_error(f"Failed to create directories: {e}"))
            return False
        time.sleep(0.5)

        # Clone repository
        print(status_processing("Cloning DeOldify repository..."))
        try:
            core.clone_repository()
        except Exception as e:
            print(status_error(f"Failed to clone repository: {e}"))
            return False
        time.sleep(0.5)

        # Download models
        print(status_processing("Downloading model weights..."))
        try:
            core.download_model_weights()
        except Exception as e:
            print(status_error(f"Failed to download model weights: {e}"))
            return False
        time.sleep(0.5)

        # Create default config
        print(status_processing("Creating configuration..."))
        try:
            core.create_default_config()
        except Exception as e:
            print(status_error(f"Failed to create configuration: {e}"))
            return False
        time.sleep(0.5)

        # Install required Python packages via pip3
        print(status_processing("Installing Python dependencies via pip3..."))
        try:
            subprocess.run(
                ["pip3", "install", "ffmpeg-python", "yt_dlp", "IPython", "opencv-python"],
                check=True
            )
        except Exception as e:
            print(status_error(f"Failed to install Python dependencies: {e}"))
            return False
        time.sleep(0.5)

        # Check if setup is complete
        if core.check_setup():
            print(status_success("DeOldify setup complete!"))
            time.sleep(1)
            return True
        else:
            print(status_error("Setup incomplete. Please try again."))
            return False
    except Exception as e:
        print(status_error(f"Setup error: {e}"))
        traceback.print_exc()
        time.sleep(1)
        return False

# Menu functions with improved styling
def display_main_menu():
    """Display the main menu options with improved styling"""
    print_header()

    menu_width = 40
    padding = max(0, (TERM_WIDTH - menu_width) // 2)
    pad = " " * padding

    # Define menu items
    items = ["Colorize a video", "Settings", "About", "Exit"]

    # Draw top border
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (menu_width-2)}┐{Colors.RESET}")

    # Title row
    title = "Main Menu"
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{title:^{menu_width-2}}│{Colors.RESET}")

    # Separator
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}├{'─' * (menu_width-2)}┤{Colors.RESET}")

    # Menu entries
    for idx, item in enumerate(items, start=1):
        # Build entry text and pad to width
        entry = f"{idx}. {item}"
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{entry: <{menu_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")

    # Bottom border
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (menu_width-2)}┘{Colors.RESET}")

    print_footer()
    choice = input(f"\n{Colors.ACCENT}Enter your choice (1-{len(items)}): {Colors.RESET}")
    return choice

def select_video_file():
    """Select a video file or enter a new path."""
    print_header()
    
    # Create a nice header
    menu_width = 50
    padding = (TERM_WIDTH - menu_width) // 2
    pad = " " * padding
    
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (menu_width-2)}┐{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{f' Colorize Video ':^{menu_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (menu_width-2)}┘{Colors.RESET}\n")
    
    print(f"  {Colors.ACCENT}n.{Colors.RESET} Enter a video file path")
    print(f"  {Colors.ACCENT}b.{Colors.RESET} Back to Main Menu\n")
    
    choice = input(f"Choose an option: ")
    
    if choice.lower() == 'b':
        return None
    
    if choice.lower() == 'n':
        file_path = input(f"\nEnter the path to a video file: ")
        if not file_path:
            return None
        
        # Validate the file exists
        if not os.path.exists(file_path):
            print(status_error(f"File not found: {file_path}"))
            time.sleep(2)
            return select_video_file()  # Try again
            
        # Update recent files in config
        config = core.load_config()
        core.update_recent_files(file_path, config)
        
        return file_path
    
    print(status_error("Invalid selection."))
    time.sleep(1)
    return select_video_file()  # Recursive call to try again

def colorize_video_workflow(input_file):
    """Handle the video colorization workflow."""
    
    # Redirect all DeOldify and other logging to a file
    import logging
    # Force all logging to go to a file instead of stderr
    logging.basicConfig(filename='/tmp/colorrevive.log', level=logging.WARNING, force=True)
    
    # Store the original stdout and stderr
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Clear screen before starting new workflow
    clear_screen()
    
    # Get video info
    video_info = core.get_video_info(input_file)
    if not video_info:
        print(status_error(f"Could not read video information from {input_file}"))
        input("\nPress Enter to return to main menu...")
        return
    
    # Load config
    config = core.load_config()
    
    # Print header with video info
    print_header()
    
    # Display processing options in a box
    options_width = 70
    padding = (TERM_WIDTH - options_width) // 2
    pad = " " * padding
    
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (options_width-2)}┐{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{f' Processing Options ':^{options_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (options_width-2)}┘{Colors.RESET}\n")

    # Display video information as simple lines below the processing options box
    print(f"{pad}  {Colors.SUBTLE}Video:{Colors.RESET} {Path(input_file).name}")
    
    # Format video info
    duration_str = format_time(video_info.get('duration', 0))
    resolution = f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
    frame_count = video_info.get('frame_count', 0)
    fps = video_info.get('fps', 0)
    
    print(f"{pad}  {Colors.SUBTLE}Duration:{Colors.RESET} {duration_str}")
    print(f"{pad}  {Colors.SUBTLE}Resolution:{Colors.RESET} {resolution}")
    print(f"{pad}  {Colors.SUBTLE}Frames:{Colors.RESET} {frame_count}")
    print(f"{pad}  {Colors.SUBTLE}FPS:{Colors.RESET} {fps:.2f}")
    print()  # Add an empty line for spacing

    # Get render factor with validation
    render_factor = config.get("render_factor", 35)
    while True:
        render_input = input(f"{Colors.ACCENT}Render Factor{Colors.RESET} (10-40, higher = better quality but slower) [{render_factor}]: ")
        if not render_input.strip():
            break
        if render_input.isdigit():
            val = int(render_input)
            if 10 <= val <= 40:
                render_factor = val
                break
        print(status_error("Invalid value. Must be an integer between 10 and 40."))

    # Get artistic mode with validation
    artistic_default = "y" if config.get("artistic_mode", True) else "n"
    while True:
        artistic_input = input(f"{Colors.ACCENT}Artistic Mode{Colors.RESET} (y/n, artistic = more vibrant colors) [{artistic_default}]: ").lower()
        if not artistic_input:
            artistic = config.get("artistic_mode", True)
            break
        if artistic_input in ('y', 'n'):
            artistic = artistic_input == 'y'
            break
        print(status_error("Invalid input. Enter 'y' or 'n'."))

    # Get output directory
    default_output = config.get("output_dir", str(Path.home() / "Videos" / "Colorized"))
    output_dir = input(f"{Colors.ACCENT}Output Directory{Colors.RESET} [{default_output}]: ")
    if not output_dir.strip():
        output_dir = default_output
    output_path = Path(output_dir)
    try:
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(status_error(f"Cannot create output directory: {e}"))
        time.sleep(2)
        return

    # Generate output filename
    input_name = Path(input_file).stem
    output_file = output_path / f"{input_name}_colorized.mp4"

    config["render_factor"] = render_factor
    config["artistic_mode"] = artistic
    config["output_dir"] = str(output_dir)
    core.save_config(config)
    core.update_recent_files(input_file, config)

    # Clear screen before starting processing
    clear_screen()
    print_header()
    
    # Create a nice processing header
    processing_width = 70
    padding = (TERM_WIDTH - processing_width) // 2
    pad = " " * padding
    
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (processing_width-2)}┐{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{f' Processing Video: {Path(input_file).name}':^{processing_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (processing_width-2)}┘{Colors.RESET}\n")
    
    print(f"{Colors.SUBTLE}Output will be saved to:{Colors.RESET} {output_file}\n")
    start_time = time.time()

    # Processing state
    process_done = threading.Event()
    processing_completed = [False]
    process_success = [False]
    process_exception = [None]

    # Progress tracking data structure
    progress_data = {
        'total': video_info.get('frame_count', 100),  # Use actual frame count if available
        'current': 0,
        'message': 'Initializing...',
        'last_update': time.time(),
        'temp_messages': []  # Store temporary messages
    }

    # Progress callback for core
    def progress_callback(progress, total, message):
        # Skip unwanted messages completely
        if (message.startswith("INFO:") or
            "Apple Silicon" in message or
            "device: mps" in message or
            "workers" in message or
            "Colorizing" in message and "frames using" in message or
            "Colorization cancelled" in message):
            return
            
        if total > 0:
            progress_data['total'] = total
            progress_data['current'] = min(progress, total)
            progress_data['message'] = message
            progress_data['last_update'] = time.time()

    # Completion callback for core
    def completion_callback(success, message, output_file=None):
        process_success[0] = success
        processing_completed[0] = True
        if success:
            progress_data['current'] = progress_data['total']
        progress_data['message'] = message
        process_done.set()

    # Processing thread
    def process_thread_func():
        try:
            # Disable any other logging output from core
            core.verbose = False
            if hasattr(core, 'suppress_output'):
                core.suppress_output = True
                
            # Process the video
            core.process_video(
                input_file, str(output_file), render_factor, artistic,
                config.get("auto_clean", True),
                progress_callback, completion_callback
            )
        except Exception as e:
            process_success[0] = False
            process_exception[0] = e
            progress_data['message'] = f"Error: {e}"
            processing_completed[0] = True
            process_done.set()

    # Register global progress callback in core
    core.progress_callback_func = progress_callback
    threading.Thread(target=process_thread_func, daemon=True).start()

    # ---- IMPROVED SINGLE-LINE PROGRESS BAR IMPLEMENTATION ----
    try:
        last_current = -1
        last_message = ""
        last_frame_num = 0
        
        # Print initial empty progress bar
        while not process_done.is_set():
            # Skip fake progress during initialization phase
            if progress_data['message'].lower().startswith('initializing'):
                time.sleep(0.1)
                continue
                
            # Skip unwanted messages
            message = progress_data['message']
            if (message.startswith("INFO:") or
                "Apple Silicon" in message or
                "device: mps" in message or
                "workers" in message or
                "Colorizing" in message and "frames using" in message):
                time.sleep(0.1)
                continue
                
            current = max(0, progress_data['current'])
            total = max(1, progress_data['total'])

            # Only update when there's a change
            if current != last_current or message != last_message:
                elapsed = time.time() - start_time
                
                # Clear the current line and write the new progress
                original_stdout.write(CLEAR_LINE)
                
                # Create the progress bar without ETA and time information
                if "Colorizing frame" in message:
                    # Extract frame number from message if possible
                    frame_num = 0
                    try:
                        frame_part = message.split("frame")[1].strip().split("/")[0]
                        frame_num = int(frame_part)
                        last_frame_num = frame_num
                    except (ValueError, IndexError):
                        frame_num = last_frame_num
                    
                    total_frames = video_info.get('frame_count', total)
                    display_message = "Colorizing"
                    
                    # Calculate percentage based on actual frame number
                    percentage = min(100, int(100 * frame_num / total_frames)) if total_frames > 0 else 0
                    
                    bar = progress_bar(
                        percentage, 100,  # Use percentage for the bar
                        prefix=f"{display_message:15}",
                        suffix=f"[{frame_num}/{total_frames}]",
                        length=40
                    )
                elif not message.startswith("Extracting frames"):  # Skip extraction progress
                    # Generic progress bar without ETA
                    bar = progress_bar(
                        current, total,
                        prefix=f"{message[:15]:15}",
                        suffix=f"[{current}/{total}]",
                        length=40
                    )
                else:
                    # Skip displaying extraction frames progress
                    time.sleep(0.1)
                    continue

                # Write the new progress bar and flush immediately
                original_stdout.write(bar)
                original_stdout.flush()

                last_current = current
                last_message = message

            time.sleep(0.1)

        # Final display at 100%
        elapsed = time.time() - start_time
        bar = progress_bar(
            progress_data['total'], progress_data['total'],
            prefix=f"{progress_data['message'][:15]:15}",
            suffix=f"[{progress_data['total']}/{progress_data['total']}] Complete!",
            length=40
        )
        original_stdout.write(CLEAR_LINE)
        original_stdout.write(bar + "\n\n")  # Add newlines after completion
        original_stdout.flush()

    except KeyboardInterrupt:
        # Handle user cancellation
        original_stdout.write(CLEAR_LINE)  # Clear the current line first
        original_stdout.write(status_warning("Processing cancelled by user") + "\n")
        original_stdout.flush()
        
        # Set a flag in the core module to stop processing
        if hasattr(core, 'cancel_processing'):
            core.cancel_processing = True
        
        # Signal the process to stop
        process_success[0] = False
        process_done.set()
        
        # Suppress any further INFO messages
        core.progress_callback_func = lambda *args, **kwargs: None
        
        # Force terminate any running processes
        try:
            # This is a more aggressive approach to ensure processing stops
            import signal
            os.kill(os.getpid(), signal.SIGTERM)
        except:
            pass
            
        # Wait a moment for the process to clean up
        time.sleep(0.5)
        
        original_stdout.write(status_error("Video processing failed!") + "\n")
        original_stdout.write(f"Failed after {format_time(time.time() - start_time)}\n")
        original_stdout.flush()
        
        # Clear any pending output before showing the prompt
        original_stdout.write("\nPress Enter to return to main menu...")
        original_stdout.flush()
        input()
        return
    finally:
        # IMPORTANT: Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # Processing completed
    elapsed_time = time.time() - start_time
    
    if process_success[0]:
        print(status_success("Video processing completed successfully!"))
        print(f"Completed in {format_time(elapsed_time)}")
        print(f"\nOutput saved to: {output_file}")
    else:
        print(status_error("Video processing failed!"))
        if process_exception[0]:
            print(f"Error: {process_exception[0]}")
        print(f"Failed after {format_time(elapsed_time)}")
    
    print("\nPress Enter to return to main menu...", end="", flush=True)
    input()
def display_settings():
    """Display and modify settings with improved UI"""
    config = core.load_config()
    
    while True:
        print_header()
        
        # Create a nice framed settings menu
        settings_width = 70
        padding = (TERM_WIDTH - settings_width) // 2
        pad = " " * padding
        
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (settings_width-2)}┐{Colors.RESET}")
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{' Settings':^{settings_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}├{'─' * (settings_width-2)}┤{Colors.RESET}")
        
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}1.{Colors.RESET} Default Render Factor: {config.get('render_factor', 35)}{' ' * (settings_width - 30 - len(str(config.get('render_factor', 35))))}│")
        
        artistic_text = 'On' if config.get('artistic_mode', True) else 'Off'
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}2.{Colors.RESET} Default Artistic Mode: {artistic_text}{' ' * (settings_width - 28 - len(artistic_text))}│")
        
        output_dir = config.get('output_dir', str(Path.home() / 'Videos' / 'Colorized'))
        # Truncate output_dir if too long
        if len(output_dir) > settings_width - 35:
            output_dir = "..." + output_dir[-(settings_width - 38):]
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}3.{Colors.RESET} Default Output Directory: {output_dir}{' ' * (settings_width - 31 - len(output_dir))}│")
        
        auto_clean_text = 'On' if config.get('auto_clean', True) else 'Off'
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}4.{Colors.RESET} Auto-clean Workspace: {auto_clean_text}{' ' * (settings_width - 27 - len(auto_clean_text))}│")
        
        preview_quality = config.get('preview_quality', 'medium').capitalize()
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}5.{Colors.RESET} Default Preview Quality: {preview_quality}{' ' * (settings_width - 30 - len(preview_quality))}│")
        
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}6.{Colors.RESET} Reset to Defaults{' ' * (settings_width - 20)}│")
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET} {Colors.PRIMARY}7.{Colors.RESET} Back to Main Menu{' ' * (settings_width - 20)}│")
        print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (settings_width-2)}┘{Colors.RESET}")
        
        print_footer()
        choice = input(f"\n{Colors.ACCENT}Enter your choice (1-7): {Colors.RESET}")
        
        if choice == '1':
            # Change render factor
            value = input(f"{Colors.ACCENT}Enter new default render factor (10-40): {Colors.RESET}")
            if value.isdigit() and 10 <= int(value) <= 40:
                config['render_factor'] = int(value)
                print(status_success("Default render factor updated"))
            else:
                print(status_error("Invalid value. Must be between 10 and 40"))
        
        elif choice == '2':
            # Change artistic mode
            value = input(f"{Colors.ACCENT}Enable artistic mode by default? (y/n): {Colors.RESET}").lower()
            if value in ('y', 'n'):
                config['artistic_mode'] = (value == 'y')
                print(status_success("Default artistic mode updated"))
            else:
                print(status_error("Invalid input. Enter 'y' or 'n'"))
        
        elif choice == '3':
            # Change output directory
            value = input(f"{Colors.ACCENT}Enter new default output directory: {Colors.RESET}")
            if value.strip():
                config['output_dir'] = value
                print(status_success("Default output directory updated"))
            else:
                print(status_error("Directory cannot be empty"))
        
        elif choice == '4':
            # Change auto-clean
            value = input(f"{Colors.ACCENT}Enable auto-clean by default? (y/n): {Colors.RESET}").lower()
            if value in ('y', 'n'):
                config['auto_clean'] = (value == 'y')
                print(status_success("Auto-clean setting updated"))
            else:
                print(status_error("Invalid input. Enter 'y' or 'n'"))
        
        elif choice == '5':
            # Change preview quality
            print("\nAvailable preview qualities:")
            print(f"  {Colors.PRIMARY}l{Colors.RESET} - Low (faster)")
            print(f"  {Colors.PRIMARY}m{Colors.RESET} - Medium (balanced)")
            print(f"  {Colors.PRIMARY}h{Colors.RESET} - High (slower)")
            
            value = input(f"{Colors.ACCENT}Enter preview quality (l/m/h): {Colors.RESET}").lower()
            if value in ('l', 'm', 'h'):
                quality_map = {'l': 'low', 'm': 'medium', 'h': 'high'}
                config['preview_quality'] = quality_map[value]
                print(status_success("Preview quality updated"))
            else:
                print(status_error("Invalid input. Enter 'l', 'm', or 'h'"))
        
        elif choice == '6':
            # Reset to defaults
            confirm = input(f"{Colors.WARNING}Are you sure you want to reset all settings to defaults? (y/n): {Colors.RESET}").lower()
            if confirm == 'y':
                core.create_default_config()
                config = core.load_config()
                print(status_success("Settings reset to defaults"))
            else:
                print(status_info("Reset cancelled"))
        
        elif choice == '7':
            # Save settings and return to main menu
            core.save_config(config)
            return
        
        else:
            print(status_error("Invalid choice. Please try again."))
        
        # Save settings after each change
        core.save_config(config)
        time.sleep(1)

def display_about():
    """Display about information with improved styling"""
    print_header()
    
    # Create a nice framed about section
    about_width = 70
    padding = (TERM_WIDTH - about_width) // 2
    pad = " " * padding
    
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}┌{'─' * (about_width-2)}┐{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}{' About DeOldify':^{about_width-2}}{Colors.HEADING}{Colors.BOLD}│{Colors.RESET}")
    print(f"{pad}{Colors.HEADING}{Colors.BOLD}└{'─' * (about_width-2)}┘{Colors.RESET}\n")
    
    about_text = [
        f"{Colors.TEXT}DeOldify is an open-source deep learning model that colorizes and restores old",
        f"black and white images and videos. The project was created by Jason Antic and has",
        f"been made available for public use.{Colors.RESET}",
        "",
        f"{Colors.PRIMARY}{Colors.BOLD}Features:{Colors.RESET}",
        f"• Automated video colorization using deep learning",
        f"• Support for various video formats and resolutions",
        f"• Artistic mode for more vibrant colorization",
        f"• Quality control through render factor adjustment",
        "",
        f"{Colors.PRIMARY}{Colors.BOLD}How it works:{Colors.RESET}",
        f"DeOldify uses a type of GAN (Generative Adversarial Network) called NoGAN,",
        f"which combines traditional GAN training with transfer learning approaches.",
        f"This allows for high-quality colorization while maintaining stability.",
        "",
        f"{Colors.PRIMARY}{Colors.BOLD}Credits:{Colors.RESET}",
        f"• Original DeOldify model by Jason Antic",
        f"• CLI interface improvements by the DeOldify community",
        "",
        f"{Colors.SUBTLE}For more information, visit: https://github.com/jantic/DeOldify{Colors.RESET}"
    ]
    
    for line in about_text:
        print(f"{pad}{line}")
    
    print_footer()
    input(f"\n{Colors.ACCENT}Press Enter to return to main menu...{Colors.RESET}")

def main():
    """Main entry point for the application."""
    # Check if setup is required
    if not core.check_setup():
        if not setup_environment():
            print(status_error("Setup failed. Please try again."))
            sys.exit(1)
    
    try:
        while True:
            choice = display_main_menu()
            
            if choice == '1':  # Colorize a video
                input_file = select_video_file()
                if input_file:
                    colorize_video_workflow(input_file)
            elif choice == '2':  # Settings
                settings_menu()
            elif choice == '3':  # About
                show_about()
            elif choice == '4' or choice.lower() == 'q':  # Exit
                print_centered("Thank you for using ColorRevive!", Colors.ACCENT)
                time.sleep(0.5)
                sys.exit(0)
            else:
                print(status_error("Invalid choice. Please try again."))
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nExiting ColorRevive. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    try:
        # Set up argument parser for command line options
        parser = argparse.ArgumentParser(description='DeOldify Video Colorizer CLI')
        parser.add_argument('--input', '-i', help='Input video file path')
        parser.add_argument('--output', '-o', help='Output video file path')
        parser.add_argument('--render-factor', '-r', type=int, help='Render factor (10-40)')
        parser.add_argument('--artistic', '-a', action='store_true', help='Use artistic mode')
        parser.add_argument('--setup', '-s', action='store_true', help='Force setup even if already configured')
        parser.add_argument('--version', '-v', action='store_true', help='Display version information')
        
        args = parser.parse_args()
        
        # Check for version flag
        if args.version:
            print(f"{Colors.PRIMARY}ColorRevive CLI v1.1{Colors.RESET}")
            sys.exit(0)
        
        # Check for setup flag
        if args.setup:
            setup_environment()
            sys.exit(0)
        
        # Check for direct video processing flags
        if args.input:
            config = core.load_config()
            
            # Use command line arguments or defaults from config
            input_file = args.input
            output_file = args.output or str(Path(Path(input_file).stem + "_colorized.mp4"))
            render_factor = args.render_factor or config.get('render_factor', 35)
            artistic = args.artistic or config.get('artistic_mode', True)
            
            print_header()
            print(status_info(f"Processing video: {input_file}"))
            print(status_info(f"Output file: {output_file}"))
            print(status_info(f"Render factor: {render_factor}"))
            print(status_info(f"Artistic mode: {'On' if artistic else 'Off'}"))
            
            # Update recent files and save config
            core.update_recent_files(input_file, config)
            core.save_config(config)
            
            # Process video
            start_time = time.time()
            try:
                success = core.process_video(
                    input_file, output_file, render_factor, artistic,
                    config.get('auto_clean', True)
                )
                elapsed = time.time() - start_time
                
                if success:
                    print(status_success(f"Video processing completed in {format_time(elapsed)}"))
                    print(status_success(f"Output saved to: {output_file}"))
                else:
                    print(status_error("Video processing failed!"))
            except Exception as e:
                print(status_error(f"Error processing video: {e}"))
                traceback.print_exc()
            
            sys.exit(0)
        
        # Run the interactive CLI
        main()
    except KeyboardInterrupt:
        print(f"\n{status_info('DeOldify CLI terminated by user.')}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{status_error(f'An unexpected error occurred: {e}')}")
        traceback.print_exc()
        sys.exit(1)
