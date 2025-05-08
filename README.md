# ColorRevive
ColorRevive is a user‑friendly terminal application for colorizing black & white videos using the DeOldify AI model. 

## About

ColorRevive is a user-friendly CLI application that colorizes black & white videos using the DeOldify AI model.

## Features

- Colorize black & white videos using DeOldify AI
- Easy-to-use terminal interface
- Supports various video formats
- Fast processing with GPU support

## 💾 Installation

1. Clone the repo
2. Install dependencies via `pip install -r requirements.txt`

## Setup

Run the setup script to initialize any required files and directories:

```bash
sh initial_setup.sh
```

## Usage

### Interactive Mode

Launch the application and navigate the menu:

```bash
python cli.py
```

### One-Shot Processing

Colorize a single video file without the interactive menu:

```bash
python cli.py --input path/to/old_video.mp4 --output path/to/colorized.mp4
```

## Configuration

Default settings are stored in `config.yaml`. You can adjust:

- `render_factor`: Controls color intensity (e.g., 10–40)
- `artistic_mode`: `true` or `false`
- `output_dir`: Directory for colorized videos
- `auto_clean`: `true` or `false` to remove temporary files
- `preview_quality`: `low`, `medium`, or `high`

---

Proudly made with the help of AI. Credits to DeOldify.
