#!/usr/bin/env bash
set -e

# initial_setup.sh
# Bootstrap a Miniforge-based conda environment for DeOldify on Apple Silicon macOS.
# After running this script, manually activate the environment and run deoldify_gui.py.

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
INSTALLER="$HOME/miniforge.sh"
PREFIX="$HOME/miniforge3"
ENV_NAME="deoldify"

echo "=== DeOldify Environment Setup ==="

# 1) Install Miniforge if missing
if [ ! -x "$PREFIX/bin/conda" ]; then
  echo "Installing Miniforge..."
  curl -L "$MINIFORGE_URL" -o "$INSTALLER"
  bash "$INSTALLER" -b -p "$PREFIX"
else
  echo "Miniforge already installed."
fi

# 2) Initialize conda in this shell
source "$PREFIX/etc/profile.d/conda.sh"
# 2b) Configure conda for interactive shells (Zsh)
echo "Configuring conda in your shell..."
"$PREFIX/bin/conda" init zsh
echo "Please restart your terminal or run 'source ~/.zshrc' to enable 'conda activate'."

# 3) Ensure the conda environment exists
echo "Ensuring conda environment '$ENV_NAME' exists..."
conda create -n "$ENV_NAME" python=3.9 -y || echo "Environment '$ENV_NAME' already exists; skipping creation."

# 4) Install or update core dependencies via conda-forge
echo "Installing core dependencies via conda-forge..."
conda install -n "$ENV_NAME" -y -c conda-forge \
    pytorch torchvision \
    pillow 'numpy<2' scipy pandas matplotlib fastprogress beautifulsoup4 numexpr spacy \
    opencv \
    ffmpeg git jpeg zlib textual requests pyyaml tqdm

# 5) Pip-install FastAI v1 (no-deps) via pip...
echo "Installing FastAI v1 (no-deps) and supplementary Python packages via pip..."
conda run -n "$ENV_NAME" pip install fastai==1.0.60 --no-deps ffmpeg-python yt_dlp IPython opencv-python pyyaml tqdm

echo "=== Setup complete ==="
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  python deoldify_gui.py"
