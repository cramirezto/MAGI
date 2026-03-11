#!/bin/bash
set -euo pipefail

# Only run in Claude Code on the web
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing protein visualization dependencies..."

# Install core packages from requirements.txt,
# skipping pymol-open-source (requires system libs not available in all envs)
pip install --quiet \
  numpy \
  scipy \
  pandas \
  biopython \
  biopandas \
  mdanalysis \
  rdkit \
  py3Dmol \
  nglview \
  matplotlib \
  plotly \
  seaborn \
  ipywidgets \
  requests

echo "All protein visualization packages installed successfully."
