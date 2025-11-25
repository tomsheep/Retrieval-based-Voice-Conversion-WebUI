#!/bin/sh

set -fa

# Check if Python is installed
if ! command -v python; then
  echo "Python not found. Please install Python using your package manager or via PyEnv."
  exit 1
fi

# Run the main script
python web.py --pycmd python
