#!/bin/bash

echo "Installing application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python from your package manager or https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if setup.py exists
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found in current directory"
    exit 1
fi

# Run setup.py
echo "Running setup.py with $PYTHON_CMD..."
$PYTHON_CMD setup.py install

if [ $? -eq 0 ]; then
    echo
    echo "Installation completed successfully!"
else
    echo
    echo "Installation failed!"
    exit 1
fi