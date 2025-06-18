#!/bin/bash
# Setup script for deploying a Streamlit application on a Linux server

# Ensure the script is run with sudo privileges
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

echo "Updating system packages..."
sudo apt-get update

echo "Installing system dependencies required for the application..."
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

echo "Installing Python dependencies..."
pip install opencv-python-headless==4.9.0.80

# If you have additional Python packages, consider adding a requirements.txt file
# pip install -r requirements.txt

echo "Setup completed successfully."
