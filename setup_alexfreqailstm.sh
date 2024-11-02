#!/bin/bash

# Ensure the script is executed with root privileges
if [ "$EUID" -ne 0 ]
then
  echo "Please run as root or use sudo"
  exit
fi

# Clone the FreqAI LSTM repository if it doesn’t already exist
if [ ! -d "freqailstm" ]; then
  echo "Cloning the FreqAI LSTM repository..."
  git clone https://github.com/AlexCryptoKing/freqailstm.git
else
  echo "Repository already cloned."
fi

# Create necessary directories in ./user_data if they don’t already exist
echo "Creating directories in user_data..."
mkdir -p ./user_data/models
mkdir -p ./user_data/Notebooks
mkdir -p ./user_data/strategies
mkdir -p ./user_data/logs

# Copy strategies and data from the cloned repo to ./user_data
echo "Copying data from freqailstm to ./user_data/..."
cp -r ./freqailstm/V9/* ./user_data/strategies
cp -r ./freqailstm/user_data/* ./user_data/

# Set permissions for the ./user_data folder
echo "Changing permissions of the user_data directory to 0777..."
chmod -R 0777 ./user_data

# Build the Docker image using the Dockerfile
echo "Building the Docker image with tag 'alexfreqai'..."
docker build -f Dockerfile -t alexfreqai .

echo "Setup complete."
