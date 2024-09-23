#!/bin/bash

# Ensure the script is executed with root privileges
if [ "$EUID" -ne 0 ]
then
  echo "Please run as root or use sudo"
  exit
fi

# Clone the FreqAI LSTM repository
echo "Cloning the FreqAI LSTM repository..."
sudo git clone https://github.com/AlexCryptoKing/freqailstm.git

# Create necessary directories in ./user_data
echo "Creating directories in user_data..."
mkdir -p ./user_data/models
mkdir -p ./user_data/Notebooks
mkdir -p ./user_data/strategies
mkdir -p ./user_data/logs

cp -r ./freqailstm/V9/* ./user_data/strategies

# Copy contents from the cloned repo's user_data to the local user_data
echo "Copying data from freqailstm/user_data to ./user_data/..."
cp -r ./freqailstm/user_data/* ./user_data/

# Change permissions of the ./user_data folder and all its subdirectories to 0777
echo "Changing permissions of the user_data directory to 0777..."
chmod -R 0777 ./user_data

# Build the Docker image using the Dockerfile
echo "Building the Docker image with tag 'alexfreqai'..."
docker build -f Dockerfile -t alexfreqai .

echo "Setup complete."
