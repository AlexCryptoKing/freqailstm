# Use Freqtrade's base image for FreqAI RL
FROM freqtradeorg/freqtrade:develop_freqairl

# Set user to root for installing additional dependencies
USER root

# Copy any additional requirements (if needed)
#COPY requirements-freqai.txt requirements-freqai-rl.txt /freqtrade/

# Install additional dependencies via pip
RUN pip install --no-cache-dir optuna \
  && pip install --no-cache-dir torch \
  && pip install --no-cache-dir sb3_contrib \
  && pip install --no-cache-dir datasieve

# Copy the custom models and strategy files
COPY config-torch_example.json /freqtrade/user_data/config-torch.json
COPY V9/AlexStrategyFinalV90.py /freqtrade/user_data/strategies/
COPY torch/BasePyTorchModel.py /freqtrade/freqtrade/freqai/base_models/
COPY torch/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/
COPY torch/PyTorchModelTrainer.py /freqtrade/freqtrade/freqai/torch/
COPY torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/
COPY torch/PyTorchLSTMRegressor_Cuda.py /freqtrade/user_data/freqaimodels/

# Ensure the correct permissions are set for the copied files
RUN chown -R ftuser:ftuser /freqtrade

# Switch back to the ftuser for runtime
USER ftuser

# Install and initialize Freqtrade UI
RUN freqtrade install-ui

# Set Freqtrade as the default entrypoint
ENTRYPOINT ["freqtrade"]

# Default command to start in 'trade' mode
CMD ["trade"]
