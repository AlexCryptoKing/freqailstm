# Use Freqtrade's base image for FreqAI RL
FROM freqtradeorg/freqtrade:stable_freqairl

# Set user to root for installing additional dependencies
USER root

# Install system dependencies
RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

# Install additional Python dependencies via pip
RUN pip install --no-cache-dir optuna \
  && pip install --no-cache-dir torch \
  && pip install --no-cache-dir sb3_contrib \
  && pip install --no-cache-dir datasieve

# Copy contents from the host machine’s freqailstm directory to the container
COPY ./freqailstm /freqtrade

# Ensure the correct permissions are set for the copied files
RUN chown -R ftuser:ftuser /freqtrade

# Switch back to the ftuser for runtime
USER ftuser

# Install and initialize Freqtrade UI
RUN freqtrade install-ui

# Set Freqtrade as the default entrypoint
ENTRYPOINT ["freqtrade"]

# Copy additional user_data files and strategies from freqailstm
RUN mkdir -p /freqtrade/user_data/strategies /freqtrade/user_data/freqaimodels \
    && cp torch/BasePyTorchModel.py /freqtrade/freqtrade/freqai/base_models/ \
    && cp torch/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/ \
    && cp torch/PyTorchModelTrainer.py /freqtrade/freqtrade/freqai/torch/ \
    && cp torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/ \
    && cp torch/PyTorchLSTMRegressor_Cuda.py /freqtrade/user_data/freqaimodels/ \
    && cp config/config-torch.json /freqtrade/user_data/ \
    && cp V9/* /freqtrade/user_data/strategies/

# Default command to start in 'trade' mode
CMD ["trade", "--logfile", "/freqtrade/user_data/logs/freqtrade.log", "--db-url", "sqlite:////freqtrade/user_data/tradesv3.sqlite", "--config", "/freqtrade/user_data/config-torch.json", "--strategy", "AlexStrategyFinalV9"]
