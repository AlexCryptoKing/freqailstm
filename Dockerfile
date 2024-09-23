# Use Freqtrade's base image for FreqAI RL
FROM freqtradeorg/freqtrade:develop_freqairl

# Set user to root for installing additional dependencies
USER root

# Copy any additional requirements (if needed)
#COPY requirements-freqai.txt requirements-freqai-rl.txt /freqtrade/

RUN apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc

# Install additional dependencies via pip
RUN pip install --no-cache-dir optuna \
  && pip install --no-cache-dir torch \
  && pip install --no-cache-dir sb3_contrib \
  && pip install --no-cache-dir datasieve

ADD --keep-git-dir=true https://github.com/AlexCryptoKing/freqailstm.git /freqtrade
WORKDIR /freqtrade

# Ensure the correct permissions are set for the copied files
RUN chown -R ftuser:ftuser /freqtrade

# Switch back to the ftuser for runtime
USER ftuser

# Install and initialize Freqtrade UI
RUN freqtrade install-ui

# Set Freqtrade as the default entrypoint
ENTRYPOINT ["freqtrade"]

RUN cp torch/BasePyTorchModel.py /freqtrade/freqtrade/freqai/base_models/ \
    && cp torch/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/ \
    && cp torch/PyTorchModelTrainer.py /freqtrade/freqtrade/freqai/torch/ \
    && cp torch/PyTorchLSTMRegressor.py /freqtrade/user_data/freqaimodels/ \
    && cp torch/PyTorchLSTMRegressor_Cuda.py /freqtrade/user_data/freqaimodels/

	

# Default command to start in 'trade' mode
CMD ["trade"]
