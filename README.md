[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# A Trading Model Utilizing a Dynamic Weighting and Aggregate Scoring System with LSTM Networks

A regression model and trading strategy for  [FreqAI](https://www.freqtrade.io/en/stable/freqai/) module
from [freqtrade](https://github.com/freqtrade/freqtrade), a crypto trading bot.


## Overview

This project aims to develop a trading model that utilizes a dynamic weighting and aggregate scoring system to make more informed trading decisions. The model was initially built using TensorFlow and the Keras API, but has been ported to PyTorch to take advantage of its better GPU support across platforms and faster development process.

## Quick Start

1. Clone the repository

```shell
git clone https://github.com/AlexCryptoKing/freqailstm.git
```
2. Copy the files to the freqtrade directory

```shell 
cp torch/BasePyTorchModel.py <freqtrade dir>/freqtrade/freqai/base_models/
cp torch/PyTorchLSTMModel.py <freqtrade dir >/freqtrade/freqai/torch/
cp torch/PyTorchModelTrainer.py <freqtrade dir>/freqtrade/freqai/torch/
cp torch/PyTorchLSTMRegressor.py <freqtrade dir>/user_data/freqaimodels/
cp config-torch_example.json <freqtrade dir>/user_data/config.json
cp V8/1HOUR/AlexStrategyFinalV8.json <freqtrade dir>/user_data/strategies/
cp V8/1HOUR/AlexStrategyFinalV8.py <freqtrade dir>/user_data/strategies/
cp V8/1HOUR/AlexStrategyFinalV8Hyper.json <freqtrade dir>/user_data/strategies/
cp V8/1HOUR/AlexStrategyFinalV8Hyper.py <freqtrade dir>/user_data/strategies/
cp V8/1HOUR/config-torch.json <freqtrade dir>/user_data/

```
3. Download the data minimum 1 Month!
```shell
freqtrade download-data -c user_data/config-torch.json --timerange 20240701-20240801 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase
```
4. Edit "freqtrade/configuration/config_validation.py"
```python
...
def _validate_freqai_include_timeframes()
...
    if freqai_enabled:
        main_tf = conf.get('timeframe', '5m') -> change to '15m/30m/1h' or the **min** timeframe of your choosing
```
5. Make sure your package is edible after the the changes
```shell
pip install -e .
```

7. Run the backtest chosse a timeframe minimum 2-3months
```shell
freqtrade backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240601-20240801 
````

## Quick Start with docker

1. Clone the repository

```shell
git clone [https://github.com/AlexCryptoKing/freqailstm.git]
```
2. Build local docker images

```shell
cd freqailstm
docker build -f torch/Dockerfile  -t freqai .
```
3. Download data and Run the backtest
```
docker run -v ./data:/freqtrade/user_data/data  -it freqai  download-data -c user_data/config-torch.json --timerange 2020601-2024081 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase

docker run -v ./data:/freqtrade/user_data/data  -it freqai  backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240701-20240801 
```

## Model Architecture
config.json 

The core of the model is a Long Short-Term Memory (LSTM) network, which is a type of recurrent neural network that excels at handling sequential data and capturing long-term dependencies.

The LSTM model (PyTorchLSTMModel) has the following architecture:

1. The input data is passed through a series of LSTM layers (the number of layers is configurable via the `num_lstm_layers` parameter.). Each LSTM layer is followed by a Batch Normalization layer and a Dropout layer for regularization.
2. The output from the last LSTM layer is then passed through a fully connected layer with ReLU activation.
3. An Alpha Dropout layer is applied for additional regularization.
4. Finally, the output is passed through another fully connected layer to produce the final predictions.

The model's hyperparameters, such as the number of LSTM layers, hidden dimensions, dropout rates, and others, can be easily configured through the `model_kwargs` parameter in the `model_training_parameters` section of the configuration file.

Here's an example of how the model_training_parameters can be set up:

```json
"model_training_parameters": {
    "learning_rate": 3e-3,
    "trainer_kwargs": {
    "n_steps": null,
    "batch_size": 32,
    "n_epochs": 10,
    },
    "model_kwargs": {
    "num_lstm_layers": 3,
    "hidden_dim": 128,
    "dropout_percent": 0.4,
    "window_size": 5
    }
}
```
Let's go through each of these parameters:

- `learning_rate`: This is the learning rate used by the optimizer during training. It controls the step size at which the model's weights are updated in response to the estimated error each time the model weights are updated.
- `trainer_kwargs`: These are keyword arguments passed to the `PyTorchLSTMTrainer` which is located in PyTorchModelTrainer.
    - `n_steps`: The number of training iterations. If set to null, the number of epochs (n_epochs) will be used instead.
    - `batch_size`: The number of samples per gradient update.
    -  `n_epochs`: The number of times to iterate over the dataset.
- `model_kwargs`: These are keyword arguments passed to the `PyTorchLSTMModel`.
    - `num_lstm_layers`: The number of LSTM layers in the model.
    - `hidden_dim`: The dimensionality of the output space (i.e., the number of hidden units) in each LSTM layer.
    - `dropout_percent`: The dropout rate for regularization. Dropout is a technique used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
    - `window_size`: The number of time steps (or data points in the above case) to look back when making predictions.


## The Strategy

At its core, this strategy is all about making smart trading decisions by looking at the market from different angles. It's like having a team of experts, each focusing on a specific aspect of the market, and then combining their insights to make a well-informed decision.

Here's how it works:

1. **Indicators**: The strategy calculates a bunch of technical indicators, which are like different lenses to view the market. These indicators help identify trends, momentum, volatility, and other important market characteristics.

2. **Normalization**: To make sure all the indicators are on the same page. it normalizes them by calculating the z-score. This step ensures that the indicators are comparable and can be weighted appropriately.

3. **Dynamic Weighting**: The strategy is adaptable and can adjust the importance of different indicators based on market conditions.

4. **Aggregate Score**: All the normalized indicators are combined into a single score, which represents the overall market sentiment. Just like taking a vote among the experts to reach a consensus.

5. **Market Regime Filters**: The strategy considers the current market regime, whether it's bullish, bearish, or neutral. Looking up the weather before deciding on an outfit. 🌞🌧️?

6. **Volatility Adjustments**: It takes into account the market's volatility and adjusts the target score accordingly. We want to be cautious when the market is choppy and more aggressive when it's calm.

7. **Final Target Score**: All these factors are combined into a final target score, which is like a concise and informative signal for the LSTM model to learn from. It's like giving the model a clear and focused task to work on.

8. **Entry and Exit Signals**: we use the predicted target score and set thresholds to determine when to enter or exit a trade.

## Contributing
JOIN MY DISCORD - https://discord.gg/vfJQ5pftwX


Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an
issue or submit a pull request on the [GitHub repository](https://github.com/AlexCryptoKing/freqailstm.git).


