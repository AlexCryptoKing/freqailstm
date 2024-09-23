from typing import Dict, Any

import torch

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchLSTMTrainer


class PyTorchLSTMRegressor(BasePyTorchRegressor):
    """
    PyTorchLSTMRegressor uses a PyTorch LSTM model to predict a continuous target variable.

    Model Training Parameters:
        "learning_rate": 3e-3,
        "trainer_kwargs": {
            "n_steps": null,
            "batch_size": 32,
            "n_epochs": 10,
        },
        "model_kwargs": {
            "num_lstm_layers": 3,
            "hidden_dim": 128,
            "window_size": 5,
            "dropout_percent": 0.4
        }
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        """Return the default PyTorch data convertor."""
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PyTorchLSTMRegressor with model and trainer parameters.
        """
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        
        self.learning_rate: float = config.get("learning_rate", 3e-3)
        self.model_kwargs: Dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: Dict[str, Any] = config.get("trainer_kwargs", {})
        self.window_size = self.model_kwargs.get('window_size', 5)

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate model and training parameters."""
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {self.learning_rate}")

        if not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError(f"Invalid window size: {self.window_size}")

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Train the LSTM model using the provided data.

        :param data_dictionary: Dictionary containing training data features and labels.
        :param dk: An instance of FreqaiDataKitchen providing data utilities.
        :return: The trained model.
        """
        try:
            n_features = data_dictionary["train_features"].shape[-1]
            model = PyTorchLSTMModel(input_dim=n_features, output_dim=1, **self.model_kwargs)
            model.to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.MSELoss(reduction='mean')

            trainer = self.get_init_model(dk.pair)
            if trainer is None:
                trainer = PyTorchLSTMTrainer(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=self.device,
                    data_convertor=self.data_convertor,
                    tb_logger=self.tb_logger,
                    window_size=self.window_size,
                    **self.trainer_kwargs,
                )

            trainer.fit(data_dictionary, self.splits)
            self.model = trainer
            return trainer

        except Exception as e:
            print(f"An error occurred during model training: {e}")
            raise
