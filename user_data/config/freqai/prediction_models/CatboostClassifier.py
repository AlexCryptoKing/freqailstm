import logging
from pathlib import Path
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class CatboostClassifier(BaseClassifierModel):
    """
    User created prediction model. The class inherits IFreqaiModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        train_data = Pool(
            data=data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            test_data = None
        else:
            test_data = Pool(
                data=data_dictionary["test_features"],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
            )

        cbr = CatBoostClassifier(
            allow_writing_files=True,
            loss_function="MultiClass",
            train_dir=Path(dk.data_path),
            **self.model_training_parameters,
        )

        init_model = self.get_init_model(dk.pair)

        cbr.fit(
            X=train_data,
            eval_set=test_data,
            init_model=init_model,
        )

        return cbr
