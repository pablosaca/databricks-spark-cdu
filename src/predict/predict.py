from typing import Union, Dict, List
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame as DF
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegressionModel
from lightgbm import LGBMClassifier

from src.utils.logger import get_logger

logger = get_logger()


class Predict(ABC):
    def __init__(
            self,
            model: Union[LogisticRegressionModel, LGBMClassifier],
            impute_values_dict: Dict[str, Union[float, dict]],
            categorical_features: List[str]
    ):
        self.model = model
        self.impute_values_dict = impute_values_dict
        self.categorical_features = categorical_features

        if not isinstance(self.model, (LGBMClassifier, LogisticRegressionModel)):
            msg = f"Formato icompatible del modelo. Uso `LGBMClassifier` (sckit-learn) " \
                  f"o `LogisticRegressionModel` (pyspark)"
            logger.error(msg)
            raise TypeError(msg)

    @abstractmethod
    def predict(self, df: DF) -> DF:
        """
        Predicción del modelo. Uso de Pandas UDF si el modelo es en scikit-learn
        """
        pass
