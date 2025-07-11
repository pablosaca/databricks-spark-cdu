from typing import Optional, List, Dict, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame as DF
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from lightgbm import LGBMClassifier

from src.utils.logger import get_logger

logger = get_logger()


class ClassificationModel(ABC):
    # TODO: REFACTORIZA LO QUE CONSIDERES DE LA CLASE Y SUS MÉTODOS
    def __init__(
            self,
            df: DF,
            model_framework: str = "pyspark",
            target: str = "Response",
            frac_sample: float = 0.7,
            seed: int = 123
    ):
        self.model_framework = model_framework

        api_fremework_available_list = ["scikit-learn", "pyspark"]
        if self.model_framework not in api_fremework_available_list:
            msg = f"Incorrecto framework utilizado: {api_fremework_available_list}"
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(df, DF):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        self.target = target
        self.frac_sample = frac_sample
        self.seed = seed

        self.model = None  # se define una vez entrenado el modelo

    @staticmethod
    def __get_metrics(
            y_true: Union[DF, pd.Series, np.ndarray],
            y_pred: Union[DF, pd.Series, np.ndarray],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calcula métricas de bondad de ajuste (accuracy, precision, recall y f1-score).
        Puedes utilizar otras métricas de clasificación si lo consideras
        """

        if isinstance(y_true, DF) and isinstance(y_pred, DF):
            y_true = y_true.toPandas()
            y_pred = y_pred.toPandas()

        elif isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
            y_true = y_true.copy()
            y_pred = y_pred.copy()

        elif type(y_true) != type(y_pred):
            msg = "y_true y y_pred deben ser del mismo tipo: ambos Spark dataFrame o ambos Pandas dataframe"
            logger.error(msg)
            raise TypeError(msg)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
        logger.info("Métricas de bondad de ajuste son calculadas")
        return metrics

    def get_train_val_sample(self, df: DF) -> Tuple[DF, DF]:
        """
        Obtiene la muestra de entrenamiento y de validación
        """
        if "id" not in df.columns:
            msg = "`id` ha sido eliminado del dataset. La columna de identificación debe mantenerse"
            logger.error(msg)
            raise ValueError(msg)
        # TODO: En este caso, utilizamos un muestro aleatorio simple pero puedes plantear posibles mejoras
        train_df = df.sample(fraction=self.frac_sample, seed=self.seed)
        val_df = df.join(train_df, on="id", how="left_anti")
        return train_df, val_df

    @abstractmethod
    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Union[DF, pd.DataFrame], Union[DF, pd.DataFrame]]:
        """
        Entrenamiento del modelo
        """
        pass


class ScikitLearnModel(ClassificationModel):
    # TODO: uso de un modelo de boosting -> LigthGBM pero puedes plantearte otro modelo
    def __init__(
            self,
            df: DF,
            model_framework: str = "scikit-learn",
            target: str = "Response",
            frac_sample: float = 0.7,
            categorical_features: Optional[List[str]] = None,
            lightgbm_params: Optional[Dict[str, int]] = None,
            seed: int = 123
    ):
        super().__init__(df, model_framework, target, frac_sample, seed)

        api_framework_available = "scikit-learn"
        if not self.model_framework == api_framework_available:
            logger.info(
                f"Incorrecto framework de modelización a utilizar. Solo puede usarse {api_framework_available}"
            )

        self.features = [col for col in df.columns if col not in ['target']]
        aux_cat_feats = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        model_params = {"n_estimators": 115, "learning_rate": 0.03}
        self.lightgbm_params = lightgbm_params if lightgbm_params is not None else model_params
        self.model = None  # se actualiza cuando se entrena el modelo

    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict[str, Union[float, np.ndarray]]]:
        """
        Entrenamiento del modelo de scikit-learn
        """
        if not isinstance(train_df, DF) or not isinstance(val_df, DF):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        logger.info("Conversión de los Spark dataframes a Pandas dataframes")
        train_pdf = train_df.toPandas()
        val_pdf = val_df.toPandas()

        X_train = train_pdf[self.features].copy()
        y_train = train_pdf[self.target]
        X_val = val_pdf[self.features].copy()
        y_val = val_pdf[self.target]

        X_train = self.__select_categorical_features(X_train)
        X_val = self.__select_categorical_features(X_val)

        logger.info("Entrenamiento de un modelo lightgbm en scikit-learn")
        self.__model_fitted(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        logger.info("Realización de las predicciones del modelo")

        metrics_train = self.__get_metrics(y_train, y_pred_train)
        metrics_val = self.__get_metrics(y_val, y_pred_val)
        logger.info("Obtenidas las métricas del modelo para la muestra de entrenamiento y validación")
        return metrics_train, metrics_val

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo LGBM a partir de la muestra de entrenamiento (feats and target column)
        """
        self.model = LGBMClassifier(
            n_estimators=self.lightgbm_params["n_estimators"],
            learning_rate=self.lightgbm_params["learning_rate"],
            random_state=self.seed,
            force_col_wise=True,
        )
        self.model.fit(
            X_train, y_train,
            categorical_feature=self.categorical_features
        )
        logger.info(
            "Modelo lightgbm ha sido entrenado "
            f'Número de estimadores {self.lightgbm_params["n_estimators"]} '
            f'y tasa de aprendizaje {self.lightgbm_params["learning_rate"]}'
        )

    def __select_categorical_features(self, df: Union[DF, pd.DataFrame]) -> Union[DF, pd.DataFrame]:
        """
        Se tienen en cuenta las variables categóricas.
        Debes implementar el método si usas la modelización en spark
        """

        if isinstance(df, pd.DataFrame):
            for col in self.categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    logger.info(f"La variable {col} es convertida a categórica")
        else:
            msg = "Este método es utilizado cuando el dataset es Pandas dataframe"
            logger.info(msg)
            raise TypeError(msg)
        return df


class PySparkModel(ClassificationModel):

    def __init__(
            self,
            df: DF,
            model_framework: str = "pyspark",
            target: str = "Response",
            frac_sample: float = 0.7,
            categorical_features: Optional[List[str]] = None,
            seed: int = 123
    ):
        super().__init__(df, model_framework, target, frac_sample, seed)

        api_framework_available = "pyspark"
        if not self.model_framework == api_framework_available:
            logger.info(
                f"Incorrecto framework de modelización a utilizar. Solo puede usarse {api_framework_available}"
            )

        self.features = [col for col in df.columns if col not in ['target']]
        aux_cat_feats = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        self.model = None  # se actualiza cuando se entrena el modelo

    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict[str, Union[float, np.ndarray]]]:
        """
        Entrenamiento del modelo
        """
        if not isinstance(train_df, DF) or not isinstance(val_df, DF):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.info(msg)
            raise TypeError(msg)

        logger.info("Entrenamiento de un modelo de regresión logística en spark")
        # TODO: DEFINIR EL CÓDIGO PARA IMPLEMENTAR EL ENTRENAMIENTO DE UNA REGRESIÓN LOGÍSTICA EN PYSPARK

        y_train = None
        y_pred_train = None
        y_val = None
        y_pred_val = None

        metrics_train = self.__get_metrics(y_train, y_pred_train)
        metrics_val = self.__get_metrics(y_val, y_pred_val)
        return metrics_train, metrics_val

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo LGBM a partir de la muestra de entrenamiento (feats and target column)
        """
        # TODO PSC: IMPLEMENTACIÓN DEL ENTRENAMIENTO DE UN MODELO DE REGRESIÓN LOGÍSTICA
        pass
