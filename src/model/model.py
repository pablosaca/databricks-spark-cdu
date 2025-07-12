from typing import Optional, List, Dict, Tuple, Union
from abc import ABC, abstractmethod

import joblib
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


class ClassificationTrainer(ABC):
    # TODO: REFACTORIZA LO QUE CONSIDERES DE LA CLASE Y SUS MÉTODOS
    def __init__(
            self,
            model_framework: str = "pyspark",
            target: str = "Response",
            frac_sample: float = 0.7,
            seed: int = 123,
            file_name: str = "prueba_practica_santalucia",
            model_name: str = "ml_model",
            path: str = "/databricks/driver"
    ):
        self.model_framework = model_framework

        api_fremework_available_list = ["scikit-learn", "pyspark"]
        if self.model_framework not in api_fremework_available_list:
            msg = f"Incorrecto framework utilizado: {api_fremework_available_list}"
            logger.error(msg)
            raise ValueError(msg)

        self.target = target
        self.frac_sample = frac_sample
        self.seed = seed

        self.file_name = file_name
        self.model_name = model_name
        self.path = path

        self.model = None  # se define una vez entrenado el modelo

    @staticmethod
    def _get_metrics(
            df: Union[DF, pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calcula métricas de bondad de ajuste (accuracy, precision, recall y f1-score).
        Puedes utilizar otras métricas de clasificación si lo consideras
        """

        if isinstance(df, DF):
            df = df.toPandas()
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            msg = "El dataframe de predicciones no tiene el formato adecuado. Debe ser Spark / Pandas dataFrame"
            logger.error(msg)
            raise TypeError(msg)

        metrics = {
            "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
            "precision": precision_score(df["y_true"], df["y_pred"]),
            "recall": recall_score(df["y_true"], df["y_pred"]),
            "f1": f1_score(df["y_true"], df["y_pred"])
        }
        logger.info("Métricas de bondad de ajuste son calculadas")
        return metrics

    def get_train_val_sample(self, df: DF) -> Tuple[DF, DF]:
        """
        Obtiene la muestra de entrenamiento y de validación
        """

        if not isinstance(df, DF):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        if "id" not in df.columns:
            msg = "`id` ha sido eliminado del dataset. La columna de identificación debe mantenerse"
            logger.error(msg)
            raise ValueError(msg)

        # TODO: En este caso, utilizamos un muestro aleatorio simple pero puedes plantear posibles mejoras
        train_df = df.sample(fraction=self.frac_sample, seed=self.seed)
        val_df = df.join(train_df, on="id", how="left_anti")
        return train_df, val_df

    def _previous_check_train_model(self, train_df: DF, val_df: DF) -> None:
        """
        Chequeo formato
        """
        if not isinstance(train_df, DF) or not isinstance(val_df, DF):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        if self.target not in train_df.columns or self.target not in val_df.columns:
            msg = f"Error. La variable {self.target} no se encuentra en el dataset de entrenamiento o validación"
            logger.error(msg)
            raise ValueError(msg)

    @abstractmethod
    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Union[DF, pd.DataFrame], Union[DF, pd.DataFrame]]:
        """
        Entrenamiento del modelo
        """
        pass

    @abstractmethod
    def save_model(self):
        # TODO: borrar para que lo defina el candidato
        model_file = f"{self.path}/{self.file_name}/{self.model_name}.joblib"
        joblib.dump(self.model, model_file)


class ScikitLearnTrainer(ClassificationTrainer):
    # TODO: uso de un modelo de boosting -> LigthGBM pero puedes plantearte otro modelo
    def __init__(
            self,
            model_framework: str = "scikit-learn",
            target: str = "Response",
            frac_sample: float = 0.7,
            categorical_features: Optional[List[str]] = None,
            lightgbm_params: Optional[Dict[str, int]] = None,
            seed: int = 123,
            file_name: str = "prueba_practica_santalucia",
            model_name: str = "ml_model",
            path: str = "/databricks/driver"
    ):
        super().__init__(model_framework, target, frac_sample, seed, file_name, model_name, path)

        api_framework_available = "scikit-learn"
        if not self.model_framework == api_framework_available:
            logger.info(
                f"Incorrecto framework de modelización a utilizar. Solo puede usarse {api_framework_available}"
            )

        aux_cat_feats = ["Gender", "Vehicle_Age", "Vehicle_Damage", "Quality_of_Life"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        model_params = {"n_estimators": 115, "learning_rate": 0.03}
        self.lightgbm_params = lightgbm_params if lightgbm_params is not None else model_params
        self.model = None  # se actualiza cuando se entrena el modelo

    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict[str, Union[float, np.ndarray]]]:
        """
        Entrenamiento del modelo de scikit-learn partiendo de un modelo lightgbm
        # TODO: se dispone de un proceso de entrenamiento estándar
        """
        self._previous_check_train_model(train_df, val_df)
        features_model = [col for col in train_df.columns if col not in self.target]
        logger.info(f"Obtenidas las features del modelo: {features_model}")

        logger.info("Conversión de los Spark dataframes a Pandas dataframes")
        train_pdf = train_df.toPandas()
        val_pdf = val_df.toPandas()

        X_train = train_pdf[features_model].copy()
        y_train = train_pdf[self.target]
        X_val = val_pdf[features_model].copy()
        y_val = val_pdf[self.target]

        logger.info("Identificación variables categóricas para usar en el modelo")
        X_train = self.__select_categorical_features(X_train)
        X_val = self.__select_categorical_features(X_val)

        logger.info("Entrenamiento de un modelo lightgbm usando scikit-learn")
        self.__model_fitted(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        logger.info("Realización de las predicciones del modelo")

        predictions_train_df = self.__model_predictions_format(y_train, y_pred_train)
        predictions_val_df = self.__model_predictions_format(y_val, y_pred_val)
        metrics_train = self._get_metrics(predictions_train_df)
        metrics_val = self._get_metrics(predictions_val_df)
        logger.info("Obtenidas las métricas del modelo para la muestra de entrenamiento y validación")
        return metrics_train, metrics_val

    @staticmethod
    def __model_predictions_format(y_real: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Paso de formato a pandas dataframe
        """
        if len(y_real) != len(y_pred):
            msg = "Dimensiones de y_real y y_pred no coinciden"
            logger.error(msg)
            raise ValueError(msg)

        result_df = pd.DataFrame(
            {
                "y_true": y_real.values,  # es una serie obtenemos el array
                "y_pred": y_pred.flatten()  # es un np.ndarray pero por si fuese (n,1)
             }
        )
        return result_df

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo LGBM a partir de la muestra de entrenamiento (feats and target column)
        # TODO: se dispone de un entrenamiento básico, se permite cualquier mejora al respecto en función de los datos de partida
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

    def save_model(self):
        """
        Guardado del modelo (el directorio se obtiene de los atributos de la clase)
        """
        # TODO: borrar para que lo defina el candidato
        model_file = f"{self.path}/{self.file_name}/{self.model_name}.joblib"
        joblib.dump(self.model, model_file)


class PySparkTrainer(ClassificationTrainer):

    def __init__(
            self,
            model_framework: str = "pyspark",
            target: str = "Response",
            frac_sample: float = 0.7,
            categorical_features: Optional[List[str]] = None,
            seed: int = 123,
            file_name: str = "prueba_practica_santalucia",
            model_name: str = "ml_model",
            path: str = "/databricks/driver"
    ):
        super().__init__(model_framework, target, frac_sample, seed, file_name, model_name, path)

        api_framework_available = "pyspark"
        if not self.model_framework == api_framework_available:
            logger.info(
                f"Incorrecto framework de modelización a utilizar. Solo puede usarse {api_framework_available}"
            )

        aux_cat_feats = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        self.model = None  # se actualiza cuando se entrena el modelo

    def train_model(
            self, train_df: DF, val_df: DF
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict[str, Union[float, np.ndarray]]]:
        """
        Entrenamiento del modelo
        """
        self._previous_check_train_model(train_df, val_df)

        logger.info("Entrenamiento de un modelo de regresión logística en spark")
        # TODO: DEFINIR EL CÓDIGO PARA IMPLEMENTAR EL ENTRENAMIENTO DE UNA REGRESIÓN LOGÍSTICA EN PYSPARK

        y_train = None
        y_pred_train = None
        y_val = None
        y_pred_val = None

        metrics_train = self._get_metrics(y_train, y_pred_train)
        metrics_val = self._get_metrics(y_val, y_pred_val)
        return metrics_train, metrics_val

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo LGBM a partir de la muestra de entrenamiento (feats and target column)
        """
        # TODO PSC: IMPLEMENTACIÓN DEL ENTRENAMIENTO DE UN MODELO DE REGRESIÓN LOGÍSTICA
        pass

    def save_model(self):
        """
        Guardado del modelo (el directorio se obtiene de los atributos de la clase)
        """
        # TODO: borrar para que lo defina el candidato
        pass
