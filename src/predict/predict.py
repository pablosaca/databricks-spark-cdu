from typing import List, Optional

import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from pyspark.sql import DataFrame as DF
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

from src.utils.logger import get_logger

logger = get_logger()


class Predict:
    # TODO: EL CANDIDATO PODRÁ REFACTORIZAR LO QUE CONSIDERE DE LA CLASE Y SUS MÉTODOS
    def __init__(
            self,
            model_framework: str = "spark-mllib",
            is_pipeline: bool = False,
            file_name: str = "prueba_practica_santalucia",
            model_name: str = "ml_model",
            path: str = "/databricks/driver",
            categorical_features: Optional[List[str]] = None
    ):

        self.model_framework = model_framework
        self.is_pipeline = is_pipeline

        api_fremework_available_list = ["scikit-learn", "spark-mllib"]
        if self.model_framework not in api_fremework_available_list:
            msg = f"Incorrecto framework utilizado: {api_fremework_available_list}"
            logger.error(msg)
            raise ValueError(msg)

        self.file_name = file_name
        self.model_name = model_name
        self.path = path

        aux_cat_feats = ["Gender", "Vehicle_Age", "Vehicle_Damage", "Quality_of_Life"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

    def load_model(self):
        """
        Carga del modelo diferenciando si es de scikit-learn o pyspark
        """
        model_path = f"{self.path}/{self.file_name}"
        if self.model_framework == "scikit-learn":
            model_file = f"{model_path}/{self.model_name}.joblib"
            model = joblib.load(model_file)
        else:
            if self.is_pipeline:
                model = PipelineModel.load(f"{model_path}/{self.model_name}")
            else:
                model = LogisticRegressionModel.load(f"{model_path}/{self.model_name}")
        return model

    def predict(self, df: DF) -> DF:
        """
        Predicción del modelo. Uso de Pandas UDF si el modelo es en scikit-learn
        La salida de la pandas-udf será un campo "struct" (diccionario con claves `proba_0` y `proba_1`)

        La salida final de este método es el Spark dataframe con las variables de entrada (features)
        + 2 columnas adicionales que serán las probabilidades de presencia o ausencia del evento
        """
        # TODO PSC: EL CANDIDATO TENDRÁ QUE DEFINIR UNA PANDAS-UDF PARA HACER LA PREDICCIÓN DEL MODELO DE SCIKIT-LEARN
        # TODO PSC: LE DAMOS LA ESTRUCTURA DE SALIDA
        model = self.load_model()
        if isinstance(model, LGBMClassifier):
            categorical_features = self.categorical_features.copy()
            proba_schema = StructType([
                StructField("proba_0", DoubleType()),
                StructField("proba_1", DoubleType())
            ])

            @pandas_udf(proba_schema)
            def predict_proba_udf(*cols: pd.Series) -> pd.DataFrame:

                features_df = pd.concat(cols, axis=1)
                features_df.columns = model.feature_names_in_  # para estar en el mismo orden uso el atributo de lgbm

                # como en el caso del entrenamiento del modelo lgbm
                # hacemos que las variables categóricas establecidas tengan el tipado adecuado
                for col in categorical_features:
                    if col in features_df.columns:
                        features_df[col] = features_df[col].astype("category")

                preds_proba = model.predict_proba(features_df)
                result_df = pd.DataFrame(preds_proba, columns=["proba_0", "proba_1"])
                return result_df
            predictions_df = df.withColumn(
                "probs",
                predict_proba_udf(*[df[col] for col in model.feature_names_in_])
            )
            logger.info("Aplicada pandas-udf para obtener las predicciones")
        else:
            predictions_df = model.transform(df)
        predictions_df = self.__output_predictions_format(predictions_df)
        return predictions_df

    def __output_predictions_format(self, df: DF) -> DF:
        """
        Método para disponer formato estandarizado en la salida del dataframe de predicciones
        ya sea usando un modelo de scikit-learn o de spark.
        # TODO: es necesario implementar el formateo de salida en un modelo de spark
        """
        if self.model_framework == "scikit-learn":
            df = df.select(
                "*",
                F.col("probs.proba_0").alias("Probs_0"),
                F.col("probs.proba_1").alias("Probs_1")
            ).drop("probs")
        else:
            df = df.select("*")
        logger.info(f"Formateada la salida del Spark dataframe de las predicciones {self.model_framework}")
        return df
