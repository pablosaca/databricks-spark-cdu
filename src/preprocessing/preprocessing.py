from typing import Optional, Union, Tuple

from pyspark.sql import DataFrame as DF
from pyspark.sql import functions as F

from src.utils.logger import get_logger

logger = get_logger()


class Preprocessing:
    """
    Clase utilizada para imputar variables numéricas o categóricas.
    Utiliza una imputación simple (por mediana, media, de forma estratificada,
    categoría más relevante o agrupación de categorías)
    """

    @staticmethod
    def impute_nulls_for_numeric_cols(
            df: DF, method_name: str, col_name: str, stratific_col: Optional[str] = None
    ) -> DF:
        """
        Imputación de datos faltantes para una columna específica.

        Si fuese necesario utilizar este método, usa métodos internos definir la función
        """
        pass

    @staticmethod
    def impute_stats_numeric_cols(
            df: DF, method_name: str, col_name: str, stratific_col: Optional[str]
    ) -> Tuple[DF, Union[int, DF]]:
        """

        """
        if stratific_col is None:
            if method_name == "mean":
                value = df.agg(F.avg(col_name).alias(col_name)).first()[f"{method_name}_{col_name}"]
            elif method_name == "median":
                value = df.approxQuantile("prima", [0.5], 0.01)[0]
            else:
                msg = (
                    f"No disponible el método {method_name} para la imputación de variables numéricas: {col_name}"
                )
                logger.info(msg)
                raise ValueError(msg)
            msg = (
                f"Calculo {method_name} para imputación de variables numéricas. "
                f"Los valores nulos de {col_name} se imputarán a {value}"
            )
            logger.info(msg)
        elif stratific_col in df.columns:
            if method_name == "mean":
                value = df.groupBy(stratific_col).agg(F.avg(col_name).alias(f"{method_name}_{col_name}"))
            elif method_name == "median":
                expr = f"percentile_approx({col_name}, 0.5)"
                value = df.groupBy("tipo_seguro").agg(F.expr(expr).alias(f"{method_name}_{col_name}"))
            else:
                msg = (
                    f"No disponible el método {method_name} para la imputación de variables numéricas: {col_name}"
                )
                logger.info(msg)
                raise ValueError(msg)
            msg = (
                f"Calculo {method_name} para imputación de variables numéricas. "
                f"Los valores nulos de {col_name} se imputarán como {value.show()}"
            )
            logger.info(msg)
            df = df.join(value, on=stratific_col, how="left")
        return df, value

    @staticmethod
    def impute_nulls_for_categorical_cols(df: DF, colname: str, stratific_col: Optional[str] = None) -> DF:
        """
        Imputación de datos faltantes para una columna específica.

        Si fuese necesario utilizar este método, usa métodos internos definir la función
        """
        pass

    @staticmethod
    def create_new_col(df: DF, colname: str) -> DF:
        """
        Utiliza este método si necesitas crear nuevas variables
        """
        pass
