from typing import Optional, Union, Tuple, List, Dict

from pyspark.sql import DataFrame as DF
from pyspark.sql import functions as F

from src.utils.logger import get_logger

logger = get_logger()


def impute_nulls_for_numeric_cols(
        df: DF, method_name: str, col_name: str, stratific_col: Optional[str] = None
) -> Tuple[DF, Dict[str, int]]:
    """
    Método para imputar valores nulos en el dataset.
    Imputación simple: mediana, media, de forma estratificada
    Devuelve la tabla y un diccionario (clave: nombre de la variable y valor: valor por el que se imputa)
    # TODO: EL CANDIDATO DEBERÁ REFACTORIZAR LA ESTRUCTURA DEL CÓDIGO SI LO CREE CONVENIENTE
    """
    if stratific_col is None:
        if method_name == "mean":
            value = df.agg(F.avg(col_name).alias(f"{method_name}_{col_name}")).first()[f"{method_name}_{col_name}"]
        elif method_name == "median":
            value = df.approxQuantile(col_name, [0.5], 0.01)[0]
        else:
            msg = (
                f"No disponible el método {method_name} para la imputación de variables numéricas: {col_name}"
            )
            logger.error(msg)
            raise ValueError(msg)
        msg = (
            f"Calculo {method_name} para imputación de variables numéricas. "
            f"Los valores nulos de {col_name} se imputarán a {value}"
        )
        logger.info(msg)
        df = df.fillna({col_name: value})
        value_dict = {col_name: value}
    elif stratific_col in df.columns:
        if method_name == "mean":
            value = df.groupBy(stratific_col).agg(F.round(F.avg(col_name)).alias(f"{method_name}_{col_name}"))
        elif method_name == "median":
            expr = f"percentile_approx({col_name}, 0.5)"
            value = df.groupBy(stratific_col).agg(F.expr(expr).alias(f"{method_name}_{col_name}"))
        else:
            msg = (
                f"No disponible el método {method_name} para la imputación de variables numéricas: {col_name}"
            )
            logger.error(msg)
            raise ValueError(msg)
        msg = (
            f"Calculo {method_name} para imputación de variables numéricas. "
            f"Los valores nulos de {col_name} se imputarán como {value.show()}"
        )
        logger.info(msg)
        df = (
                 df.join(value, on=stratific_col, how="left")
                 .withColumn(
                     col_name,
                     F.coalesce(F.col(col_name), F.col(f"{method_name}_{col_name}"))
                 ).drop(f"{method_name}_{col_name}")
        )
        value_dict = {
            col_name: {row[stratific_col]: row[f"{method_name}_{col_name}"] for row in value.collect()}
        }
    return df, value_dict


def impute_nulls_for_numerical_cols_out_sample(
        df: DF,
        col_name: str,
        impute_value_or_mapping: [Union[float, int, Dict[str, Union[float, int]]]],
        stratific_col: Optional[str] = None
):
    """
    Imputación de valores para las predicciones futuras

    impute_value_or_mapping es un diccionario cuya clave es la variable a imputar
    Puede tener como valores el valor a imputar o un diccionario anidado
    si se emplea un tipo de imputación estratificada

    stratfic_col es la variable utilizada como soporte para hacer una imputación estratificada
    """
    if isinstance(impute_value_or_mapping, (float, int)):
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), impute_value_or_mapping).otherwise(F.col(col_name))
        )

    # asumimos que es un diccionario anidado con valores float o enteros
    elif isinstance(impute_value_or_mapping, dict):
        logger.info(f"imputando {col_name} usando {stratific_col}")
        # uso de un bucle para encadenar las condiciones (en cascada)
        expr = F.col(col_name)  # inicialización de la expresión con la condición inicial
        for cond_val, impute_val in impute_value_or_mapping.items():
            expr = F.when(
                (F.col(stratific_col) == cond_val) & F.col(col_name).isNull(), impute_val
            ).otherwise(expr)
        df = df.withColumn(col_name, expr)
    logger.info(f"{col_name} ha sido imputada según {impute_value_or_mapping}")
    return df


def impute_nulls_for_categorical_cols(df: DF, colname: str, stratific_col: Optional[str] = None) -> DF:
    """
    Imputación de datos faltantes para una columna específica.

    Si fuese necesario utilizar este método, usa métodos internos definir la función
    """
    # TODO: SI FUESE NECESARIO IMPLEMENTAR CÓDIGO FUENTE PARA IMPUTAR VARIABLES CATEGÓRICAS
    pass


def create_new_col(
        df: DF,
        colname: Union[str, List[str]],
        group_levels: Optional[List[Union[str, float]]] = None
) -> DF:
    """
    Utiliza este método si necesitas crear nuevas variables
    """
    # TODO: SI FUESE NECESARIO IMPLEMENTAR CÓDIGO FUENTE PARA CREAR NUEVA VARIABLE
    pass
