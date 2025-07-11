import os
from typing import Optional, List

import joblib
import json
import pandas as pd
from lightgbm.sklearn import LGBMRegressor

from src.utils.logger import get_logger

logger = get_logger()


def get_model_df(
        df: pd.DataFrame, save_columns: Optional[List[str]] = None,
        path: Optional[str] = None,
        name: Optional[str] = None
) -> pd.DataFrame:
    """
    Crea tabla con features
    """
    aux_save_columns = [
        "fecha",
        "consumo_gas",
        "consumo_gas_lag_1",
        "consumo_gas_rolling_window_15",
        "consumo_gas_rolling_window_8_std",
        "consumo_gas_rolling_window_15_std",
        "temperatura_rolling_window_3",
        "temperatura_lag_14",
        "velocidad_viento_rolling_window_3",
        "velocidad_viento_rolling_window_8",
        "velocidad_viento_rolling_window_15",
        "es_festivo",
        "dia_semana",
        "festivo_previo",
        "festivo_siguiente",
        "semana_anyo"
    ]
    save_columns = aux_save_columns if save_columns is None else save_columns
    df = df[save_columns]
    df = df.set_index("fecha")
    logger.info("Obtención de la tabla final previo entrenamiento del modelo - incluye histórico y features")

    if path is not None and name is not None:
        df.to_csv(f"../{path}/{name}.csv")
        logger.info(f"Guardado fichero {name} en {path}")
    return df


def save_model(model: LGBMRegressor, path: str, model_name: str) -> None:
    """
    Guardar el modelo en el directorio correspondiente
    """
    path = f"../{path}"
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directorio {path} creado para guardar el artefacto del modelo")
    joblib.dump(model, os.path.join(path, f"{model_name}.pkl"))
    logger.info(f"Modelo guardado en {os.path.join(path, f'{model_name}.pkl')}")


def save_features(features: List[str], path: str, features_name: str) -> None:
    """
    Guardar diccionario con las columnas utilizadas
    """
    path = f"../{path}"
    features_dict = {"features": features}
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directorio {path} creado para guardar el artefacto del modelo")
    with open(f"{path}/{features_name}.json", "w", encoding="utf-8") as archivo:
        json.dump(features_dict, archivo, ensure_ascii=False, indent=4)
    logger.info(f"Features guardadas en {os.path.join(path, f'{features_name}.json')}")


def load_data(table_name: str) -> pd.DataFrame:
    """
    Carga fichero csv como pandas DataFrame
    """
    logger.info(f"Se carga el fichero {table_name}")
    df = pd.read_csv(f"../data/{table_name}.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")
    logger.info("Se pasa la variable 'fecha' como índice en el dataframe")
    return df
