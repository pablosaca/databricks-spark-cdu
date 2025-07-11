from typing import Tuple, Union
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame as DF

from src.utils.logger import get_logger


logger = get_logger()


class BasicData:
    """
    Clase que obtiene los tablones necesarios para realizar el entrenamiento y predicción del modelo
    """

    def __init__(
            self,
            spark_session: SparkSession,
            train_name: str,
            test_name: str,
            target_name: str
    ):

        self.spark_session = spark_session
        self.train_name = train_name
        self.test_name = test_name
        self.target_name = target_name

    def get_basic_data(self) -> Union[DF, Tuple[DF, DF]]:
        """
        Obtención de tablón con target del problema y features asociadas

        En este método se cargan los ficheros y se genera el tablón o tablones base para continuar
        """

        logger.info(
            "Obtención del macrotablón base. Incluye features y target"
        )
        df = self.__load_data(self.train_name)
        return df

    def __load_data(self, table_name: str) -> DF:
        """
        Carga tabla de la database de databricks como Dataframe de spark
        """
        logger.info(f"Se carga el fichero {table_name}")
        return self.spark_session.read.table(table_name)

    @staticmethod
    def __merge_tables(df1: DF, df2: DF, cols_merge: list) -> DF:
        """
        Cruce de tablas para construir el macrotablón
        """
        logger.info(f"Se cruzan dos tablas (left-join) por {cols_merge}")
        pass
