from typing import Union, Optional
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame as DF

from src.utils.logger import get_logger


logger = get_logger()


class LoadData:
    """
    Clase que obtiene los tablones necesarios para realizar el entrenamiento y predicción del modelo
    """

    def __init__(
            self,
            spark_session: SparkSession,
            profile_table_name: str,
            performance_table_name: str,
            socioeconomic_table_name: str,
            target_table_name: Optional[str] = None
    ):

        self.spark_session = spark_session
        self.profile_table_name = profile_table_name
        self.performance_table_name = performance_table_name
        self.socioeconomic_table_name = socioeconomic_table_name
        self.target_table_name = target_table_name

    def get_basic_data(self) -> DF:
        """
        Obtención de tablón de features (y target asociado si aplica)
        En este método se cargan los ficheros y se genera el tablón para continuar el proceso
        Debe ser un método con el que se disponga de un macrotablón para realizar las diferentes operativas:
        - EDA y modelización
        - Predicción del modelo ante una simulación productiva del modelo
        """
        # TODO PSC: este método será borrado y tendrá que ser definido por el candidato
        # TODO PSC: el método no hace más que
        profile_table_name_df = self.__load_data(self.profile_table_name)
        performance_table_name_df = self.__load_data(self.performance_table_name)
        socioeconomic_table_name_df = self.__load_data(self.socioeconomic_table_name)

        df = self.__merge_tables(
            profile_table_name_df, performance_table_name_df, "id"
        )
        df = self.__merge_tables(
            df, socioeconomic_table_name_df, "Region_Code"
        )
        logger.info("Obtención del tablón de features")
        if self.target_table_name is not None:
            target_df = self.__load_data(self.target_table_name)
            df = self.__merge_tables(df, target_df, "id")
            logger.info("Obtención de tablón con target")
        return df

    def __load_data(self, table_name: str) -> DF:
        """
        Carga tabla de la database de databricks como Dataframe de spark
        """
        logger.info(f"Se carga el fichero {table_name}")
        return self.spark_session.read.table(table_name)

    @staticmethod
    def __merge_tables(df1: DF, df2: DF, cols_merge: Union[str, list], how: str = "left") -> DF:
        """
        Cruce de tablas para construir el macrotablón
        """
        # TODO PSC: este método será borrado y tendrá que ser definido por el candidato
        logger.info(f"Se cruzan dos tablas ({how}-join) por {cols_merge}")
        df1 = df1.join(df2, on=cols_merge, how=how)
        return df1
