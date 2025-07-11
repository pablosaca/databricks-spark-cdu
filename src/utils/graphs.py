from typing import List

import pandas as pd
import plotly.express as px

from pyspark.sql import DataFrame as DF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


def plot_correlation_matrix(df: DF, num_cols: List[str], title: str = "Matriz de Correlación") -> None:
    """
    Partiendo de un dataframe de spark se obtiene la correlación entre las variables numéricas
    """

    df = df.select(*num_cols)

    # vector de características
    assembler = VectorAssembler(inputCols=num_cols, outputCol="features")
    df_vector = assembler.transform(df)
    corr_matrix = Correlation.corr(df_vector, "features", method="pearson").head()[0]

    corr_matrix_pdf = pd.DataFrame(corr_matrix, columns=num_cols, index=num_cols)
    fig = px.imshow(
        corr_matrix_pdf, color_continuous_scale='RdBu', color_continuous_midpoint=0, title=title
        )

    # Ajustar el tamaño del gráfico
    fig.update_layout(
        width=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        height=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        title_font_size=20
    )
    fig.show()
