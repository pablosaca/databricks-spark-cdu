from typing import List, Tuple, Dict, Optional

import pandas as pd
import plotly.express as px

from pyspark.sql import DataFrame as DF
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.stat import Correlation, ChiSquareTest


def boxplot(df: DF, num_col: str, cat_col: Optional[str] = None) -> None:
    """
    Partiendo de un dataframe de spark se obtiene un gráfico de cajas
    de una variable numérica o estratificada por una variable categórica
    """
    cols_to_select = [num_col] + ([cat_col] if cat_col is not None else [])
    pdf = df.select(*cols_to_select).toPandas()
    fig = px.box(pdf, x=cat_col, y=num_col)
    fig.show()


def scatterplot(df: DF, col_1: str, col_2: str, cat_col: Optional[str] = None) -> None:
    """
    Partiendo de un dataframe de spark se obtiene un gráfico de dispersión entre dos variables numéricas

    Importante: si se usa cat_col difirente de None la variable tiene que ser categórica (no valen variables enteras)
    """

    cols_to_select = [col_1, col_2] + ([cat_col] if cat_col is not None else [])
    pdf = df.select(*cols_to_select).toPandas()

    fig = px.scatter(
        pdf,
        x=col_1,
        y=col_2,
        color=cat_col,
        symbol=cat_col,
        size_max=60
    )
    fig.show()


def correlation_matrix_plot(df: DF, num_cols: List[str]) -> None:
    """
    Partiendo de un dataframe de spark se obtiene la correlación entre las variables numéricas
    """

    df = df.select(*num_cols)

    assembler = VectorAssembler(inputCols=num_cols, outputCol="features")
    df_vector = assembler.transform(df)
    corr_matrix = Correlation.corr(df_vector, "features", method="pearson").head()

    corr_matrix_dense = corr_matrix[0]
    corr_array = corr_matrix_dense.toArray()
    corr_matrix_pdf = pd.DataFrame(corr_array, columns=num_cols, index=num_cols)
    fig = px.imshow(
        corr_matrix_pdf, color_continuous_scale='RdBu', color_continuous_midpoint=0, text_auto=".2f"
        )

    # Ajustar el tamaño del gráfico
    fig.update_layout(
        width=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        height=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        title_font_size=20
    )
    fig.show()


def chi_square_test(df: DF, col_1: str, col_2: str) -> Tuple[DF, Dict[str, float]]:
    """
    Partiendo de un dataframe de spark se obtiene la chi-square entre dos variables.
    Incluye también un count asociado col_1 es la columna a indexar
    """
    indexer = StringIndexer(inputCol=col_1, outputCol=f"{col_1}_indexed")
    df_indexed = indexer.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=[f"{col_1}_indexed"], outputCol="features")
    df_vector = assembler.transform(df_indexed)

    output = ChiSquareTest.test(df_vector, "features", col_2).head()
    chi_square_dict = {
        "value": output["statistics"][0],
        "p-value": output["pValues"][0]
        }
    df_agg_count = df.groupBy(col_1, col_2).count()
    return df_agg_count, chi_square_dict
