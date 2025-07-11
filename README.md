# Documentación

Este proyecto trata sobre xsell de productos en spark.

```

│
├── README.md                                       # Lee el fichero                             
│                              
├── src  
│   └── model                                       # código fuente para el entrenamiento
│         └── model.py                             
│
│    └── predict                                    # código fuente para la predicción
│         └── predict.py
│
│     └──preprocessing                              # código fuente para la limpieza de las series y la generación de features
│         └── clean_data.py                           
│
│   └── utils                                       # código fuente para la carga de datos, visualización y otros 
│         └── graphs.py                           
│         └── load_data.py
│         └── logger.py                           
│
│
├── data                                    # se encuentran disponibles ficheros csv como input del proyecto 
│  
├── requirements.txt                        # requirements list

```

## Instalación

Creación de un entorno virtual del proyecto

```
conda create -n forecasting python=3.10
```

Para activar el entorno virtual usa la siguiente instrucción

```
conda activate forecasting
```

Así, instala las dependencias del fichero `requirements.txt` usando `pip`

```
pip install -r requirements.txt
```

