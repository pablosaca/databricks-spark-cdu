# Documentación

Este proyecto trata sobre un CdU de venta cruzada de productos usando pyspark.

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
│   └──data                                         # código fuente para la carga de datos y obtención de muestras
│         └── load_data.py      
│
│   └──preprocessing                                # código fuente para la limpieza de las muestras
│         └── preprocessing.py                           
│
│   └── utils                                       # código fuente para la carga de datos, visualización y otros 
│         └── graphs.py                           
│         └── load_data.py
│         └── logger.py                           
│
│
├── input_data                              # se encuentran disponibles ficheros csv como input del proyecto 
│  
├── requirements.txt                        # requirements list

```

## Instalación

Creación de un entorno virtual del proyecto

```
conda create -n cdu python=3.10
```

Para activar el entorno virtual usa la siguiente instrucción

```
conda activate cdu
```

Así, instala las dependencias del fichero `requirements.txt` usando `pip`

```
pip install -r requirements.txt
```

