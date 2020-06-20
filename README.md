# Predicción de la demanda frente a pandemias globales - Análisis de series temporales


## Sobre este proyecto

Company Challenge.

Título: Predicción de la demanda frente a pandemias globales.
 *fbprophet, Anaconda, Jupyter Notebooks, Python, Spark y Git.*

Descripción y objetivo:  Se realiza una implementación del algoritmo de Facebook Prophet en pyspark para el caso de uso de
predicción de demanda de productos por tienda, a partir de series temporales. El demostrador se despliega en diferentes plataformas:
local, docker, cloudera, aws o google collab. 
 
Tiempo total: 1 hora de demostración teorico-práctica.

## Estructura del repositorio

```{bash}
.
├── doc                   # Documentación de apoyo
├── exploratory           # Notebooks exploratorios
├── src                   # Código fuente para productivización
├── .gitignore            # Listado de ficheros no trackeados para el repositorio
├── docker-entrypoint.sh  # 
├── Dockerfile            # 
├── environment.yml       # 
├── README.md             # Introducción del repo, contribuidores  
├── requirements.txt      # Dependencias necesarias para instalar
```
## Datos de Entrada - Salida

**/src/main/resources/kaggle/train.csv** 

Contiene los ficheros con los datos diarios del registro de ventas por tienda y tipo de producto.
De Enero de 2013 a Diciembre de 2017 (train.csv).

**/src/main/resources/output** 

Contiene los resultados del procesamiento del forecasting, gráficas de distribución y temporalidad en diferentes 
niveles de agregación y predicciones a n días (csvs).

## Instrucciones Despliegue y Ejecución - Docker 

1. Construir la imagen del Docker:

```bash
docker build -t fp_ts/single:1 .
```

Donde single indica el nombre de la aplicación y 1 el número de versión del Docker.
Es importante el punto al final, ya que es el path al directorio que contiene el código (```.src/main/app```) 
y el fichero ```docker-entrypoint.sh```.

2. Crear el contenedor Docker a partir de la imagen creada en el punto anterior:

```bash
docker run \ 
    -v <directorio_host_data>: ~/fp_ts/src/main/resources/
    --name fp_ts_fbprophet
    fp_ts/single:1
```

donde ```fp_ts_fbprophet``` indica el nombre del contenedor Docker y se realiza mapeo de directorios entre el host 
y el contenedor. Estos directorios son:

- ```<directorio_host_data>: ``` Ubicación de los ficheros de entrada y de los png y ficheros csv de salida del forecasting


3. Parar el contenedor con el nombre del contenedor:

```bash
docker fp_ts_fbprophet
```

**Recursos**
- [Blog Databricks: Time Series Forecasting](https://bit.ly/blog-prophet)
- [Databricks: Notebook de partida](https://bit.ly/fbprophet)
- [Pystan library](https://pystan.readthedocs.io/en/latest/installation_beginner.html)
- [Anaconda fbprophet](https://anaconda.org/conda-forge/fbprophet)


**Contribuidores**
- [Estela Balboa](estela.balboa@gmail.com)

**Próximos Pasos**
- Mejorar el modelo.
- Migrar pandas_udf a pyspark.
- Implementar retorno de logs de ejecución.
- Desplegar en servicio web heroku.

```bash
spark-submit --name single --conf spark.default.parallelism=3 --conf spark.sql.shuffle.partitions=3 \
   --num-executors 2 --executor-memory 3G --executor-cores 2 --driver-memory 6G src/main/app/multi_item_store_forecasting.py
```