# Databricks notebook source
# MAGIC %md
# MAGIC #UNIDOS E INTERNET PARTE 2

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, when, upper, trim, translate, count,
    regexp_replace, sum as Fsum, avg, max as Fmax, min as Fmin
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Carga de Datos

# COMMAND ----------

UNIDOS_PATH = "/Volumes/workspace/proyecto/proyectvolume/dataframes/dataframes/Beneficiarios-Estrategia-UNIDOS.csv"
df_unidos = spark.read.csv(UNIDOS_PATH, header=True, inferSchema=False, sep=",")

display(df_unidos.limit(10))

# COMMAND ----------

INTERNET_PATH = "/Volumes/workspace/proyecto/proyectvolume/dataframes/dataframes/Internet-Fijo-Accesos-por-Tecnología-y-Segmento.csv"
df_internet = spark.read.csv(INTERNET_PATH, header=True, inferSchema=False, sep=",")

print("Internet Fijo — filas:", df_internet.count(), "| columnas:", len(df_internet.columns))
display(df_internet.limit(10))

# COMMAND ----------

def estandarizar_municipio(df, nombre_columna):
    return (
        df
        .withColumn("municipio_std", upper(trim(col(nombre_columna))))
        .withColumn("municipio_std", translate(col("municipio_std"), "ÁÉÍÓÚÄËÏÖÜÑ", "AEIOUAEIOUN"))
    )#estandarizacion para el join

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtros

# COMMAND ----------

print("Filas antes:", df_unidos.count())
df_unidos = df_unidos.filter(col("EstadoBeneficiario") == "ACTIVO")
print("Tras filtrar ACTIVOS:", df_unidos.count())

# COMMAND ----------

df_unidos = df_unidos.filter(col("Estrato") != "99")
print("Tras eliminar Estrato 99:", df_unidos.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformaciones

# COMMAND ----------

cols_eliminar = [
    "Pais", "CondicionSexual", "Etnia",
    "BeneficiarioSISBEN", "NombreDepartamentoAtencion",
    "CodigoDepartamentoAtencion"
]
df_unidos = df_unidos.drop(*cols_eliminar)
print("Columnas tras eliminación:", len(df_unidos.columns))

# COMMAND ----------

df_unidos = df_unidos.withColumn("Parentesco", upper(col("Parentesco")))
df_unidos.groupBy("Parentesco").count().orderBy(col("count").desc()).show(5)

# COMMAND ----------

for c in ["Genero", "Parentesco", "Discapacidad", "EstadoCivil"]:
    df_unidos = df_unidos.withColumn(
        c, when(col(c) == "ND", None).otherwise(col(c))
    )

# COMMAND ----------

logros = [f"Logro{i}" for i in range(1, 27)]
df_unidos = df_unidos.withColumn(
    "total_logros_alcanzados",
    sum([when(col(l) == "ALCANZADO", 1).otherwise(0) for l in logros])
)

# COMMAND ----------

niveles_bajos = [
    "BÁSICA PRIMARIA 1°", "BÁSICA PRIMARIA 2°", "BÁSICA PRIMARIA 3°",
    "BÁSICA PRIMARIA 4°", "BÁSICA PRIMARIA 5°", "NINGUNO"
]
df_unidos = df_unidos.withColumn(
    "nivel_educativo_bajo",
    when(col("PE42").isin(niveles_bajos), 1).otherwise(0)
)

# COMMAND ----------

df_unidos = estandarizar_municipio(df_unidos, "NombreMunicipioAtencion")

display(df_unidos.select(
    "municipio_std", "TipoPoblacion", "Estrato",
    "total_logros_alcanzados", "nivel_educativo_bajo"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregación por Municipio

# COMMAND ----------

dfUNIDOSAGG = df_unidos.groupBy("municipio_std").agg(
    count("*").alias("hogares_unidos"),
    avg("total_logros_alcanzados").alias("avg_logros_alcanzados"),
    (Fsum("nivel_educativo_bajo") / count("*")).alias("pct_nivel_educativo_bajo"),
    (count(when(col("TipoPoblacion") == "UNIDOS RURAL", 1)) / count("*"))
        .alias("pct_rural_unidos"),
    avg(col("Estrato").cast("double")).alias("avg_estrato_unidos")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalización
# MAGIC

# COMMAND ----------

w = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

dfUNIDOSAGG = dfUNIDOSAGG.withColumn(
    "hogares_unidos_norm",
    (col("hogares_unidos") - Fmin("hogares_unidos").over(w)) /
    (Fmax("hogares_unidos").over(w) - Fmin("hogares_unidos").over(w))
)

print("dfUNIDOSAGG — municipios:", dfUNIDOSAGG.count())
display(dfUNIDOSAGG.orderBy(col("hogares_unidos").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC #Internet

# COMMAND ----------

df_internet = df_internet \
    .withColumn("VELOCIDAD_BAJADA",
        regexp_replace(col("VELOCIDAD_BAJADA"), ",", ".").cast("float")) \
    .withColumn("VELOCIDAD_SUBIDA",
        regexp_replace(col("VELOCIDAD_SUBIDA"), ",", ".").cast("float")) \
    .withColumn("NUM_ACCESOS", col("`No DE ACCESOS`").cast("integer")) #pasar a numeros

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtros

# COMMAND ----------

print(df_internet.columns)


# COMMAND ----------

print("Filas antes:", df_internet.count())
df_internet = df_internet.filter(col("NUM_ACCESOS") > 0)
print("Tras eliminar accesos == 0:", df_internet.count())

# COMMAND ----------

df_internet = df_internet.filter(
    col("SEGMENTO") != "USO PROPIO INTERNO DEL OPERADOR"
)
print("Tras eliminar uso interno operador:", df_internet.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformaciones

# COMMAND ----------

df_internet = df_internet.drop("COD_DEPARTAMENTO", "DEPARTAMENTO")
print("Columnas tras eliminación:", len(df_internet.columns))

# COMMAND ----------

df_internet = df_internet \
    .withColumn("VELOCIDAD_BAJADA",
        when(col("VELOCIDAD_BAJADA") == 0, None).otherwise(col("VELOCIDAD_BAJADA"))) \
    .withColumn("VELOCIDAD_SUBIDA",
        when(col("VELOCIDAD_SUBIDA") == 0, None).otherwise(col("VELOCIDAD_SUBIDA")))

# COMMAND ----------

df_internet = df_internet.withColumn(
    "TIPO_TECNOLOGIA",
    when(col("TECNOLOGIA").contains("FIBER") | col("TECNOLOGIA").contains("FTTX"), "FIBRA")
    .when(col("TECNOLOGIA").isin("CABLE", "HYBRID FIBER COAXIAL (HFC)"), "CABLE/HFC")
    .when(col("TECNOLOGIA") == "XDSL", "XDSL")
    .when(col("TECNOLOGIA") == "SATELITAL", "SATELITAL")
    .when(col("TECNOLOGIA").isin("WIFI", "WIMAX", "OTRAS TECNOLOGÍAS INALÁMBRICAS"), "INALÁMBRICO")
    .otherwise("OTRAS")
)

df_internet.groupBy("TIPO_TECNOLOGIA").count().orderBy(col("count").desc()).show()

# COMMAND ----------

df_internet = df_internet.withColumn(
    "ES_RESIDENCIAL",
    when(col("SEGMENTO").startswith("RESIDENCIAL"), 1).otherwise(0)
)

df_internet.groupBy("ES_RESIDENCIAL").count().show()

# COMMAND ----------

df_internet = estandarizar_municipio(df_internet, "MUNICIPIO")

display(df_internet.select(
    "municipio_std", "TIPO_TECNOLOGIA", "ES_RESIDENCIAL",
    "VELOCIDAD_BAJADA", "VELOCIDAD_SUBIDA", "NUM_ACCESOS"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregación por Municipio

# COMMAND ----------

w = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

df_internet_vel = df_internet.groupBy("municipio_std").agg(
    (Fsum(col("VELOCIDAD_BAJADA") * col("NUM_ACCESOS")) /
     Fsum("NUM_ACCESOS")).alias("avg_vel_bajada"),
    (Fsum(col("VELOCIDAD_SUBIDA") * col("NUM_ACCESOS")) /
     Fsum("NUM_ACCESOS")).alias("avg_vel_subida"),
    Fsum("NUM_ACCESOS").alias("total_accesos_fijo"),
    count("*").alias("registros_internet")
)

df_internet_vel = df_internet_vel \
    .withColumn("avg_velocidad_fijo",
        (col("avg_vel_bajada") + col("avg_vel_subida")) / 2) \
    .withColumn("velocidad_norm_fijo",
        (col("avg_velocidad_fijo") - Fmin("avg_velocidad_fijo").over(w)) /
        (Fmax("avg_velocidad_fijo").over(w) - Fmin("avg_velocidad_fijo").over(w)))

display(df_internet_vel.orderBy(col("total_accesos_fijo").desc()).limit(10))

# COMMAND ----------

df_internet_res = df_internet.groupBy("municipio_std").agg(
    (Fsum(when(col("ES_RESIDENCIAL") == 1, col("NUM_ACCESOS")).otherwise(0)) /
     Fsum("NUM_ACCESOS")).alias("pct_accesos_residencial")
)
#accesos residenciales ponderado por suscriptores
display(df_internet_res.limit(10))

# COMMAND ----------

df_tech = (
    df_internet
    .groupBy("municipio_std", "TIPO_TECNOLOGIA")
    .agg(Fsum("NUM_ACCESOS").alias("accesos_por_tech"))
    .withColumn("rank",
        F.row_number().over(
            Window.partitionBy("municipio_std")
                  .orderBy(col("accesos_por_tech").desc())
        )
    )
    .filter(col("rank") == 1)
    .select("municipio_std", col("TIPO_TECNOLOGIA").alias("tecnologia_predominante"))
)
#tecnología predomina por municipio
display(df_tech.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### JOINS

# COMMAND ----------

dfInternetFijoAGG = (
    df_internet_vel
    .join(df_internet_res, on="municipio_std", how="left")
    .join(df_tech,         on="municipio_std", how="left")
)
dfInternetFijoAGG = dfInternetFijoAGG.withColumn(
    "acceso_digital_residencial",
    F.coalesce(col("pct_accesos_residencial"), F.lit(0.0)) *
    F.coalesce(col("velocidad_norm_fijo"),      F.lit(0.0))
)

print("dfInternetFijoAGG — municipios:", dfInternetFijoAGG.count())
display(dfInternetFijoAGG.orderBy(col("total_accesos_fijo").desc()))

# COMMAND ----------

# df_clean = JOIN único de tus dos AGGs
# Cuando trabajen juntas: reemplazar por LEFT JOINs sobre el df_clean de Nata
df_clean = dfUNIDOSAGG.join(
    dfInternetFijoAGG,
    on="municipio_std",
    how="inner"
)

print("Municipios en df_clean:", df_clean.count())
print("Columnas en df_clean:", len(df_clean.columns))

# COMMAND ----------

print("Cobertura del df_clean:")
df_clean.select(
    F.count(F.when(F.col("hogares_unidos").isNotNull(),     1)).alias("con_UNIDOS"),
    F.count(F.when(F.col("total_accesos_fijo").isNotNull(), 1)).alias("con_InternetFijo"),
    F.count("*").alias("total_municipios")
).show()

# COMMAND ----------

display(df_clean.limit(15))