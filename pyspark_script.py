"""
# Title : OC_P8 PySpark Script
# Description : PySpark script for OpenClassrooms project 8
# Author : ahonestla
# Date : 8-Mars-2023
# Version : 1.0
# Usage : spark-submit
"""

# Import modules
import io
import sys
import logging
import numpy as np
import pandas as pd
from typing import Iterator
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.ml.feature import PCA as pyPCA
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

# Logging configuration
formatter = logging.Formatter("[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Define paths
PATH_PROJ = "gs://bucket-openclassrooms-p8"
# PATH_PROJ = "/Users/victor/Documents/OPENCLASSROOMS/projet_8"
PATH_DATA = PATH_PROJ + "/data/training"
PATH_RESULTS = PATH_PROJ + "/data/results"
PATH_LOGS = PATH_PROJ + "/logs"

PCA_K = 100


def model_create(show_summary=False):
    """Create a MobileNetV2 model with top layer removed

    Returns:
        MobileNetV2 model
    """
    # Load default model
    model_base = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    # Freeze layers
    for layer in model_base.layers:
        layer.trainable = False
    # Create model without top layer
    model_new = Model(inputs=model_base.input, outputs=model_base.layers[-2].output)
    # Show summary
    if show_summary is True:
        print(model_new.summary())
    return model_new


def preprocess(content):
    """Preprocesses raw image bytes.

    Args:
        content: PIL Image

    Returns:
        Numpy array
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """Featurize a pd.Series of raw images using the input model.

    Args:
        model: CNN model
        content_series: pd.Series of image data

    Returns:
        pd.Series of image features
    """
    content_input = np.stack(content_series.map(preprocess))
    preds = model.predict(content_input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


def main():
    """Main function of spark script"""

    # Start spark code
    logger.info("Starting spark application...")
    # spark = SparkSession.builder.appName("oc_p8").master("yarn").getOrCreate()
    spark = (
        SparkSession.builder.config("spark.eventLog.dir", PATH_LOGS)
        .config("spark.history.fs.logDirectory", PATH_LOGS)
        .getOrCreate()
    )
    context = spark.sparkContext
    context.setLogLevel("INFO")

    # Display spark config
    conf = context.getConf()
    config_items = conf.getAll()
    logger.info("Spark application started with following configuration :")
    print("spark.executor.cores:", conf.get("spark.executor.cores"))
    print("spark.executor.instances:", conf.get("spark.executor.instances"))
    [print(item) for item in config_items]

    # Create broadcast weights
    logger.info("Broadcasting the model weights...")
    broadcast_weights = context.broadcast(model_create(show_summary=True).get_weights())

    @F.pandas_udf("array<float>")
    def featurize_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        """This method is a Scalar Iterator pandas UDF wrapping our featurization function.
            The decorator specifies this returns a Spark DataFrame column of type ArrayType(FloatType).

        Args:
            content_series_iter: Iterator over batches of data, where each batch
                                is a pandas Series of image data.

        Yields:
            pd.Series of image features
        """
        # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
        # for multiple data batches.  This amortizes the overhead of loading big models.
        model = model_create()
        # Broadcast weights to workers
        model.set_weights(broadcast_weights.value)
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)

    # Load all images
    logger.info("Loading the images...")
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(PATH_DATA)
    )
    images = images.withColumn("label", F.element_at(F.split(images["path"], "/"), -2))
    images.select("path", "label").show(5, False)
    images.printSchema()
    logger.info("Successfully loaded %i images!", images.count())

    # Create the image features
    logger.info("Featurizing the images...")
    features_df = images.repartition(20).select(
        F.col("path"), F.col("label"), featurize_udf("content").alias("features")
    )
    features_df = features_df.withColumn("features_vec", array_to_vector("features"))
    features_df.show(5)
    features_df.printSchema()
    logger.info("Successfully created features and vectors!")

    # Apply PCA on features
    logger.info("Applying PCA with %i components...", PCA_K)
    pca = pyPCA(k=PCA_K, inputCol="features_vec", outputCol="features_pca")
    pca_model = pca.fit(features_df)
    features_df = pca_model.transform(features_df)
    features_df.show(5)
    features_df.printSchema()
    logger.info("Successfully applying PCA on features!")

    # Save PCA output as single json file
    pca_output = features_df.select(F.col("features_pca")).withColumn("features_pca", vector_to_array("features_pca"))
    pca_output.repartition(1).write.mode("overwrite").json(PATH_RESULTS + "/PCA_output")

    # Save results as parquet files
    features_df.write.mode("overwrite").parquet(PATH_RESULTS + "/Features_output")

    # End spark code
    logger.info("Ending spark application")
    spark.stop()
    return None


# Starting point for PySpark
if __name__ == "__main__":
    main()
    sys.exit()
