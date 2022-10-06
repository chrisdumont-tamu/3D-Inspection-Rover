import tensorflow as tf
from tensorflow.keras import layers

# File name
FILE_NAME_RIPENESS = 'ripe1.tflite'
FILE_NAME_DEFECTIVENESS = 'defective1.tflite'

# Path to models to convert
RIPENESS_MODEL_PATH = r'saved_models\ripeness\ripe1'
DEFECTIVENESS_MODEL_PATH = r'saved_models\defectiveness\defect1'

# Path to save tflite models
RIPENESS_SAVE_PATH = r'saved_models\tensorflow_lite\ripeness'
DEFECTIVENESS_SAVE_PATH = r'saved_models\tensorflow_lite\defectiveness'

# Load model to convert
model = tf.keras.models.load_model(RIPENESS_MODEL_PATH)

# Convert tf model to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize model for smaller model size
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Create tflite model
tflite_model = converter.convert()

# Save destination
SAVE_DIRECTORY = RIPENESS_SAVE_PATH + '\\' + FILE_NAME_RIPENESS

# Save model
open(SAVE_DIRECTORY, 'wb').write(tflite_model)