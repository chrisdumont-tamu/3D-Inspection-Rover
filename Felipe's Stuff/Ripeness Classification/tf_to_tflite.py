import tensorflow as tf
from tensorflow.keras import layers

########################## Ripeness

# # File name
# FILE_NAME_RIPENESS = 'ripe.tflite'

# # Path to models to convert
# RIPENESS_MODEL_PATH = r'saved_models\ripeness\ripe1'

# # Path to save tflite models
# RIPENESS_SAVE_PATH = r'saved_models\tensorflow_lite\ripeness'

# # Load model to convert
# ripeness_model = tf.keras.models.load_model(RIPENESS_MODEL_PATH)

# # Convert tf model to tflite model
# ripeness_converter = tf.lite.TFLiteConverter.from_keras_model(ripeness_model)

# # Optimize model for smaller model size
# # ripeness_converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Create tflite model
# tflite_model_ripeness = ripeness_converter.convert()

# # Save destination
# SAVE_DIRECTORY_RIPENESS = RIPENESS_SAVE_PATH + '\\' + FILE_NAME_RIPENESS

# # Save model
# open(SAVE_DIRECTORY_RIPENESS, 'wb').write(tflite_model_ripeness)



########################## Defectiveness

# File name
FILE_NAME_DEFECTIVENESS = 'defect_oversample_binary_optimized_80.tflite'

# Path to models to convert
DEFECTIVENESS_MODEL_PATH = r'saved_models\defectiveness\defect_oversample_binary_80'

# Path to save tflite models
DEFECTIVENESS_SAVE_PATH = r'saved_models\tensorflow_lite\defectiveness'

# Load model to convert
defectiveness_model = tf.keras.models.load_model(DEFECTIVENESS_MODEL_PATH)

# Convert tf model to tflite model
defectiveness_converter = tf.lite.TFLiteConverter.from_keras_model(defectiveness_model)

# Optimize model for smaller model size
# defectiveness_converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Create tflite model
tflite_model_defectiveness = defectiveness_converter.convert()

# Save destination
SAVE_DIRECTORY_DEFECTIVENESS = DEFECTIVENESS_SAVE_PATH + '\\' + FILE_NAME_DEFECTIVENESS

# Save model
open(SAVE_DIRECTORY_DEFECTIVENESS, 'wb').write(tflite_model_defectiveness)