import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def getPrediction_TF(model, ds, batch_size):
    
    predictions = model.predict(x=ds)#, batch_size=batch_size)
    print(predictions)
    pred_labels = np.argmax(predictions, axis=-1)
    
    actual_labels = np.concatenate([labels for images, labels in ds], axis=0)

    return pred_labels, actual_labels


# Constants
IMAGE_EDGE_SIZE_X = 256 # Size of the image width 
IMAGE_EDGE_SIZE_Y = 256 # Size of the image height 
IMAGE_SHAPE = (IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y) # image a 256 by 256 matrix; (256, 256) shape
BATCH_SIZE = 24
TEST_DS_DIR = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\defective\balanced_test_ds_oversample_binary'
MODEL_DIR_DEFECT = r"C:\Users\felvi\ecen403programs\ripeness_classification\saved_models\defectiveness\defect_oversample_binary_80"

# tf29
# MODEL_DIR_DEFECT = r"C:\Users\felvi\ecen403programs\ripeness_classification\saved_models\tf_29\defectiveness29\defect_oversample_binary_80_tf29"

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DS_DIR,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# Get class names from data
class_names = test_ds.class_names

# Load model
model_defective = tf.keras.models.load_model(MODEL_DIR_DEFECT)

# Obtain predictions from test data
pred_labels, actual_labels = getPrediction_TF(model_defective, test_ds, BATCH_SIZE)

print(pred_labels, '\n\n')
print(actual_labels)
# Setup confusion matrix with data
ConfusionMatrixDisplay.from_predictions(y_true=actual_labels, y_pred=pred_labels)

# # Display confusion matrix
plt.show()