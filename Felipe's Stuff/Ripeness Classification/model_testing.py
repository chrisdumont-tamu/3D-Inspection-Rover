import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

## Functions
def testPrediction(model, ds):
    for images_batch, labels_batch in ds.take(1):
    
        image = images_batch[0]
    
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = model.predict(img_array)[0] 
        max_confidence_index = np.argmax(predictions)
        confidence = round(100 * (predictions[max_confidence_index]), 2)
        
        print(f'Predictions: {predictions}')
        print(f'Confidence: {confidence}')

# def testPredictions(model, ds):
    # for images_batch, labels_batch in ds.take(1):
    
        # image = images_batch[0]
    
        # img_array = tf.keras.preprocessing.image.img_to_array(image)
        # img_array = tf.expand_dims(img_array, 0) # create a batch

        # predictions = model.predict(img_array)[0] 
        # max_confidence_index = np.argmax(predictions)
        # confidence = round(100 * (predictions[max_confidence_index]), 2)
        
        # print(f'Predictions: {predictions}')
        # print(f'Confidence: {confidence}')

def showPrediction(model, test_ds):
    # Demo a prediction on a single image
    for images_batch, labels_batch in test_ds.take(1):
    
        # get an image form the dataset and its correspoding label
        first_image = images_batch[0].numpy().astype('uint8')
        first_label = labels_batch[0].numpy()
        
        actual_label = class_names[first_label]
        
        print("Prediction on a single image:")
        print("actual label:", actual_label)

        batch_prediction = model.predict(images_batch)
        prediction_label = class_names[np.argmax(batch_prediction[0])]
        print("predicted label:", prediction_label)
        plt.imshow(first_image)
        plt.title(f"actual class:{actual_label}, prediction class:{prediction_label}")
        plt.show()

# Useful Constants
IMAGE_EDGE_SIZE_X = 256 # Size of the image width 
IMAGE_EDGE_SIZE_Y = 256 # Size of the image height 
IMAGE_SHAPE = (IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y) # image a 256 by 256 matrix; (256, 256) shape
BATCH_SIZE = 10 # size of batch; how many images are to be loaded into the dataset at a time

# Directory to saved models
TESTING_DS_RIPENESS = r'testing_images\ripeness_test_ds'
MODEL_DIR_RIPENESS = r'saved_models\ripeness\ripe2_google'

TESTING_DS_DEFECT = r'testing_images\defective_test_ds'
MODEL_DIR_DEFECT = r'saved_models\defectiveness\defect1'

# Load testing dataset
ripeness_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TESTING_DS_RIPENESS,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# defective_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # TESTING_DS_DEFECT,
    # shuffle=True,
    # image_size = IMAGE_SHAPE,
    # batch_size = BATCH_SIZE
# )

# Get class names from dataset
# ripeness_names = ripeness_ds.class_names
# defective_names = defective_ds.class_names

# Load saved modelt into model var
model_ripeness = tf.keras.models.load_model(MODEL_DIR_RIPENESS)
# model_defective = tf.keras.models.load_model(MODEL_DIR_DEFECT)

# Debug Confidence > 100%
# testPrediction(model_ripeness, ripeness_ds)

scores = model_ripeness.evaluate(ripeness_ds)
print(f'Scores: {scores}')