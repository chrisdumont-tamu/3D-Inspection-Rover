import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ## Functions
def testPrediction_TF(model, ds):
    for images_batch, labels_batch in ds.take(1):
    
        image = images_batch[0]
    
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = model.predict(img_array)[0] 
        max_confidence_index = np.argmax(predictions)
        confidence = round(100 * (predictions[max_confidence_index]), 2)
        
        print(f'Predictions: {predictions}')
        print(f'Confidence: {confidence}')

def showPrediction(model, test_ds, class_names):
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



# # Useful Constants
IMAGE_EDGE_SIZE_X = 256 # Size of the image width 
IMAGE_EDGE_SIZE_Y = 256 # Size of the image height 
IMAGE_SHAPE = (IMAGE_EDGE_SIZE_X, IMAGE_EDGE_SIZE_Y) # image a 2000 by 2000 matrix; (2000, 2000) shape
BATCH_SIZE = 10 # size of batch; how many images are to be loaded into the dataset at a time

# # Directory to datasets
TESTING_DS_RIPENESS = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\ripeness\test_ds_oversample'
# TESTING_DS_DEFECT = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\defective\balanced_test_ds_oversample'
TESTING_DS_DEFECT = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\defective\balanced_test_ds_oversample_binary'
# TESTING_DS_DEFECT = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\defective\balanced_test_ds_small'

# ## Load testing dataset
ripeness_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TESTING_DS_RIPENESS,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

defective_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TESTING_DS_DEFECT,
    shuffle=True,
    image_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE
)

# # Get class names from dataset
ripeness_names = ripeness_ds.class_names
defective_names = defective_ds.class_names
print(defective_names)

# ############################################## Testing tf model

# Directory for TF saved model
# MODEL_DIR_RIPENESS = r'./saved_models/ripeness/ripe1'
MODEL_DIR_DEFECT = r'./saved_models/defectiveness/defect_oversample_binary_80'

# Load saved modelt into model var
# model_ripeness = tf.keras.models.load_model(MODEL_DIR_RIPENESS)
model_defective = tf.keras.models.load_model(MODEL_DIR_DEFECT)

# Evaulate TF model and print scores
# scores_ripeness =  model_ripeness.evaluate(ripeness_ds)
scores_defective = model_defective.evaluate(defective_ds)
# print(f'Ripeness: {scores_ripeness}')
print(f'Defectiveness: {scores_defective}')

showPrediction(model_defective, defective_ds, defective_names)
# testPrediction_TF(model_defective, defective_ds)
# ############################################## Testing tf lite model

# def testPrediction_TFLITE(model_path, image_path, classes, img_shape=(256, 256)):
    # # This function returns a prediction of a tf lite model given a model path and image path
    # # The prediction is a string with the most probable classification as determined by the model

    # # Load tflite model into interpreter and allocate tensors
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    
    # # Get input and output tensors
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # # print(f'\ninput details:\t{input_details}\n')
    # # print(f'output details:\t{output_details}\n')
    
    # # Load image to pass into model & preprocess it 
    # img = cv2.imread(image_path)    
    # img = np.float32(img)   # Change dtype of image from uint8 to float32 as model expects
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change image from BRG to RGB format
    # # print(f'img shape:\t{np.shape(img)}\n')
    # img_resized = cv2.resize(img, img_shape, interpolation=cv2.INTER_LINEAR)   # Resize image to expected input size (256x256)
    # # print(f'resized img shape:\t{np.shape(img_resized)}\n')
    # input_img = np.expand_dims(img_resized, 0)  # Model expects batch, create a single image batch by expanding dims from (256, 256, 3) to (1, 256, 256, 3)
    # # print(f'input img shape:\t{np.shape(input_img)}\n')
    
    # # print(f'input_img dtype: {input_img.dtype}')
    
    # # return None
    
    # # Apply the model
    # interpreter.set_tensor(input_details[0]['index'], input_img)
    
    # interpreter.invoke()
    
    # # Get output data of tf lite model. It is a list of decimals representing the confidence for each 
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    # # Use the output of the model to get the class of the highest confidence
    # max_confidence_index = np.argmax(output_data)
    # classification = classes[max_confidence_index]
    
    # return classification

# Constants

# Directory for saved models
# MODEL_DIR_RIPENESS_TFLITE = r'./saved_models/tensorflow_lite/ripeness/ripe1.tflite'
# MODEL_DIR_DEFECT_TFLITE = r'./saved_models/tensorflow_lite/defectiveness/defect1.tflite'

# IMAGE_PATH = r'C:\Users\felvi\ecen403programs\ripeness_classification\datasets\in_use_datasets\ripeness\old\test_ds\1_green\green59_3.jpg'

# ripeness_classes = ['green', 'breaker', 'turning', 'pink', 'light red', 'red']
# defectiveness_classes = ['good', 'blight', 'moldy', 'old']

# classification = testPrediction_TFLITE(MODEL_DIR_RIPENESS_TFLITE, IMAGE_PATH, ripeness_classes)
# print(classification)
# # testPrediction_TFLITE(MODEL_PATH, IMAGE_PATH)