import tensorflow as tf
import cv2
import numpy as np
import time

# Constants
IMAGE_PATH = r'./data/test2.jpg'
INPUT_SIZE = 416
MODEL_PATH = r'./checkpoints/yolov4-tiny-416.tflite'
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25

# Import the image and preprocess it
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_data = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))    # Resize image
image_data = image_data / 255.  # Normalize image

# Prepaer image data for model application
images_data = []
images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)    # Convert images_data to numpy array                                                          

start_time = time.time()

## Apply tflite model to image data
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)    # Load model into interpreter
interpreter.allocate_tensors()  # Allocate tensors

input_details = interpreter.get_input_details() # Get model input tensors
output_details = interpreter.get_output_details()   # Get model output tensors
print(f'\ninput details:\t{input_details}\n')
print(f'\noutput details:\t{output_details}\n')
print(f'\noutput details len:\t{len(output_details)}\n')

interpreter.set_tensor(input_details[0]['index'], images_data)  # Set the interpreter tensor

interpreter.invoke()    # Invoke 

pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] # Predictions
# boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([INPUT_SIZE, INPUT_SIZE]))

end_time = start_time = time.time()

print(f'\npred:\t{pred}\n')
print(f'\pred len:\t{len(pred)}\n')
print(f'\npred[0] len:\t{len(pred[0])}\n')