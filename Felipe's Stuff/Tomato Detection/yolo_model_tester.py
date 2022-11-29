# The followign code was used to do some quick tests of the yolov4-tiny models
# after they were converted into tensorflow and tensorflow lite models

import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

import tom_localization_toolset as tlt

def setup_camera():
    ## Camera setup
    # Create a pipeline object to own and handle realsnese camera
    pipeline = rs.pipeline()

    # Configure the streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Begin streaming
    pipeline.start(config)
    
    # Get frames
    frames = pipeline.wait_for_frames()
    
    return frames


def testPrediction_TF(model, image):
        predictions = model.predict(image)
        return predictions

# Constants
# IMAGE_PATH = r'C:\Users\felvi\ecen403programs\volume_estimation\saved_frames\test_2.jpg'
YOLO_MODEL_PATH = r'.\ml_models_testing\yolov4-tiny-416.tflite'
YOLO_INPUT_SIZE = 416
IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.25

# LOAD_IMAGE = False

# Variables
for i in range(10):
    frames = setup_camera()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
# print(f'shape: {np.shape(color_image)}')
# print(f'dtype: {color_image.dtype}')

# Get data depending on LOAD_IMAGE
# if LOAD_IMAGE:
    # image, Tomatoes_Collection = tlt.process_extract_tomatoes(YOLO_INPUT_SIZE, YOLO_MODEL_PATH, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, load_image = LOAD_IMAGE, image_path=IMAGE_PATH)
# else:
    # Tomatoes_Collection = tlt.process_extract_tomatoes(YOLO_INPUT_SIZE, YOLO_MODEL_PATH, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, img=color_image, load_image=LOAD_IMAGE)

# Testing code
# print(f'Num of tomatoes: {Tomatoes_Collection.get_number_of_tomatoes()}')
# if Tomatoes_Collection.get_number_of_tomatoes() != 0:
    # regions = Tomatoes_Collection.get_tomato_regions(image, np.shape(image)[0], np.shape(image)[1])
    # tlt.show_regions(regions)
# else:
    # print("\n\nNo tomatoes\n\n")

# image = cv2.imread(IMAGE_PATH)
# print(f'image shape: {np.shape(image)}')
# print(f'image dtype: {image.dtype}')

# YOLO_MODEL_PATH = r'C:\Users\felvi\ecen403programs\volume_estimation\ml_models_testing\yolov4-tiny-416'
# yolo_lite_model = tf.keras.models.load_model(YOLO_MODEL_PATH)

# pred = testPrediction_TF(yolo_lite_model, color_image)
# image_data = cv2.resize(color_image, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))    # Resize image for model application
# image_batch = np.expand_dims(image_data, 0)
# pred = yolo_lite_model.predict(image_batch)
tlt.show_image(color_image, 'image')
img_data = tlt.process_img(color_image, YOLO_INPUT_SIZE)
predictions = yolo_lite_model.predict(img_data)
boxes, scores, classes, valid_detections = tlt.get_data_arrays(predictions, CONFIDENCE_THRESHOLD, YOLO_INPUT_SIZE, IOU_THRESHOLD)

print(f'\n\n{boxes}\n\n')
print(f'\n\n{scores}\n\n')
print(f'\n\n{classes}\n\n')
print(f'\n\n{valid_detections}\n\n')