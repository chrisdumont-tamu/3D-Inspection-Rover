import tensorflow as tf
import cv2
import numpy as np
import time

def pprint(x, name):
    print(f'\n{name}:\t{x}')

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    # Returns 2 arrays, one containing the bbox coordinate values in an array
    # the other containing the confidance values of the predictions
    
    scores_max = tf.math.reduce_max(scores, axis=-1)    # Reduce axis
    
    mask = scores_max >= score_threshold    # Get mask to apply so only valid elements are kept
    class_boxes = tf.boolean_mask(box_xywh, mask)   # Apply mask to xywh information array
    pred_conf = tf.boolean_mask(scores, mask)   # Apply mask to scores (confidences) information array
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]) # Reshape class_boxes so that it coforms to original scores shape
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])   # Reshape pred_conf so that that if conforms to original scores shape

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1) # Separate xy and wh information array

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape   # Calculate min coordinate values (both x and y)
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape  # Calculate max coordinate values (both x and y)
    
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    
    # Output 'boxes' is a tensor of shape (1, m, 4), where m is the number of detections found
    # The axis with 4 elements is a tensor containing the normalized bbox coordinates in order:
    # [y_min x_min y_max x_max]
   
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)
    

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

## Apply tflite model to image data
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)    # Load model into interpreter
interpreter.allocate_tensors()  # Allocate tensors

input_details = interpreter.get_input_details() # Get model input details
output_details = interpreter.get_output_details()   # Get model output details
# print(f'\ninput details:\t{input_details}\n')
# print(f'\noutput details:\t{output_details}\n')

interpreter.set_tensor(input_details[0]['index'], images_data)  # Set the interpreter tensor

interpreter.invoke()    # Invoke 

pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] # Predictions
# print(f'\npred: {pred}\n')

boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([INPUT_SIZE, INPUT_SIZE]))

pprint(boxes, 'boxes_pre')
pprint(tf.shape(boxes), 'boxes shape')
# pprint(pred_conf, 'pred_conf')

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=20,
    max_total_size=20,
    iou_threshold=IOU_THRESHOLD,
    score_threshold=SCORE_THRESHOLD
)

pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

print(f'\nBoxes: {boxes}\n')
# print(f'\nScores: {scores}\n')
# print(f'\nClasses: {classes}\n')
# print(f'\nValid Detections: {valid_detections}\n')
# print(f'\nPredicted Bbox: {pred_bbox}\n')