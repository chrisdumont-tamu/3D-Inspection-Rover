
### This script is used to extract the yolov4-tiny data from an input image

# Import necessary libraries
import tensorflow as tf
import cv2
import numpy as np
import math
import tom_locator_toolset_test as tlt
# import tom_locator_toolset_original as tlt

# # Make a classes to hold tomato data
# class Tomato:

    # classification = 0

    # def __init__(self, tomato_id, coordinates, confidence):
        # self.tomato_id = tomato_id
        # self.coordinates = coordinates # Coords are normalized and in format [y_min x_min y_max x_max]
        # self.confidence = confidence
        # self.y_min = coordinates[0]
        # self.x_min = coordinates[1]
        # self.y_max = coordinates[2]
        # self.x_max = coordinates[3]
        
    
    # @classmethod
    # def get_tomato_classification(cls):
        # # Get tomato classification value
        # return Tomato.classification
    
    # def get_id(self):
        # # Get tomato id
        # return self.tomato_id
    
    # def get_coordinates(self):
        # # Get tomato bbox coordinates
        # return self.coordinates
    
    # def get_confidence(self):
        # # Get confidence rating for the tomato instance
        # return self.confidence
    
    # def get_coordinate(self, coord_type):
        # # Gets a specified coordinate from the tomato instance
        # coord_dict = {'y_min': self.y_min, 'x_min': self.x_min, 'y_max': self.y_max, 'x_max': self.x_max}
        
        # return coord_dict[coord_type]
    
    # def denormalized_coords(self, img_width=0, img_height=0):
        # # Gets a set of de-normalized coordinates for specified image dimensions
        # y_min_denormalized = math.floor(self.y_min * img_height)
        # x_min_denormalized = math.floor(self.x_min * img_width)
        # y_max_denormalized = math.ceil(self.y_max * img_height)
        # x_max_denormalized = math.ceil(self.x_max * img_width)
        
        # return y_min_denormalized, x_min_denormalized, y_max_denormalized, x_max_denormalized

# class Tomatoes_Collection:
    
    # Tomato_classification = 0

    # def __init__(self):
        # self.tomatoes_collection = []
        
    # def collect_tomatoes(self, valid_detections, boxes, scores, classes):
        # # Create and store Tomato objects for all tomatoes detected
        # tomato_counter = 0
        
        # for index in range(valid_detections[0]):
            # if classes[0][index] == 0:#Tomato_classification:
                # tomato_counter += 1
                
                # tomato_id = f'tomato_{tomato_counter}'
                # coordinates = boxes[0][index]
                # confidence = scores[0][index]
                
                # tomato = Tomato(tomato_id, coordinates, confidence)
                
                # self.tomatoes_collection.append(tomato)

    # def get_tomato(self, tomato_id='', index=0, get_by_index=False):
        # # Get a specific tomato object in the collection
        # if get_by_index:
            # return self.tomatoes_collection[index]
        
        # for tomato in self.tomatoes_collection:
            # if tomato.get_id == tomato_id:
            
                # return tomato
    
    # def get_number_of_tomatoes(self):
        # # Get number of tomatoes found in the collectino (# of tomatoes detected)
        # return len(self.tomatoes_collection)
    
    # def list_tomatoes(self):
        # # List all tomatoes in the collection
        # tomatoes_list = []
        
        # for tomato in self.tomatoes_collection:
            # tomato_id = tomato.get_id()
            
            # tomatoes_list.append(tomato_id)
        
        # return tomatoes_list

# # Required Functions (Note: This functions was gathered from the detect.py script)

# def get_img_dims(img):
    # # height of image
    # h = np.shape(img)[0]
    
    # # width of image
    # w = np.shape(img)[1]
    
    # return w, h

# def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    # # Returns 2 arrays, one containing the bbox coordinate values in an array
    # # the other containing the confidance values of the predictions
    
    # scores_max = tf.math.reduce_max(scores, axis=-1)    # Reduce axis
    
    # mask = scores_max >= score_threshold    # Get mask to apply so only valid elements are kept
    # class_boxes = tf.boolean_mask(box_xywh, mask)   # Apply mask to xywh information array
    # pred_conf = tf.boolean_mask(scores, mask)   # Apply mask to scores (confidences) information array
    # class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]) # Reshape class_boxes so that it coforms to original scores shape
    # pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])   # Reshape pred_conf so that that if conforms to original scores shape

    # box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1) # Separate xy and wh information array

    # input_shape = tf.cast(input_shape, dtype=tf.float32)

    # box_yx = box_xy[..., ::-1]
    # box_hw = box_wh[..., ::-1]

    # box_mins = (box_yx - (box_hw / 2.)) / input_shape   # Calculate min coordinate values (both x and y)
    # box_maxes = (box_yx + (box_hw / 2.)) / input_shape  # Calculate max coordinate values (both x and y)
    
    # boxes = tf.concat([
        # box_mins[..., 0:1],  # y_min
        # box_mins[..., 1:2],  # x_min
        # box_maxes[..., 0:1],  # y_max
        # box_maxes[..., 1:2]  # x_max
    # ], axis=-1)
    
    # # Output 'boxes' is a tensor of shape (1, m, 4), where m is the number of detections found
    # # The axis with 4 elements is a tensor containing the normalized bbox coordinates in order:
    # # [y_min x_min y_max x_max]
   
    # # return tf.concat([boxes, pred_conf], axis=-1)
    # return (boxes, pred_conf)

# def apply_model(images_data, model_path):
    # # Apply tflite model to image data
    
    # interpreter = tf.lite.Interpreter(model_path=model_path)    # Load model into interpreter
    # interpreter.allocate_tensors()  # Allocate tensors

    # input_details = interpreter.get_input_details() # Get model input details
    # output_details = interpreter.get_output_details()   # Get model output details

    # interpreter.set_tensor(input_details[0]['index'], images_data)  # Set the interpreter tensor

    # interpreter.invoke()    # Invoke 

    # pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] # Predictions data
    
    # return pred

# # Constants
IMAGE_PATH = r'./data/color_image5.jpg'    # Path to input image
INPUT_SIZE = 416    # Size expected by model (416 by 416 pixels)      
MODEL_PATH = r'./checkpoints/yolov4-tiny-416.tflite'    # Path to tflite model
IOU_THRESHOLD = 0.45    # IOU threshold value
SCORE_THRESHOLD = 0.25  # Min confidence value for inclusion

# ##################################################### loadNprocess_img

# # Load image
# img = cv2.imread(IMAGE_PATH)    # Use cv2.imread to load image into img variable, cv2 gets images in BGR format
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color format of image from BGR to RGB

# # Process image
# image_data = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))    # Resize image for model application
# image_data = image_data / 255.  # Normalize image

# # Prepare image_data for model application
# images_data = []
# images_data.append(image_data)
# images_data = np.asarray(images_data).astype(np.float32)    # Convert images_data to numpy array                                                          

# ##################################################### 

# ##################################################### apply_model

# # Apply model on image and get predictions data
# predictions_data = apply_model(images_data, MODEL_PATH)    # Returns predictions data

# ##################################################### 

# ##################################################### get_data_arrays

# # Use 'filter_boxes' to extract bbox data and prediction confidence data
# boxes, pred_conf = filter_boxes(predictions_data[0], predictions_data[1], score_threshold=SCORE_THRESHOLD, input_shape=tf.constant([INPUT_SIZE, INPUT_SIZE]))

# # Use tf.image.combined_non_max_suppression to acquire data adn get rid of repeated bboxes
# boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    # boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    # scores=tf.reshape(
    # pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    # max_output_size_per_class=20,
    # max_total_size=20,
    # iou_threshold=IOU_THRESHOLD,
    # score_threshold=SCORE_THRESHOLD
# )

# # Turn data from tensorflow tensor into numpy array
# boxes = boxes.numpy()
# scores = scores.numpy()
# classes = classes.numpy()
# valid_detections = valid_detections.numpy()

# #####################################################


img, Tomatoes_Container = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH, IOU_THRESHOLD, SCORE_THRESHOLD, load_image=True, image_path=IMAGE_PATH)
# Tomatoes_Container = tlt.process_extract_tomatoes(IMAGE_PATH, INPUT_SIZE, MODEL_PATH, IOU_THRESHOLD, SCORE_THRESHOLD, img=img)

tomato1 = Tomatoes_Container.get_tomato(index=0, get_by_index=True)

img_width, img_height = tlt.get_img_dims(img)#tlt.get_img_dims(img)

# ymin, xmin, ymax, xmax = tomato1.denormalized_coords(img_width, img_height) 
ymin, xmin, ymax, xmax = tomato1.denormalized_bbox(img_width, img_height)


new_img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
cv2.imshow('bbox_tomat', new_img)
cv2.waitKey(0)