## Note:
# This module is actively used in the project as it avoids needless imports. If a clear reference of where each library is used in the code refer to 
# tomato_localization_toolset.py which does not import library functions/classes individually.

## Import necessary libraries
# Use from ... import ..., construction to optimize imports and avoid importing full library when not necessary

## TensorFlow imports
# import tensorflow as tf
from tensorflow import constant
from tensorflow import math
from tensorflow import boolean_mask
from tensorflow import reshape
from tensorflow import split
from tensorflow import cast
from tensorflow import concat
from tensorflow import shape
from tensorflow import image
from tensorflow import float32
from tensorflow import lite

# Math imports to avoid conflict with TensorFlow math
from math import ceil
from math import floor

# Below imports don't require as much resources as TensorFlow
import cv2
import numpy as np
from time import localtime, strftime

import pyrealsense2 as rs

################################################## Tomato Class
class Tomato:
    # This class is used to store relevant information for each tomato, think of each instance of the Tomato class as a file for a specific tomato
    # Each tomato detected has a Tomato class instance and the tomato's information can be accessed from it

    def __init__(self, tomato_id, bbox, confidence):
        self.tomato_id = tomato_id  # used to idenfity individual tomatoes in logs (string)
                                    # tomato_425 would be the 5th tomato detected when rover is in aisle
        self.bbox = bbox # coordinates are normalized and in format [y_min x_min y_max x_max] (float)
        self.confidence = confidence # confidence value that a detected toamto is a tomato (float)
        self.y_min = bbox[0] # normalized ymin of bbox for tomato instance (float)
        self.x_min = bbox[1] # normalized xmin of bbox for tomato instance (float)
        self.y_max = bbox[2] # normalized ymax of bbox for tomato instance (float)
        self.x_max = bbox[3] # normalized xmax of bbox for tomato instance (float)
        self.classifcation = '' # classification of tomato, either ripeness of tomato or 'defective' (string)
        self.volume = 0 # volume calculated for tomato
    
    ## Functions used to set attributes in Tomato
    
    def set_classification(self, classification):
        # Set the classification attribute of the tomato (string)
        self.classification = classification
    
    def set_volume(self, volume):
        # Set the volume of the tomato (float)
        # volume is represented in literes
        self.volume = volume
    
    ## Functions used to get data from Tomato
    
    def get_volume(self):
        # Get the volume of the tomato in liters (float)
        return self.volume
    
    def get_classification(self):
        # Get the string of a tomato (string)
        return self.classification
    
    def get_id(self):
        # Get tomato id of tomato (string)
        return self.tomato_id
    
    def get_bbox(self):
        # Get tomato bbox coordinates (array of floats) in format [y_min x_min y_max x_max]
        return self.bbox
    
    def get_confidence(self):
        # Get confidence rating for the tomato instance (float)
        return self.confidence
    
    def get_denormalized_bbox(self, w, h):
        # Gets a set of de-normalized bbox coordinates for specified image dimensions (w, h)
        # This works for an image which has been input into the yolov4 model
        # Floor and ceil functions used since de-normalized bbox coordinates need to be integers
        y_min_denormalized = floor(self.y_min * h)
        x_min_denormalized = floor(self.x_min * w)
        y_max_denormalized = ceil(self.y_max * h)
        x_max_denormalized = ceil(self.x_max * w)
        
        # denormalized bbox coordinates in format (y_min x_min y_max x_max) (integer)
        return y_min_denormalized, x_min_denormalized, y_max_denormalized, x_max_denormalized

################################################## 

################################################## Tomato Collection Class
class Tomatoes_Collection:
    # This class is used to create and hold tomato instances for each tomato found in the image. Each Tomato instance is stored in 'tomatoes_collection' attribure.
    # To access the Tomato instances, use the 'get_collection' method to acquire the list 'tomatoes_collection', then index whichever Tomato instance you desire to get.
    # A Tomatoes_Collection instance only holds data for a single image and is re-instanteated when a new image is taken

    def __init__(self):
        self.tomatoes_collection = [] # A list of instances from the 'Tomato' class form above, all tomatoes detected are given a Tomato class instance and stored
                                      # in this list for a specific Plant and Frame (Tomato instance)
        self.coordinates = (0, 0) # Represents (Aisle, Plant) coordinate position of the rover (integer)
        self.collection_timestamp = '' # Timestamp at which the image was taken for a (Aisle, Plant)
        self.collection_time = ''
        self.collection_date = ''

    ## Functions used to set attributes in Tomatoes_Collection

    def collect_tomatoes(self, valid_detections, boxes, scores, classes):
        
        # Create and store Tomato instances for all tomatoes detected
        #tomato_counter = 0 # used for creating tomato ids
        
        rng = np.random.default_rng() # used for creating tomato ids
        
        for index in range(valid_detections[0]):
            if classes[0][index] == 0:  # tomato has classification code 0 in yolov4 model, so if class is 0 generate data
                
                id_number = int(rng.integers(low=0, high=1000000, size=1))
                
                tomato_id = f'id_{id_number}' # generate tomato id
                bbox = boxes[0][index] # get tomato normalized bbox coordinates
                confidence = scores[0][index] # get tomato confidence rating
                
                tomato = Tomato(tomato_id, bbox, confidence) # create tomato instance with all the data gathered for a single tomato
                
                self.tomatoes_collection.append(tomato) # append tomato instance to tomatoes_collection list in Tomatoes_Collection instance
    
    def set_timestamp(self):
        # Set the rought time and date at which a collection was taken (string)
        timestamp = strftime("%m-%d-%Y %H:%M:%S", localtime())
        self.collection_timestamp = timestamp
        self.collection_date, self.collection_time = timestamp.split(" ")
    
    def set_coordinates(self, aisle, plant):
        # Sets the aisle, plant coordintaes of the rover (2-tuple of integers)
        self.coordinates = (aisle, plant)

    ## Functions used to get data from Tomatoes_Collection
    
    def get_collection(self):
        # Get list of Tomato instances, this is how you access data about each tomato (string of Tomato instances)
        return self.tomatoes_collection
    
    def get_tomato(self, index=0):
        # Get a specific tomato object in the collection (Tomato instance)
        return self.tomatoes_collection[index]

    def get_number_of_tomatoes(self):
        # Get number of tomatoes found in the collectino (# of tomatoes detected) (integer)
        return len(self.tomatoes_collection)
    
    def list_tomato_ids(self):
        # List all tomatoes ids in the collection (list of strings)
        tomatoes_list = []
        
        for tomato in self.tomatoes_collection:
            tomato_id = tomato.get_id()
            
            tomatoes_list.append(tomato_id)
        
        return tomatoes_list

    def get_frame_volume_yield(self):
        # Get total volume found in Tomatoes_Collection in liters (float)
        total_volume = 0
        
        for tomato in self.tomatoes_collection:
            total_volume += tomato.get_volume()
        
        return total_volume
    
    def get_collection_coordinates(self):
        # Get the (aisle, plant) coordinates of the rover (2-tuple of integers)
        return self.coordinates
    
    def get_timestamp(self):
        # Get the timestamp of when image was recorded (string)
        return self.collection_timestamp
    
    def get_time(self):
        # Get the timestamp of when image was recorded (string)
        return self.collection_times
    
    def get_date(self):
        # Get the timestamp of when image was recorded (string)
        return self.collection_date


## Auxilary Functions

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = constant([416,416])):
    # Returns 2 arrays, one containing the bbox coordinate values in an array and
    # the other containing the confidance values of the predictions
    
    scores_max = math.reduce_max(scores, axis=-1)    # Reduce axis
    
    mask = scores_max >= score_threshold    # Get mask to apply so only valid elements are kept
    class_boxes = boolean_mask(box_xywh, mask)   # Apply mask to xywh information array
    pred_conf = boolean_mask(scores, mask)   # Apply mask to scores (confidences) information array
    class_boxes = reshape(class_boxes, [shape(scores)[0], -1, shape(class_boxes)[-1]]) # Reshape class_boxes so that it coforms to original scores shape
    pred_conf = reshape(pred_conf, [shape(scores)[0], -1, shape(pred_conf)[-1]])   # Reshape pred_conf so that that if conforms to original scores shape

    box_xy, box_wh = split(class_boxes, (2, 2), axis=-1) # Separate xy and wh information array

    input_shape = cast(input_shape, dtype=float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape   # Calculate min coordinate values (both x and y)
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape  # Calculate max coordinate values (both x and y)
    
    boxes = concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    
    # Output 'boxes' is a tensor of shape (1, m, 4), where m is the number of detections found
    # The axis with 4 elements is a tensor containing the normalized bbox coordinates in order:
    # [y_min x_min y_max x_max]
   
    return (boxes, pred_conf)



## Process Functions
def load_and_process_img(image_path, input_size):
    # This functions loads the image into a variable and processes it for localization model
    # This function is mainly used for testing, applying the yolov4 model on an image stored in memory

    # Load image
    img = cv2.imread(image_path)    # Use cv2.imread to load image into img variable, cv2 gets images in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color format of image from BGR to RGB

    # Process image
    image_data = cv2.resize(img, (input_size, input_size))    # Resize image for model application
    image_data = image_data / 255.  # Normalize image

    # Prepare image_data for model application
    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)    # Convert images_data to numpy array
    
    return img, images_data # return the image and the image data for further processing
    
def process_img(img, input_size):
    # This functions processes the image for localization model
    # This function is mainly use when using the camera, as camera provides an image which 
    # does note need to be loaded from memory
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color format of image from BGR to RGB
    
    # Process image
    image_data = cv2.resize(img, (input_size, input_size))    # Resize image for model application
    image_data = image_data / 255.  # Normalize image

    # Prepare image_data for model application
    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)    # Convert images_data to numpy array
    
    return images_data

def apply_yolo_model(images_data, model_path):
    # Apply yolov4 model to image data
    # For this project, a yolov4-lite model was converted into a tensorflow lite model, you must use the tflite model with this function
    
    interpreter = lite.Interpreter(model_path=model_path)    # Load model into interpreter
    interpreter.allocate_tensors()  # Allocate tensors

    input_details = interpreter.get_input_details() # Get model input details
    output_details = interpreter.get_output_details()   # Get model output details

    interpreter.set_tensor(input_details[0]['index'], images_data)  # Set the interpreter tensor

    interpreter.invoke()    # Invoke 

    predictions = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] # Predictions data
    
    return predictions

def get_data_arrays(predictions, score_threshold, input_size, iou_threshold, num_objects=20, num_objects_per_class = 20):
    # This function processes and separates output data from model
    # This function returns 4 arrays as follows: boxes-bbox coordinates, scores-confidence ratings, 
    # classes-type of object detected (not ripeness classificaiton), valid_detections-the number of valid detections found
    # For boxes, scores, and classes; they correspond with eachother by index
    
    # Use 'filter_boxes' to extract bbox data and prediction confidence data
    boxes, pred_conf = filter_boxes(predictions[0], predictions[1], score_threshold=score_threshold, input_shape=constant([input_size, input_size]))

    # Use tf.image.combined_non_max_suppression to acquire data adn get rid of repeated bboxes
    boxes, scores, classes, valid_detections = image.combined_non_max_suppression(
        boxes=reshape(boxes, (shape(boxes)[0], -1, 1, 4)),
        scores=reshape(
        pred_conf, (shape(pred_conf)[0], -1, shape(pred_conf)[-1])),
        max_output_size_per_class=num_objects_per_class,
        max_total_size=num_objects,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    # Turn data from tensorflow tensor into numpy array
    return boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()

def process_extract_tomatoes(input_size, model_path, iou_threshold, score_threshold, img=None, load_image = False, image_path=''):
    # This function goes through the entire process of extracting data from detected tomatoes and returns a Tomatoes_Collection instance
    # for the image. It may also return an image for testing purposes if it is loaded form memory.
    
    # Either load and process the image, or just process depending on need
    if load_image:
        # Load and process the image for model application
        img, images_data = load_and_process_img(image_path, input_size) 
        
    else:
        images_data = process_img(img, input_size)
        
    # Apply the model to the processed image
    predictions = apply_yolo_model(images_data, model_path) 
    
    # Get data arrays from model output
    boxes, scores, classes, valid_detections = get_data_arrays(predictions, score_threshold, input_size, iou_threshold)
    
    # Make a tomato collection and populate with tomato data
    Tomatoes_Container = Tomatoes_Collection()
    Tomatoes_Container.collect_tomatoes(valid_detections, boxes, scores, classes)
    
    # Return objects depend on load_image
    if load_image:
        return img, Tomatoes_Container 
        
    else:
        return Tomatoes_Container

###### Debugging tools

def get_img_dims(img):
    # Returns the dimensions of an input image 
    
    # height of image
    h = np.shape(img)[0]
    
    # width of image
    w = np.shape(img)[1]
    
    return w, h

def show_bbox_tomato(Tomatoes_Collection, image, w, h, index, wait_key=0):
    # Show a bbox on an image

    tomato = Tomatoes_Collection.get_tomato(index=index) # Get Tomato instance from Tomatoes_Collection
    ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(w, h) # Get denormalized bbox coordinates from Tomato instance
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(wait_key)
    cv2.destroyWindow('image')

def show_point_on_image(image, x, y, thickness):
    # Show a point in an image
    
    image = cv2.circle(image, (x,y), radius=thickness, color=(0, 0, 255), thickness=-1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

def show_point_and_bbox_image(Tomatoes_Collection, image, w, h, index, x, y, thickness=-1):
    # Show a point and a bbox in an image

    tomato = Tomatoes_Collection.get_tomato(index=index)
    ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(w, h)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    image = cv2.circle(image, (x,y), radius=thickness, color=(0, 0, 255), thickness=-1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

def show_image(image, title, waitKey):
    # Show an image

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(waitKey)
    cv2.destroyWindow('image')

def show_regions(regions, index=-1):
    # Show regions, images of the individual tomatoes in the image
    
    if index == -1:
        for i, region in enumerate(regions):
            show_image(region, f'region{i}')
    else:
        region = regions[index]
        show_image(region, f'region{index}')

def save_bbox_image(Tomatoes_Collection, image, w, h, index, save_path):
    # Save an image to local memory

    tomato = Tomatoes_Collection.get_tomato(index=index) # Get Tomato instance from Tomatoes_Collection
    ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(w, h) # Get denormalized bbox coordinates from Tomato instance
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    save_image_flag = input(f'\nSave image (y or n)?: ') == 'y'
    
    if save_image_flag:
        img_name = input('Image name: ')
    
        cv2.imwrite(f'{save_path}/{img_name}.jpg', image)
    
    return None

#################################################
# Process:

    # 1) Define followign constants:
        # a) IMAGE_PATH (string): path to image from which you want to extract tomato data
        # b) INPUT_SIZE (int): size of images that ML model expects, 512
        # c) MODEL_PATH (string): path to tf lite model
        # d) IOU_THRESHOLD (floating point): IOU threshold for eliminating redundant coordinates, usually 0.45
        # e) SCORES_THRESHOLD (floating point): threshold for eliminating low confidence coordinates, usually 0.7
    
    # 2) Use process_extract_tomatoes to get either the collection of tomato data for each tomato detected, or 
       # the colleciont of tomato data and the image loaded
       # ex.
             # Tomatoes_Container = process_extract_tomatoes(image_path, input_size, model_path, iou_threshold, score_threshold, img=image)
        # or   img, Tomatoes_Container = process_extract_tomatoes(image_path, input_size, model_path, iou_threshold, score_threshold, load_image = True)
    
    # 3) Tomatoes_Container instance now has a list of Tomato instances, which hold all the data of interest for each tomato detected. To access this data, each  
       # Tomato instance can be accessed by Tomatoes_Container methods, and the data for each Tomato instance can be accessed by the Tomato instance methods.

#################################################

# Testing Area
    
