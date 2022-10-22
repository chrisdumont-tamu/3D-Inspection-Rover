## Import necessary libraries
import tensorflow as tf
import cv2
import numpy as np
import math
from time import gmtime, strftime


## Classes
class Tomato:

    def __init__(self, tomato_id, bbox, confidence):
        self.tomato_id = tomato_id
        self.bbox = bbox # Coords are normalized and in format [y_min x_min y_max x_max]
        self.confidence = confidence
        self.y_min = bbox[0]
        self.x_min = bbox[1]
        self.y_max = bbox[2]
        self.x_max = bbox[3]
        self.classifcation = ''
        self.volume = 0
    
    def set_classification(self, classification):
        # Set the classification attribute of the tomato
        # classification must be a string
        self.classification = classification
    
    def set_volume(self, volume):
        # Set the volume of the tomato
        self.volume = volume
    
    def get_volume(self):
        return self.volume
    
    def get_classification(self):
        # Classificaiton is a string
        return self.classification
    
    def get_id(self):
        # Get tomato id
        return self.tomato_id
    
    def get_bbox(self):
        # Get tomato bbox coordinates
        return self.bbox
    
    def get_confidence(self):
        # Get confidence rating for the tomato instance
        return self.confidence
    
    def get_bbox(self, bbox_type):
        # Gets a specified coordinate from the tomato instance
        bbox_dict = {'y_min': self.y_min, 'x_min': self.x_min, 'y_max': self.y_max, 'x_max': self.x_max}
        
        return bbox_dict[bbox_type]
    
    def get_denormalized_bbox(self, img_width=0, img_height=0):
        # Gets a set of de-normalized bbox coordinates for specified image dimensions
        y_min_denormalized = math.floor(self.y_min * img_height)
        x_min_denormalized = math.floor(self.x_min * img_width)
        y_max_denormalized = math.ceil(self.y_max * img_height)
        x_max_denormalized = math.ceil(self.x_max * img_width)
        
        return y_min_denormalized, x_min_denormalized, y_max_denormalized, x_max_denormalized


class Tomatoes_Collection:
    
    Tomato_classification_code = 0

    def __init__(self):
        self.tomatoes_collection = []
        self.coordinates = (0, 0)   # Represents (aisle, plant)
        self.total_volume_yield = 0
        self.collection_date = ''
        self.collection_time = ''
        
    def collect_tomatoes(self, valid_detections, boxes, scores, classes):
        # Create and store Tomato objects for all tomatoes detected
        tomato_counter = 0
        
        for index in range(valid_detections[0]):
            if classes[0][index] == 0:  # If object detected is a tomato
                tomato_counter += 1
                
                tomato_id = f'tomato_{tomato_counter}'
                bbox = boxes[0][index]
                confidence = scores[0][index]
                
                tomato = Tomato(tomato_id, bbox, confidence)
                
                self.tomatoes_collection.append(tomato)
    
    def set_time_date(self):
        # Sets the rought time and date at which a collection was taken
        time_date = strftime("%m-%d-%Y %H:%M:%S", gmtime())

        self.date, self.time = time_date.split(" ")

    
    def set_classifications(self, classifications):
        # Set the classification for each tomato in the collection
        for index, tomato in enumerate(self.tomatoes_collection):
            classification = classifications[index]
            tomato.set_classification(classification)
    
    def set_collection_coordinates(self, x, y):
        self.coordinates = (x, y)
    
    def set_volumes(self, volume_yields):
        # Set the classification for each tomato in the collection
        for index, tomato in enumerate(self.tomatoes_collection):
            volume = volume[index]
            tomato.set_volume(volume)
    
    # def set_total_volume_yield(self, total_volume_yield):
        # self.total_volume_yield = total_volume_yield
    
    def get_total_volume_yield(self):
        total_volume = 0
        
        for tomato in self.tomatoes_collection:
            total_volume += tomato.get_volume()
        
        return total_volume
    
    def get_tomato(self, tomato_id='', index=0, get_by_index=False):
        # Get a specific tomato object in the collection
        if get_by_index:
            return self.tomatoes_collection[index]
        
        for tomato in self.tomatoes_collection:
            if tomato.get_id == tomato_id:
            
                return tomato
        
    def get_collection(self):
        return self.tomatoes_collection
        
    def get_number_of_tomatoes(self):
        # Get number of tomatoes found in the collectino (# of tomatoes detected)
        return len(self.tomatoes_collection)
    
    def list_tomato_ids(self):
        # List all tomatoes in the collection
        tomatoes_list = []
        
        for tomato in self.tomatoes_collection:
            tomato_id = tomato.get_id()
            
            tomatoes_list.append(tomato_id)
        
        return tomatoes_list
        
    def get_bboxes_denormalized(self, width, height):
        tomato_bbox_dict = dict()
        
        for tomato in self.tomatoes_collection:
            tomato_bbox_dict[tomato.get_id()] = tomato.denormalized_bbox(width, height)
        
        return tomato_bbox_dict
    
    def get_edge_bbox_verts(self):
        # max edge coordinates
        ymin, xmin, ymax, xmax = 0, 0, 0, 0
        
        for tomato in self.tomatoes_collection:
            ymin = min(ymin, tomato.get_bbox('y_min'))
            xmin = min(xmin, tomato.get_bbox('x_min'))
            ymax = max(ymax, tomato.get_bbox('y_max'))
            xmax = max(xmax, tomato.get_bbox('x_max'))
        
        return ymin, xmin, ymax, xmax
        
    def get_max_bbox_verts_denormalized(self, w, h):
        # Max edge coordinates
        ymin, xmin, ymax, xmax = self.get_edge_bbox_verts()
        
        # Denormalize max edge coordinates
        y_min_denormalized = math.floor(ymin * h)
        x_min_denormalized = math.floor(xmin * w)
        y_max_denormalized = math.ceil(ymax * h)
        x_max_denormalized = math.ceil(xmax * w)
        
        return y_min_denormalized, x_min_denormalized, y_max_denormalized, x_max_denormalized
    
    def get_collection_coordinates(self):
        return self.coordinates
    
    def get_time_date(self):
        return (self.collection_time, self.collection_date) # (time, date)




## Auxilary Functions
def get_img_dims(img):
    # height of image
    h = np.shape(img)[0]
    
    # width of image
    w = np.shape(img)[1]
    
    return w, h

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



# Process Functions
def loadNprocess_img(image_path, input_size):
    # This functions loads the image into a variable and processes it for localization model

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
    
    return img, images_data
    
def process_img(img, input_size):
    # This functions processes the image for localization model
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color format of image from BGR to RGB
    
    # Process image
    image_data = cv2.resize(img, (input_size, input_size))    # Resize image for model application
    image_data = image_data / 255.  # Normalize image

    # Prepare image_data for model application
    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)    # Convert images_data to numpy array
    
    return images_data

def apply_model(images_data, model_path):
    # Apply tflite model to image data
    
    interpreter = tf.lite.Interpreter(model_path=model_path)    # Load model into interpreter
    interpreter.allocate_tensors()  # Allocate tensors

    input_details = interpreter.get_input_details() # Get model input details
    output_details = interpreter.get_output_details()   # Get model output details

    interpreter.set_tensor(input_details[0]['index'], images_data)  # Set the interpreter tensor

    interpreter.invoke()    # Invoke 

    predictions = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] # Predictions data
    
    return predictions

def get_data_arrays(predictions, score_threshold, input_size, iou_threshold, num_objects=20, num_objects_per_class = 20):
    # This function processes and separates output data from model
    
    # Use 'filter_boxes' to extract bbox data and prediction confidence data
    boxes, pred_conf = filter_boxes(predictions[0], predictions[1], score_threshold=score_threshold, input_shape=tf.constant([input_size, input_size]))

    # Use tf.image.combined_non_max_suppression to acquire data adn get rid of repeated bboxes
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=num_objects_per_class,
        max_total_size=num_objects,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    # Turn data from tensorflow tensor into numpy array
    return boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()

def process_extract_tomatoes(input_size, model_path, iou_threshold, score_threshold, img=0, load_image = False, image_path=''):
    # This function goes through the entire process of extracting data from detectections and 
    # returns the image and the tomato collection
    
    # Either load and process the image, or just process depending on need
    if load_image:
        # Load and process the image for model application
        img, images_data = loadNprocess_img(image_path, input_size) 
        
    else:
        images_data = process_img(img, input_size)
        
    # Apply the model to the processed image
    predictions = apply_model(images_data, model_path) 
    
    # Get data arrays from model output
    boxes, scores, classes, valid_detections = get_data_arrays(predictions, score_threshold, input_size, iou_threshold)
    
    # Make a tomato collection and populate with tomato data
    Tomatoes_Container = Tomatoes_Collection()
    Tomatoes_Container.collect_tomatoes(valid_detections, boxes, scores, classes)
    
    # Return type depends on load_image
    if load_image:
        return img, Tomatoes_Container 
        
    else:
        return Tomatoes_Container

###### Debugging tools
def show_bbox_tomato(Tomatoes_Collection, image, w, h, index):
    tomato = Tomatoes_Collection.get_tomato(index=index, get_by_index=True)
    ymin, xmin, ymax, xmax = tomato.denormalized_bbox(640, 480)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    

#################################################
# Process:

    # 1) Define followign constants:
        # a) IMAGE_PATH (string): path to image from which you want to extract tomato data
        # b) INPUT_SIZE (int): size of images that ML model expects, 416
        # c) MODEL_PATH (string): path to tf lite model
        # d) IOU_THRESHOLD (floating point): IOU threshold for eliminating redundant coordinates, usually 0.45
        # e) SCORES_THRESHOLD (floating point): threshold for eliminating low confidence coordinates, usually 0.25
    
    # 2) Use process_extract_tomatoes to get either the collection of tomato data for each tomato detected, or 
       # the colleciont of tomato data and the image loaded
       # ex.
             # Tomatoes_Container = process_extract_tomatoes(image_path, input_size, model_path, iou_threshold, score_threshold, img=image)
        # or   img, Tomatoes_Container = process_extract_tomatoes(image_path, input_size, model_path, iou_threshold, score_threshold, load_image = True)
    
    # 3) Tomatoes_Container instance now has a list of Tomato instances, which hold all the data of interest for each tomato detected. To access this data, each  
       # Tomato instance can be accessed by Tomatoes_Container methods, and the data for each Tomato instance can be accessed by the Tomato instance methods.