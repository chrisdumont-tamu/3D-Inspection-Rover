# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import chdir
from os.path import exists
import ms_program_utils as utils
#import darknet
import tomato_recognition.yolov4.compiled_darknet.darknet.darknet as darknet

##############################################
## Full Process

# 1) Load image of tomatoes into program
# 2) Run yolo_model on tomatoes image
# 3) Get bounding box coordinates 
    # - Transform bbox coords into usable form for extraction
# 4) Extract bounding box region for each tomato with Acc >= 60%
    # - Store in container
# 5) Process extracted image substes through defectiveness model 
    # - Program will detect if the tomato in the image subset is defectiveness
    # - If tomato is determined to be good, image subset will continue to ripeness program, else it will be logged and discarded
# 6) Process good extracted imagesthrough the ripeness model
    # - Their ripeness will be logged
# 7) Log and display results including: list of good image subset id's, list of bad image subset id's, display of a single good image subset with its
    # ripeness classification viewable

##############################################################################################################

### (1) - (3) Load image into program, apply yolov4, and get bounding box coordinates

# Variables
Tomatoes = {}
width = 0
height = 0
tomato_yield = 0

# Constants
CFG_FILE = r'C:\Users\felvi\ecen403programs\tomato_recognition\yolov4\compiled_darknet\darknet\cfg\yolov4-tiny-custom_test.cfg'
DATA_FILE = r'C:\Users\felvi\ecen403programs\tomato_recognition\yolov4\compiled_darknet\darknet\data\image_data.data'
WEIGHTS = r'C:\Users\felvi\ecen403programs\tomato_recognition\yolov4\compiled_darknet\darknet\backup\yolov4-tiny-custom_best.weights'
BATCH_SIZE = 1
ACC_THRESHOLD = 0.7
IMAGE_PATH = r'C:\Users\felvi\ecen403programs\saved_images\color_image3.jpg'
CONFIDENCE_THRESHOLD = 0.6

# Load trained model and other important data
network, class_names, class_colors = darknet.load_network(
        CFG_FILE,
        DATA_FILE,
        WEIGHTS,
        BATCH_SIZE
    )

# Apply the model and acquire the detection data
image, detections, bbox_img, width, height = utils.image_detection(IMAGE_PATH, network, class_names, class_colors, ACC_THRESHOLD, show_bboxes=True)

# Make image compliant with yolo bbox dimensions
# image = utils.imgCompliance(image, width, height)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
bbox_img = cv2.resize(bbox_img, (width, height), interpolation=cv2.INTER_LINEAR)

# Populate Tomatoes dict with tomato id and data
# Elements of form ID:[ID, label, confidence, coordinates]; we will append a classification to the end of list
# ID is str, label is str, confidence is str and percent, coordinates is 4-tuble giving (xmin, ymin, xmax, ymax)
Tomatoes = utils.addObjects(detections, 'Tomato', CONFIDENCE_THRESHOLD)

# Get depth data for image
# depth_frame_path = r'C:\Users\felvi\ecen403programs\saved_images'
# chdir(depth_frame_path)
# depth_frame = np.load('depth_data_0.npy')

#Modify depth_frame
# depth_frame.reshape(608, 608)

# Show full image
utils.displayImage(image)
utils.displayImage(bbox_img)

##############################################################################################################

### (4) Extract image subregions
# Subregions are stored in list
subregions_list = []
depth_frames_subregions_list = []
for tomato in Tomatoes:
    box_coords = Tomatoes[tomato][3]
    xmin, ymin, xmax, ymax = box_coords
    
    subregion_resized = cv2.resize(image[ymin:ymax, xmin:xmax], (256, 256), interpolation=cv2.INTER_LINEAR)
    
    # depth_frame_subregion = cv2.resize(depth_frame[ymin:ymax, xmin:xmax], (256, 256), interpolation=cv2.INTER_LINEAR)
    # Resize image as ripeness classification requires shape 4 by 1 input, current shape is 3 by 1
    # Value at index (0, 0) should be 0 as it represents the batch 
    subregion = np.expand_dims(subregion_resized, 0)
    # Add the subregion to the list
    elem = (tomato, subregion) # Elements are a 2-tuple of the id and the image
    subregions_list.append(elem)
    
    # depth_frames_subregions_list.append(depth_frame_subregion)
    
    # Uncomment the code below to view subregions for an image
    # plt.imshow(subregion_resized)
    # plt.axis('off')
    # plt.title(f"id: {tomato}")
    # plt.show()

# Display Subregions Alternative
utils.displaySubregions(subregions_list)


##############################################################################################################

## (5) - (6) Process image subsets through the defectivness and ripeness models

# Constants
DEFECT_MODEL_DIR = r'ripeness_classification\saved_models\defectiveness\defect1'
RIPE_MODEL_DIR = r'ripeness_classification\saved_models\ripeness\ripe1'
# # Classes should in same order as will be predicted
DEFECT_CLASS = ['Good', 'Blight', 'Moldy', 'Old']
RIPE_CLASS = ['Green', 'Breaker', 'Turning', 'Pink', 'Light Red', 'Red']

# # Load Models
defect_model = tf.keras.models.load_model(DEFECT_MODEL_DIR) # Loads defectiveness model
ripe_model = tf.keras.models.load_model(RIPE_MODEL_DIR) # Loads ripeness model

## Loop through subregions and determine which are good and get rid of the bad

## Note, unknown bug - first for-loop below stops prematurely

index_tracker = 0
for elem in subregions_list:
    # Get tomato defectiveness classification
    tomato = elem[0]
    subregion = elem[1]
    tomato_class = utils.getClass(defect_model, subregion, DEFECT_CLASS)
    
    # If not good, add classification to Tomatoes dictionary
    if tomato_class != 'Good':
        Tomatoes[tomato].append(tomato_class)
        
        # That tomato now has a classification, remove from subregions list
        del subregions_list[index_tracker]
    
    index_tracker += 1

# Use the below functions to see which subregions are still in list after removal of defectives
# utils.displaySubregions(subregions_list)


## Loop throuhg remaining subregions and determine their ripeness classification

for elem in subregions_list:
    # Get tomato defectiveness classification
    tomato = elem[0]
    subregion = elem[1]
    tomato_class = utils.getClass(ripe_model, subregion, RIPE_CLASS)
    
    # Add classification to Tomatoes dict
    Tomatoes[tomato].append(tomato_class)

##############################################################################################################

## (7) Log the tomato data

# Data will be logged as a csv file with the following format
# Format: id, confidence, classification, xmin, ymin, xmax, ymax  

log_file_path = 'logs/tomatolog.csv'

# Check if log file exists alrady
if exists(log_file_path):
    with open(log_file_path, 'a', newline='') as tlog:
        csv_writer = csv.writer(tlog, delimiter=',')

        # Use for-loop to loop through all values
        for tomato in Tomatoes:
            tom_id = Tomatoes[tomato][0]    #tomato id
            confidence = Tomatoes[tomato][2]    #confidence
            classification = Tomatoes[tomato][4]    #classification
            xmin, ymin, xmax, ymax = Tomatoes[tomato][3]    #coordinates
        
            data = [tom_id, confidence, classification, xmin, ymin, xmax, ymax]
        
            # Write data to log
            csv_writer.writerow(data)
            print("Tomato has been logged")

else:
    with open(log_file_path, 'w', newline='') as tlog:
        csv_writer = csv.writer(tlog, delimiter=',')
    
        # Specify and write field names and write it
        fields = ['id', 'confidence', 'classification', 'xmin', 'ymin', 'xmax', 'ymax', 'volume']
        csv_writer.writerow(fields)
    
        # Use for-loop to loop through all values
        for tomato in Tomatoes:
            tom_id = Tomatoes[tomato][0]    #tomato id
            confidence = Tomatoes[tomato][2]    #confidence
            classification = Tomatoes[tomato][4]    #classification
            xmin, ymin, xmax, ymax = Tomatoes[tomato][3]    #coordinates
        
            data = [tom_id, confidence, classification, xmin, ymin, xmax, ymax]
        
            # Write data to log
            csv_writer.writerow(data)
            print("Tomato has been logged")
        
