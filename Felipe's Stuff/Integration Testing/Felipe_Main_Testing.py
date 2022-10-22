# Necessary libraries
from tensorflow import lite
import cv2
import numpy as np
import math
from os.path import exists
import csv
import requests
import pyrealsense2 as rs

import tom_locator_toolset as tlt


image_path = r"C:\Users\felvi\ecen403programs\saved_images\color_frame_0.jpg"
# image_path = r"C:\Users\felvi\ecen403programs\saved_images\test_2.jpg"
################################################################################# Necessary Constants and Variables

## Tomato Localization
# Constants
MODEL_PATH_YOLO = r'C:\Users\felvi\ecen403programs\models_for_testing\yolov4-tiny-416.tflite'
IOU_THRESHOLD = 0#0.45
SCORES_THRESHOLD = 0.25
INPUT_SIZE = 416
WIDTH, HEIGHT = 640, 480

## Tomato Classification
# Constants
MODEL_PATH_RIPENESS = r'C:\Users\felvi\ecen403programs\models_for_testing\ripe1.tflite'
MODEL_PATH_DEFECTIVENESS = r'C:\Users\felvi\ecen403programs\models_for_testing\defective1.tflite'
RIPENESS_CLASSES = ['green', 'breaker', 'turning', 'pink', 'light red', 'red']
DEFECTIVENESS_CLASSES = ['good', 'blight', 'modly', 'old']

## Volume Estimation
# Constants
LOG_FILE_PATH = r'C:\Users\felvi\ecen403programs\models_for_testing\tomatoes_log.csv' # Path to CSV where data is exported


# Variables
Total_Volume_Yield = 0;

## Upload Data to Database
# Constants
DYNAMODB_ENDPOINT = 'https://cnpu0bqb4i.execute-api.us-east-1.amazonaws.com/beta/items'


## External
Aisle = 0
Plant = 0

#################################################################################

################################################################################# Necessary code to work camera

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30) # rs.format.rgb8

# # Start streaming
# pipeline.start(config)

#################################################################################

################################################################################# Functions for integration code

def Felipe_Main_Camera():
    global Total_Volume_Yield
    
    ## Get the image and extract the lozalization data
    Tomatoes_Collection, image, depth_frame = Get_Image_Data()
    
    # Set the tomatoes collection coordinates (aisle and plant)
    Tomatoes_Collection.set_collection_coordinates(Aisle, Plant)
    
    ## Are there tomatoes in the frame? For Dalton
    # Disregard everything else if there are no tomatoes in frame
    Tomatoes_in_Frame = Tomatoes_Collection.get_number_of_tomatoes() > 0
    if not Tomatoes_in_Frame:   # If there are no tomatoes in the frame
        return False    # Returns False
    
    ## Get the volume yield of tomatoes found and apply it to tomato collection
    # volume_yields = Get_Volume_Yields(Tomatoes_Collection, depth_frame,)
    # Tomatoes_Collection.set_volume_yields(volume_yields)
    # Total_Volume_Yield += Tomatos_Collection.get_total_volume_yield()
    
    ## Get a list of the regions in the image whic contaitn a tomato for input into the classification models
    tomato_regions = Get_Tomato_Regions(Tomatoes_Collection, image, WIDTH, HEIGHT)
    ##!! Look into using a method that does not store all the tomato regions in a list but rather
    ##!! extracts one region and its data at a time so that the RAM is not as heavy
    ##!! The get_max_bbox_verts_denormalized method does not seem to work, fix
    
    ## Apply the classification models and add the classificatiosn to each respective tomato
    classifications = Get_Tomato_Classifications(tomato_regions)
    Tomatoes_Collection.set_classifications(classifications)
    
    print('\n\n', Tomatoes_Collection.get_date_time(),'\n\n')
    
    # Export Tomato data to csv file
    # Datat that will be exported for each tomato: tom_id, confidence, classification, volume, aisle, plant, time, date
    log_tomato_data(Tomatoes_Collection, LOG_FILE_PATH)
    
    return True
    
def Get_Image_Data():
    # # This function takes an image from the RealSense Camera, applies the localization model, and returns a tomato collection, the color image, and depth data
    # # Get frame from camera
    # frame = pipeline.wait_for_frames()
    # color_frame = frame.get_color_frame()
    # depth_frame = frame.get_depth_frame()
    
    # # Convert color frame into color image
    # # Image will come in RGB format, as uint8, of shape (640, 480, 3)
    # color_image = np.asanyarray(color_frame.get_data())
    
    # # Extract localization data from color image
    # # Tomatoes_Container = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH_YOLO, IOU_THRESHOLD, SCORES_THRESHOLD, img=color_image, load_image=False)
    # img, Tomatoes_Container = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH_YOLO, IOU_THRESHOLD, SCORES_THRESHOLD, load_image=False)
    
    # # Set timestamp of when tomato collection was taken
    # Tomatoes_Container.set_time_and_date()
    
    # return Tomatoes_Container, color_image, depth_frame
    
    ################################################################## Testing
    # frame = pipeline.wait_for_frames()
    # color_frame = frame.get_color_frame()
    # depth_frame = frame.get_depth_frame()
    
    # Convert color frame into color image
    # Image will come in RGB format, as uint8, of shape (480, 640, 3)
    # color_image = np.asanyarray(color_frame.get_data())
    
    # Extract localization data from color image
    # Tomatoes_Container = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH_YOLO, IOU_THRESHOLD, SCORES_THRESHOLD, img=color_image, load_image=False)
    img, Tomatoes_Container = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH_YOLO, IOU_THRESHOLD, SCORES_THRESHOLD, load_image=True, image_path=image_path)
    color_image = img
    depth_frame = img
    
    # Set timestamp of when tomato collection was taken
    Tomatoes_Container.set_time_date()
    
    return Tomatoes_Container, color_image, depth_frame



def Get_Volume_Yields(Tomato_Collection, depth_frame):
    return None

def Get_Tomato_Regions(Tomatoes_Collection, img, w, h):
    tomato_regions = []
    collection = Tomatoes_Collection.get_collection()
    
    # ymin_edge, xmin_edge, ymax_edge, xmax_edge = Tomatoes_Collection.get_max_bbox_verts_denormalized(w, h) !! get_max_bbox_verts_denormalized does not work properly, fix

    # Reduce image so that cropping time for a region reduces
    # img = img[ymin_edge:ymax_edge, xmin_edge:xmax_edge]   !! get_max_bbox_verts_denormalized does not work properly, fix
    
    for tomato in collection:
        ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(w, h) #ymin, xmin, ymax, xmax
        
        region = img[ymin:ymax, xmin:xmax]
        
        tomato_regions.append(region)
    
    return tomato_regions   # Tomato regions are ready for input into classification models
    
def Apply_Classification_Model(model_path, img, classes):
    # This function returns a prediction of a tf lite model given a model path and image path
    # The prediction is a string with the most probable classification as determined by the model

    # Preprocess image so that regions are ready for input to classificaiton models
    # Models require inputs of shape (1, 256, 256, 3) and dtype float32
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR) # Resize image to 256x256
    img = np.float32(img)   # Convert image dtype from uint8 to float32
    img = np.expand_dims(img, 0)  # Change img shape from (256, 256, 3) to (1, 256, 256, 3)

    # Load tflite model into interpreter and allocate tensors
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Apply the model
    interpreter.set_tensor(input_details[0]['index'], img)
    
    interpreter.invoke()
    
    # Get output data of tf lite model. It is a list of decimals representing the confidence for each 
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Use the output of the model to get the class of the highest confidence
    max_confidence_index = np.argmax(output_data)
    classification = classes[max_confidence_index]
    
    return classification
    
def Get_Tomato_Classification(img):
    # Determine if the tomato is good or defective
    defectiveness_class = Apply_Classification_Model(MODEL_PATH_DEFECTIVENESS, img, DEFECTIVENESS_CLASSES)
    
    # If tomato is defective then its classification is its defect
    if defectiveness_class != 'good':
        return defectiveness_class
    
    # If tomato is good, is classification is its ripeness
    ripeness_class = Apply_Classification_Model(MODEL_PATH_RIPENESS, img, RIPENESS_CLASSES)
    
    return ripeness_class
    
def Get_Tomato_Classifications(regions):
    # Gets a list for the classifications of each subregion extracted
    classifications = []
    
    for region in regions:
        classification = Get_Tomato_Classification(region)
        
        classifications.append(classification)
    
    return classifications

def log_tomato_data(Tomato_Collection, log_file_path):
    # Exports tomato data into csv file
    log_exists = exists(log_file_path)  # Check if log exists_ already
    
    # If log does note exist, create one
    if not log_exists:
        with open(log_file_path, 'w', newline='') as log:
            csv_writer = csv.writer(log, delimiter=',')
        
            # Specify and write field names and write it
            fields = ['id', 'confidence', 'classification', 'volume',  'aisle', 'plant', 'time', 'date']
            csv_writer.writerow(fields)
        
            # Use for-loop to loop through all values
            for tomato in Tomato_Collection.get_collection():
                tom_id =  tomato.get_id()   # tomato id (string)
                confidence = tomato.get_confidence()  * 100  # confidence % (floating point)
                classification = tomato.get_classification()   # classification (string)
                volume = tomato.get_volume()    # coordinates (integer)
                aisle = Tomato_Collection.get_collection_coordinates()[0]    # aisle that collection is found in (integer)
                plant = Tomato_Collection.get_collection_coordinates()[1]    # plant that collection is found at (integer)
                time = Tomato_Collection.get_time_date()[0]     # timestamp (string)
                date = Tomato_Collection.get_time_date()[1]     # date (string)
                
                data = [tom_id, confidence, classification, volume, aisle, plant, time, date]
            
                # Write data to log
                csv_writer.writerow(data)
                # print("Tomato has been logged")
            
    else:
        with open(log_file_path, 'a', newline='') as log:
            csv_writer = csv.writer(log, delimiter=',')
        
            # Use for-loop to loop through all values
            for tomato in Tomato_Collection.get_collection():
                tom_id =  tomato.get_id()   # tomato id (string)
                confidence = tomato.get_confidence()  * 100  # confidence % (floating point)
                classification = tomato.get_classification()   # classification (string)
                volume = tomato.get_volume()    # coordinates (integer)
                aisle = Tomato_Collection.get_collection_coordinates()[0]    # aisle that collection is found in (integer)
                plant = Tomato_Collection.get_collection_coordinates()[1]    # plant that collection is found at (integer)
                time = Tomato_Collection.get_time_date()     # timestamp (string)
                date = Tomato_Collection.get_time_date()     # date (string)
                
                data = [tom_id, confidence, classification, volume, aisle, plant, time, date]
            
                # Write data to log
                csv_writer.writerow(data)
                # print("Tomato has been logged")

####################################################################
# The following code is for exporting the tomato log at the end of the rover run

def upload_piece_data(tom_id, confidence, classification, volume, aisle, plant, time, date):
    data = {"ID": identifier, "Confidence": confidence, "Stage": classifiation, "Volume":volume, "Aisle": aisle, "Plant": plant, "Time":time, "Date": date}
    reply = requests.put(DYNAMODB_ENDPOINT, json = data)
    return reply.status_code

def export_log_to_database(log_file_path, endpoint):
    # indexes
    line_index = 0
    (id_indx, conf_indx, class_indx, vol_indx, aisle_indx, plant_indx, time_indx, date_indx) = tuple(range(7))
    
    with open(file_name) as f:
        
        csv_reader = csv.reader(f, delimiter=',')
        
        for row in csv_reader:
            if line_index == 0:
            
                line_index += 1
                continue
            
            identifier = row[id_index]
            confidence = row[conf_indx]
            classification = row[class_indx]
            volume = row[vol_indx]
            aisle = row[aisle_indx]
            plant = row[plant_indx]
            time = row[time_indx]
            date = row[date_indx]
            
            
            
            upload_piece_data(identifier, confidence, classification, aisle, plant, volume, time)
            line_index += 1


## Testing Area
# Felipe_Main_Camera()
Tomatoes_Collection, image, depth_frame = Get_Image_Data()
print(Tomatoes_Collection.get_time_date())