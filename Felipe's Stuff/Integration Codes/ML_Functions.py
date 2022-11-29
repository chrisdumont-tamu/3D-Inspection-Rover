# Necessary libraries
from tensorflow import lite
import cv2
import numpy as np
import math
from os.path import exists
import csv
import requests
import pyrealsense2.pyrealsense2 as rs

import tomato_localization_toolset_optimized as tlt

################################################################################# Necessary Constants and Variables

## Tomato Localization
# Constants
# MODEL_PATH_YOLO = r'C:\Users\felvi\ecen403programs\volume_estimation\ml_models_testing\yolov4-tiny-416-run1.tflite'
# MODEL_PATH_YOLO = r'C:\Users\felvi\ecen403programs\volume_estimation\ml_models_testing\yolov4-tiny-416-run5-last.tflite'
MODEL_PATH_YOLO = r'/home/pi/BOB_Inspection_Rover/models/yolov4-tiny-416-run1.tflite'
IOU_THRESHOLD = 0.45 # Multiple bounding boxes (bboxes) may be detected for the same toamto, a greater IOU_THRESHOLD reduces this but may discard non-duplicate bboxes if they are to close together.
SCORES_THRESHOLD = 0.50 # Used to filter confidence of toamtoes detected, tomatoes below this threshold will not be detected
INPUT_SIZE = 416 # Input size of yolov4 detection model, usually 416 or 512. Consult 'input details' of tflite model if not sure.
WIDTH, HEIGHT = 640, 480 # Width and heigh of image that will be processed. Primarily used to get denormalized bbox coordinates.

## Tomato Classification
# Constants

# Paths used in the pi
MODEL_PATH_RIPENESS = r'/home/pi/BOB_Inspection_Rover/models/ripe1.tflite' # Path to ripeness classification model
# MODEL_PATH_DEFECTIVENESS = r'/home/pi/BOB_Inspection_Rover/models/defect_oversample_binary_80.tflite'
MODEL_PATH_DEFECTIVENESS = r'/home/pi/BOB_Inspection_Rover/models/defect_oversample_150.tflite' # Path to defectiveness classification model

# Paths used in local computer
# MODELL_PATH_RIPENESS = r'C:\Users\felvi\ecen403programs\volume_estimation\ml_models_testing\ripe1.tflite'
# MODEL_PATH_DEFECTIVENESS = r'C:\Users\felvi\ecen403programs\volume_estimation\ml_models_testing\defective1.tflite'

RIPENESS_CLASSES = ['green', 'breaker', 'turning', 'pink', 'light red', 'red'] # Classes associated with ripeness classificaiton model
DEFECTIVENESS_CLASSES = ['good', 'blight', 'modly', 'old'] # Classes associated with defectiveness classification model
#DEFECTIVENESS_CLASSES = ['good', 'defective']

## Volume Estimation
# Constants
TOMATO_DEPTH_THRESHOLD = 0.085 # Threshold to validate a valid vertical or horizontal point during 
                             # volume estimation. Used to discount objects in the background as part 
                             # of the tomato.

## Upload Data to Database
# Constants
LOG_FILE_PATH = r'/home/pi/BOB_Inspection_Rover/logs/tomatoes_log.csv' # Path to CSV where data is exported
DYNAMODB_ENDPOINT = 'https://cnpu0bqb4i.execute-api.us-east-1.amazonaws.com/beta/items'
# LOG_FILE_PATH = r'C:\Users\felvi\ecen403programs\volume_estimation\logs\tomatoes_log.csv'

#################################################################################

################################################################################# Necessary code to work RealSense D435 camera for testing purposes
## Camera setup

# Create a pipeline object to own and handle realsnese camera
#pipeline = rs.pipeline()

# Configure the streams
#config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Begin streaming
#pipeline.start(config)

# Perform color to depth alignment
#align_to = rs.stream.color
#align = rs.align(align_to)

#################################################################################

################################################################################# Main ML Function

def Process_Data(pipeline, align, aisle, plant, return_collection=False):

    print(f'\n\nBeginning data processing\n')

    # Main processing function. Gathers data from image for each tomato and uploads data to logs
    
    print(f'Aisle: {aisle}, Plant: {plant}')

    ## Get the image and extract the lozalization data
    Tomatoes_Collection, color_image, color_frame, depth_frame = Get_Image_Data(pipeline,  align)
    
    ## Set timestamp of when tomato collection was taken
    Tomatoes_Collection.set_timestamp()
    
    print(f'Timestamp: {Tomatoes_Collection.get_timestamp()}')

    ## Set the tomatoes collection coordinates (aisle and plant)
    Tomatoes_Collection.set_coordinates(aisle, plant)

    print(f'Number of Tomatoes in Frame: {Tomatoes_Collection.get_number_of_tomatoes()}')

    ## Are there tomatoes in the frame? For determining if paning action is taken by camera gimble
    Tomatoes_in_Frame = Tomatoes_Collection.get_number_of_tomatoes() > 0

    ## Return if there are no tomatoes in frame
    if not Tomatoes_in_Frame:   # If there are no tomatoes in the frame
        return False    # Returns False
    
    ######################################################### Determine and set tomato classification and yield
    collection = Tomatoes_Collection.get_collection() # Get collection from Tomatoes_Collection
    
    # Show the bbox for each tomato for testing purposes
    for index, tomato in enumerate(collection):
    	tlt.show_bbox_tomato(Tomatoes_Collection, color_image, WIDTH, HEIGHT, index=index, wait_key=5000)

    classifications_list = [] # List to hold classification for each tomato

    for tomato in collection:
        # Get bbox coordinates
        ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(WIDTH, HEIGHT) #ymin, xmin, ymax, xmax
        
        # Get tomato region
        region = color_image[ymin:ymax, xmin:xmax]
        
        # Get tomato classification
        classification = Get_Tomato_Classification(region)
              # Set tomato classification
        tomato.set_classification(classification)
        classifications_list.append(classification)
        
        # Get tomato volume
        volume = get_volume_of_tomato(depth_frame, color_frame, tomato, TOMATO_DEPTH_THRESHOLD, WIDTH, HEIGHT)
        # Set tomato volume
        tomato.set_volume(volume)
    
    # Get the total tomato volume yield in frame
    frame_volume_yield = Tomatoes_Collection.get_frame_volume_yield()

    print(f'Volume in Frame [L]: {frame_volume_yield}')
    print(f'Classifications: {classifications_list}')

    #########################################################
    
    # Export Tomato data to csv file
    # Datat that will be exported for each tomato: tom_id, confidence, classification, volume, aisle, plant, timestamp
    print(f'Logging tomato data')
    log_tomato_data(Tomatoes_Collection, LOG_FILE_PATH)

    print(f'Data processing complete\n\n')
    if return_collection:
        return True, Tomatoes_Collection
    return True
##################################################################################

################################################################################## Get Data From Image
def Get_Image_Data(pipeline, align):
    # # This function takes an image from the RealSense Camera, applies the localization model, and returns a tomato collection, the color image, and depth data
    
    # Get 15 frames for temporal filtering
    for i in range(15):
        frames = pipeline.wait_for_frames()
        if i == 14:
            # Aligned the frames
            aligned_frames = align.process(frames)

    #Color frame
    depth_frame = aligned_frames.get_depth_frame()

    #Color frame
    color_frame = aligned_frames.get_color_frame()
    
    # # Convert color frame into color image
    # # Image will come in RGB format, as uint8, of shape (640, 480, 3)
    color_image = np.asanyarray(color_frame.get_data())
    
    # # Extract localization data from color image
    Tomatoes_Collection = tlt.process_extract_tomatoes(INPUT_SIZE, MODEL_PATH_YOLO, IOU_THRESHOLD, SCORES_THRESHOLD, img=color_image, load_image=False)

    # return Tomatoes_Collection, color_image, depth_frame
    return Tomatoes_Collection, color_image, color_frame, depth_frame
##################################################################################

################################################################################## Volume Estimation Functions
def get_camera_intrinsics(depth_frame, color_frame):
    # Intrinsics needed for calculating 3D coordinates at a specified pixel
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    
    return depth_intrin
    
def get_center_pixel_coords(xmin, ymin, xmax, ymax):
    # Gets center pixel for a bbox
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)
    return (x_center, y_center)

def pixel_to_point(intrinsics, depth_frame, x, y):
    # Gets the 3D point of a specified pixel
    # Need intrinsics, [x, y] pixel, and depth to use 
    # rs.rs2_deproject_pixel_to_point(...)
    # Calculate depth of pixel
    depth = depth_frame.get_distance(x, y)
    
    # Calculate point in 3D
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    
    return point # point is a 3-tuple float of shape (1, 3)

def check_point_validity(point):
    # Checks if point is valid. Invalid point gives a [0, 0, 0] result, means 3D point in pixel is too far away or too close to camera
    return point != [0, 0, 0]

def check_depth_threshold(depth_threshold, point, center_point):
    # Checks the Z distance between 3D point at center of tomato and a different point. Used for reducing inaccuarcy due to background.
    return depth_threshold >= abs(point[2] - center_point[2])

def get_valid_horizontal_edge_point(intrinsics, depth_frame, depth_threshold, xmin, y_center, x_center, center_point, return_valid_pixel_coords=False):
    # This function calculates a valid point in the image along the horizonal direction
    # It begins at (xmin, ycenter) and incremetns xmin until (xcenter, ycenter) is reached or a valid point is reached
    while True:
        # If xmin reaches x_cetner, return a non-valid point
        point = pixel_to_point(intrinsics, depth_frame, xmin, y_center)
        valid_point = check_point_validity(point)
        valid_threshold = check_depth_threshold(depth_threshold, point, center_point)
        if xmin == x_center:
            return (0, 0, 0)
        
        elif (valid_point and valid_threshold):
            if return_valid_pixel_coords:
                return point, (xmin, y_center)
            return point
        
        else:
            xmin += 1

def get_valid_vertical_edge_point(intrinsics, depth_frame, depth_threshold, x_center, ymin, y_center, center_point, return_valid_pixel_coords=False):
    # This function calculates a valid point in the image along the vertical direction
    # It begins at (xcenter, ymin) and incremetns ymin until (xcenter, ycenter) is reached or a valid point is reached

    while True:
        # If ymin reaches y_cetner, return a non-valid point
        point = pixel_to_point(intrinsics, depth_frame, x_center, ymin)
        valid_point = check_point_validity(point)
        valid_threshold = check_depth_threshold(depth_threshold, point, center_point)
        if ymin == y_center:
            return (0, 0, 0)
        
        elif (valid_point and valid_threshold):
            if return_valid_pixel_coords:
                return point, (x_center, ymin)
            return point
        
        else:
            ymin += 1
            
def get_semi_axes(center_point, horizontal_point, vertical_point):
    # This function determines the semi-axes from the points given (in meters)
    # Used for modeling tomato as an ellipsoid
        
    x_semi_axis = abs(center_point[0] - horizontal_point[0])
    y_semi_axis = abs(center_point[1] - vertical_point[1])
    z_semi_axis = x_semi_axis # right now the assumption is that the tomato is symmetrical about the y axis, so x and z semi-axes are equal
    
    return x_semi_axis, y_semi_axis, z_semi_axis

def calc_ellipsoide_volume(x_semi_axis, y_semi_axis, z_semi_axis):
    # This function determins the ellipsoid volume after which the tomato is modeled; in cubic meters
    return (4 / 3) * math.pi * x_semi_axis * y_semi_axis * z_semi_axis

def get_volume_of_tomato(depth_frame, color_frame, tomato, depth_threshold, w, h):
    # This function puts all the volume functions together to estimate the volume of a tomato in liters
    
    # get camera intrinsics for point calculation
    depth_intrin = get_camera_intrinsics(depth_frame, color_frame)
    
    # get bbox_coordinates of tomato
    ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(w, h) 
    
    # get center pixel coordinates of bbox, is a 2-tuple of form (x_center, y_center)
    center_pixel = get_center_pixel_coords(xmin, ymin, xmax, ymax)
    
    # get center point of tomato (center point of bbox) for semi-axis calculation
    center_point = pixel_to_point(depth_intrin, depth_frame, center_pixel[0], center_pixel[1])
    
    if center_point == [0, 0, 0]:
        print(f'\nInvalid Center Point!!!\n')
        return -1
    
    # get a valid horizontal edge point of tomato for calculating x semi-axis calculation
    horizontal_point = get_valid_horizontal_edge_point(depth_intrin, depth_frame, depth_threshold, xmin, center_pixel[1], center_pixel[0], center_point)
    
    
    if horizontal_point == [0, 0, 0]:
        print(f'\nInvalid Horizontal Point!!!\n')
        return -2
    
    # get a valid vertical edge point of tomato for y semi-axis calculation
    vertical_point = get_valid_vertical_edge_point(depth_intrin, depth_frame, depth_threshold, center_pixel[0], ymin, center_pixel[1], center_point)
    
    if vertical_point == [0, 0, 0]:
        print(f'\nInvalid Horizontal Point!!!\n')
        return -2
    
    print(f'\nCenter Point: {center_point}')
    print(f'Horizontal Point: {horizontal_point}')
    print(f'Vertical Point: {vertical_point}')
    
    # Calculate semi-axes of tomato ellipsoid model
    semi_axes = get_semi_axes(center_point, horizontal_point, vertical_point)
    
    print(f'Ellipsoid Semi-axes: x = {semi_axes[0]}, y = {semi_axes[1]}, z = {semi_axes[2]}')
    
    # Calculate the volume of the tomato ellipsoid model (in m^3)
    ellipsoid_volume = calc_ellipsoide_volume(semi_axes[0], semi_axes[1], semi_axes[2])
    
    print(f'Ellipsoid Volume [m^3]: {ellipsoid_volume}')
    
    # Return the volume estimate of the tomato in liters (1 m^3 = 1000 liters)
    
    print(f'Ellipsoid Volume [L]: {round(ellipsoid_volume * 1000, 4)}\n')
    
    return round(ellipsoid_volume * 1000, 4)
################################################################################## 

################################################################################## ML Functions for Tomato Classification
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
##################################################################################

################################################################################## Function to Log Tomato Data
def log_tomato_data(Tomato_Collection, log_file_path):
    # Exports tomato data into csv file
    log_exists = exists(log_file_path)  # Check if log exists_ already
    
    # If log does note exist, create one
    if not log_exists:
        with open(log_file_path, 'w', newline='') as log:
            csv_writer = csv.writer(log, delimiter=',')
        
            # Specify and write field names and write it
            fields = ['id', 'confidence', 'classification', 'volume',  'aisle', 'plant', 'timestamp']
            csv_writer.writerow(fields)
        
            # Use for-loop to loop through all values
            for tomato in Tomato_Collection.get_collection():
                tom_id =  tomato.get_id()   # tomato id (string)
                confidence = tomato.get_confidence()  * 100  # confidence % (floating point)
                classification = tomato.get_classification()   # classification (string)
                volume = tomato.get_volume()    # coordinates (integer)
                aisle = Tomato_Collection.get_collection_coordinates()[0]    # aisle that collection is found in (integer)
                plant = Tomato_Collection.get_collection_coordinates()[1]    # plant that collection is found at (integer)
                timestamp = Tomato_Collection.get_timestamp() # timestamp (string)
                # time = Tomato_Collection.get_time_date()     # timestamp (string)
                # date = Tomato_Collection.get_time_date()     # date (string
                
                data = [tom_id, confidence, classification, volume, aisle, plant, timestamp]
            
                # Write data to log
                csv_writer.writerow(data)
            
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
                timestamp = Tomato_Collection.get_timestamp() # timestamp (string)
                # time = Tomato_Collection.get_time_date()[0]     # timestamp (string)
                # date = Tomato_Collection.get_time_date()[1]     # date (string)
                
                data = [tom_id, confidence, classification, volume, aisle, plant, timestamp]
            
                # Write data to log
                csv_writer.writerow(data)
##################################################################################


##################################################################################
# The following code is for exporting the tomato log at the end of the rover run

def upload_piece_data(identifier, confidence, classification, volume, aisle, plant, timestamp):
    data = {"id": identifier, "Confidence": confidence, "Stage": classification, "Volume":volume, "Aisle": aisle, "Plant": plant, "Timestamp": timestamp}
    reply = requests.put(DYNAMODB_ENDPOINT, json = data)
    return reply.status_code

def Export_Log_to_Database():
    
    print(f'\n\nExporting log to database\n\n')

    # indexes
    line_index = 0
    # (id_indx, conf_indx, class_indx, vol_indx, aisle_indx, plant_indx, time_indx, date_indx) = tuple(range(8))
    (id_indx, conf_indx, class_indx, vol_indx, aisle_indx, plant_indx, timestamp_indx) = tuple(range(7))
    
    with open(LOG_FILE_PATH) as f:
        
        csv_reader = csv.reader(f, delimiter=',')
        
        for row in csv_reader:
            if line_index == 0:
            
                line_index += 1
                continue
            
            identifier = row[id_indx]
            confidence = row[conf_indx]
            classification = row[class_indx]
            volume = row[vol_indx]
            aisle = row[aisle_indx]
            plant = row[plant_indx]
            timestamp = row[timestamp_indx]

            
            upload_piece_data(identifier, confidence, classification, volume, aisle, plant, timestamp)
            line_index += 1

##################################################################################

## Testing Area
#Tomatoes_Container, color_image, color_frame, depth_frame = Get_Image_Data(pipeline, align)
# Tomatoes_Container, color_image, color_frame, depth_frame = Get_Image_Data(pipeline, align)
#Tomatoes_Container.set_timestamp()
#tlt.show_image(color_image, 'image')

#print(f'Tomatoes in Frame: {Tomatoes_Container.get_number_of_tomatoes()}')
#Tomatoes_in_Frame = Tomatoes_Container.get_number_of_tomatoes() > 0

# if Tomatoes_in_Frame:
    # Tomato_in_Frame, Tomatoes_Collection = Process_Data(pipeline, align, 0, 0, return_collection=True)
    # Frame_Yield = Tomatoes_Collection.get_frame_volume_yield()

    # print(f'\n\nNumber of Tomatoes: {Tomatoes_Collection.get_number_of_tomatoes()}\n\n')
    # print(f'Frame_Yield: {Frame_Yield}', sep='\n\n')

# for tomato in Tomatoes_Collection.get_collection():
    # print(f'\n\nId: {tomato.tomato_id}, \nCondfidence: {tomato.confidence}, \nClassification: {tomato.classification}, \nVolume: {tomato.volume}, \nAisle: {0}, \nPlant: {0}, \nTimestamp: {Tomatoes_Collection.collection_timestamp}\n\n')
    
