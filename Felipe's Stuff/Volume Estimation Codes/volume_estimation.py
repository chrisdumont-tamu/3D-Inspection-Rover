import pyrealsense2 as rs
from math import pi
import numpy as np
import cv2

import tomato_localization_toolset as tlt


## Constants
# Path to tomato locator model
YOLO_MODEL_PATH = r'.\ml_models_testing\yolov4-tiny-416-run1.tflite'
# YOLO_MODEL_PATH = r'.\ml_models_testing\yolov4-full-608.tflite'
# Yolo model input size
YOLO_INPUT_SIZE = 416#608#416
# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Localization thresholds
IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.25

# Depth Threshold
DEPTH_THRESHOLD = 0.085

## Variables
xmin = 0
ymin = 0
xmax = 1
ymax = 1


######################################## Functions

def get_camera_intrinsics(depth_frame, color_frame):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    
    return depth_intrin

def get_center_pixel_coords(xmin, ymin, xmax, ymax):
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)
    return (x_center, y_center)

def pixel_to_point(intrinsics, depth_frame, x, y):
    # Need intrinsics, [x, y] pixel, and depth to use 
    # rs.rs2_deproject_pixel_to_point(...)
    
    # Calculate depth of pixel
    depth = depth_frame.get_distance(x, y)
    
    # Calculate point in 3D
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    
    return point # point is a 3-tuple float of shape (1, 3)

def check_point_validity(point):
    return point != [0, 0, 0]

def check_depth_threshold(depth_threshold, point, center_point):
    return depth_threshold >= abs(point[2] - center_point[2])

def get_valid_horizontal_edge_point(intrinsics, depth_frame, depth_threshold, xmin, y_center, x_center, center_point, return_valid_pixel_coords=False):
    # This function calculates a valid point in the image
    
    while True:
        # If xmin reaches x_cetner, return a non-valid point
        point = pixel_to_point(intrinsics, depth_frame, xmin, y_center)
        valid_point = check_point_validity(point)
        valid_threshold = check_depth_threshold(depth_threshold, point, center_point)
        print(f'point: {point}')
        print(f'pixel: {(xmin, y_center)}')
        print(f'valid_point: {valid_point}')
        print(f'valid_threshold: {valid_threshold}\n')
        if xmin == x_center:
            return (0, 0, 0)
        
        elif (valid_point and valid_threshold):
            if return_valid_pixel_coords:
                return point, (xmin, y_center)
            return point
        
        else:
            xmin += 1

def get_valid_vertical_edge_point(intrinsics, depth_frame, depth_threshold, x_center, ymin, y_center, center_point, return_valid_pixel_coords=False):
    # This function calculates a valid point in the image
    
    while True:
        # If xmin reaches x_cetner, return a non-valid point
        point = pixel_to_point(intrinsics, depth_frame, x_center, ymin)
        valid_point = check_point_validity(point)
        valid_threshold = check_depth_threshold(depth_threshold, point, center_point)
        print(f'point: {point}')
        print(f'pixel: {(x_center, ymin)}')
        print(f'valid_point: {valid_point}')
        print(f'valid_threshold: {valid_threshold}\n')
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
        
    x_semi_axis = center_point[0] - horizontal_point[0]
    y_semi_axis = center_point[1] - vertical_point[1]
    z_semi_axis = x_semi_axis # right now the assumption is that the tomato is symmetrical about the y axis, so x and z semi-axes are equal
    
    return x_semi_axis, y_semi_axis, z_semi_axis

def calc_ellipsoide_volume(x_semi_axis, y_semi_axis, z_semi_axis):
    return (4 / 3) * pi * x_semi_axis * y_semi_axis * z_semi_axis

######################################## Helper Functions

def point_meters_to_inches(point):
    return [39.3701 * i for i in point]

def show_depth_image(depth_frame):
     # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    cv2.imshow('image', depth_colormap)
    cv2.waitKey(0)

########################################

## Camera setup
# Create a pipeline object to own and handle realsnese camera
pipeline = rs.pipeline()

# Configure the streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Begin streaming
pipeline.start(config)

# Perform color to depth alignment
align_to = rs.stream.color
align = rs.align(align_to)

#######################################################

for i in range(15):
    frames = pipeline.wait_for_frames()

# Aligned the depth frame to the color frame
aligned_frames = align.process(frames)

# Depth frame
depth_frame = aligned_frames.get_depth_frame()

# Color frame
color_frame = aligned_frames.get_color_frame()

# Intrinsics
depth_intrin = get_camera_intrinsics(depth_frame, color_frame)

# Get color image
color_image = np.asanyarray(color_frame.get_data())
tlt.show_image(color_image, 'color image')
## Get point at center of tomato
# Get tomatoes collection
Tomatoes_Collection = tlt.process_extract_tomatoes(YOLO_INPUT_SIZE, YOLO_MODEL_PATH, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, img=color_image, load_image=False)

# Find volume if tomato detected
if Tomatoes_Collection.get_number_of_tomatoes() > 0:
    # Get tomato
    tomato = Tomatoes_Collection.get_tomato(index=0)

    # Get bounding box info from yolo model
    ymin, xmin, ymax, xmax = tomato.get_denormalized_bbox(FRAME_WIDTH, FRAME_HEIGHT) #ymin, xmin, ymax, xmax

    # Get center pixel coordinates of bbox/tomato (x, y)
    pix_center = get_center_pixel_coords(xmin, ymin, xmax, ymax)

    # Get center point coordinates of tomato
    point_center = pixel_to_point(depth_intrin, depth_frame, pix_center[0], pix_center[1])

    print(f'\n\nymin, xmin, ymax, xmax: {ymin}, {xmin}, {ymax} ,{xmax}\n\n')
    print(f'\n\ncenter pixel = {pix_center}\n\n')
    print(f'\n\ncenter point = {point_center}\n\n')
    tlt.show_point_and_bbox_image(Tomatoes_Collection, color_image, FRAME_WIDTH, FRAME_HEIGHT, 0, pix_center[0], pix_center[1], 2)
    show_depth_image(depth_frame)
    
    ## Test to see if get_valid_horizontal_edge_point works
    point_horizontal, pix_horizontal = get_valid_horizontal_edge_point(depth_intrin, depth_frame, DEPTH_THRESHOLD, xmin, pix_center[1], pix_center[0], point_center, return_valid_pixel_coords=True)
    # point_horizontal = get_valid_horizontal_edge_point(depth_intrin, depth_frame, DEPTH_THRESHOLD, xmin, pix_center[1], pix_center[0], point_center)
    
    print(f'\n\nhorizontal pixel: {pix_horizontal}\n\n')
    print(f'\n\nhorizontal point: {point_horizontal}\n\n')
    tlt.show_point_and_bbox_image(Tomatoes_Collection, color_image, FRAME_WIDTH, FRAME_HEIGHT, 0, pix_horizontal[0], pix_horizontal[1], 2)
    
    ## Test to see if get_valid_vertical_edge_point works
    point_vertical, pix_vertical = get_valid_vertical_edge_point(depth_intrin, depth_frame, DEPTH_THRESHOLD, pix_center[0], ymin, pix_center[1], point_center, return_valid_pixel_coords=True)
    
    print(f'\n\nvertical pixel: {pix_vertical}\n\n')
    print(f'\n\nvertical point: {point_vertical}\n\n')
    tlt.show_point_and_bbox_image(Tomatoes_Collection, color_image, FRAME_WIDTH, FRAME_HEIGHT, 0, pix_vertical[0], pix_vertical[1], 2)
    
    
    print(f'\nPoints:')
    print(f'Center Point: {point_center}')
    print(f'Horizontal Point: {point_horizontal}')
    print(f'Vertical Point: {point_vertical}\n')
    
    ## Test semi-axes calculation function
    semi_axes = get_semi_axes(point_center, point_horizontal, point_vertical)
    print(f'\n\nsemi-axes in meters: {semi_axes}\n\n')
    print(f'\n\nsemi-axes in inches: {point_meters_to_inches(semi_axes)}\n\n')
    
    ## Test volume calculation of ellipsoid model
    ellipsoid_volume = calc_ellipsoide_volume(semi_axes[0], semi_axes[1], semi_axes[2])
    print(f'\n\nVolume of Ellipsoid in meters cubed: {ellipsoid_volume}\n\n')
    print(f'\n\nVolume of Ellipsoid in liters: {ellipsoid_volume * 1000}\n\n')

else:
    print('No tomatoes found')