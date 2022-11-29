import tensorflow as tf
import argparse
import os
import glob
# import darknet
import tomato_recognition.yolov4.compiled_darknet.darknet.darknet as darknet
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
import math
import os


# Steps (1) - (3)

def image_detection(image_path, network, class_names, class_colors, thresh, show_bboxes=False):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image_with_boxes = darknet.draw_boxes(detections, image_resized, class_colors)
    
    # Take special note that if show_bbox is True, the output will be a 5-tuple isntead of a 3-Tuple
    if show_bboxes:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), width, height
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, width, height

def addObjects(detections, label, confidence_threshold):
    # Predictions element format is (Label, Confidence, Bbox Coords)
    # Identifiers are used to give the objects a unique id
    identifiers = {'Tomato':'t', 'Apple':'a', 'Plant':'p'}
    identifier = identifiers[label]
    obj_id = ''
    count = 0
    
    # Create an Objects dictionary to populate with object IDs and identifying traits
    Objects = {}
    
    for detection in detections:
        if detection[0] == label:
            if confidence_threshold <= float(detection[1]):
                # Make an id for an object
                count += 1
                obj_id = f'{identifier}{count}'
            
                # Convert yolo_bbox coords to point coords
                # bbox_points is of format (xmin, ymin, xmax, ymax)
                bbox_points = darknet.bbox2points(detection[2])
                Objects[obj_id] = [obj_id, label, detection[1], bbox_points]
            
    # Return dictionary of element format {obj_id: [obj_id, label, confidence, bbox_points]}
    return Objects

def getClass(model, image, classes):
    # Run image through model to get prediction probabilities
    prediction = model.predict(image)[0]
    
    # Get index of largest probability
    max_confidence_index = np.argmax(prediction)
    # Get class of largest probability
    predicted_class = classes[max_confidence_index]
    
    return predicted_class


# def process_defectiveness(model, tom_container, defect_classes):
    # #########################
    # # code to convert container if needed
    # #########################
    
    # # Run image subsets through the defectivness model to get classification
    # predictions = model.predict(tom_container, verbose=0)
    
## Helper Functions
def displayImage(image):
    # Used to display the imported image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def displaySubregions(subregions_list):
    # Used to display the subregions from the imported image
    for elem in subregions_list:
        tomato = elem[0]
        subregion = elem[1]
        plt.imshow(subregion[0]) # subregion is of shape (1, 256, 256, 3)
        plt.axis('off')
        plt.title(f"id: {tomato}")
        plt.show()