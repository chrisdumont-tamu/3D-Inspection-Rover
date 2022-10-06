import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


# Configure depth stream and color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Counter to save frames
frames_counter = 0
frames_saved = 0


save_img_location = r'C:\Users\felvi\ecen403programs\saved_images'
os.chdir(save_img_location)

# Start streaming
pipeline.start(config)


try:
    while True:
    
        frames_counter += 1;
        
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_data = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if (frames_counter == 30):
        
            frames_counter = 0
            
        
            # Save image
            cv2.imwrite(f'color_frame_{frames_saved}.jpg', color_image)
            
            # Log numpy depth data
            np.save(f'depth_data_{frames_saved}.npy', depth_data)
            
            frames_saved += 1
        
        if (frames_saved == 3):
            break
        
            
finally:

    # Stop streaming
    pipeline.stop()