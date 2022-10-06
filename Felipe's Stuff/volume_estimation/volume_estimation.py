import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2)
colorizer = rs.colorizer()


while True:
    # Grab camera data

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # depth_frame = decimate.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Depth and color images
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_frame.get_data())

    # depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    # pc.map_to(mapped_frame)

    # Point Cloud vertices to array
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    
    t = points.get_texture_coordinates()
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    print('w:\t', w)
    print('h:\t', h)
    print('v:\t',np.shape(verts))
    print('t:\t',np.shape(texcoords))
    break
# Stop streaming
pipeline.stop()