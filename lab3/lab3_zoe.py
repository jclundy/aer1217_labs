import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

##open photos
photo_folder = 'Lab3/output_folder'
photos = []
undistorted_photos = []
num_photos = len(os.listdir(photo_folder))

##open .csv file with pose and timestep
pose_file = 'Lab3/lab3_pose.csv'
pose_data = np.genfromtxt(pose_file, delimiter=',', skip_header=1)
#1-idx, 2-px, 3-py, 4-pz, 5-qw, 6-qx, 7-qy, 8-qz
# print(f"pose data: {pose_data[8:12]}") ##check data input is correct from csv file

##camera parameters from lab 3 document on quercus
K = np.matrix([[698.86, 0, 306.91],[0, 699.13, 150.34],[0, 0, 1]])
d = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0])
Tcb = np.matrix([[1, -1, 0, 0],[-1, 0, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])
Rcb = Tcb[:3, :3]

##landmark pixel coordinates
##must be within x=y= {-2, 2} [m] in body frame
landmark_px = []
landmark_world = []

##functions
## function to extract position and rotation matrix from .csv file
def extract_pose_from_csv_row(row):
    # 1. Extract Position
    px, py, pz = row[1], row[2], row[3]
    drone_pos = np.array([px, py, pz])
    
    # 2. Extract Quaternions and REORDER them to [x, y, z, w] for Scipy
    qw, qx, qy, qz = row[4], row[5], row[6], row[7]
    
    # 3. Create the Rotation Matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    R_body_to_world = rotation.as_matrix() # Returns a 3x3 numpy array

    return drone_pos, R_body_to_world

def transform_pixel_to_body_frame(pixel, drone_pos, Rbw):
    ##pixel to camera frame
    uv_vec = np.array([ [pixel[0]], [pixel[1]], [1.0] ])##homogenous pixel coordinates
    ray_c = np.linalg.inv(K) @ uv_vec ##ray direction in camera frame
   
    ##body to world
    rcw = Rbw @ Rcb
    ray_w = rcw @ ray_c
    ray_w_flat = ray_w.flatten() ##flatten to 1D array 

    ##calculate the scaling factor s to reach the ground (z=0) in world frame
    drone_z = drone_pos[2]
    ray_z = ray_w[2]
    #ray_z = ray_w_flat[2] ##alternative way to get z component of ray in world frame
    s = -drone_z / ray_z 

    ##scale vector to ground plane
    landmark_world_position = drone_pos + (s * ray_w_flat)

    ##flatten the landmark world position to get X, Y, Z coordinates
    landmark_world_position = np.asarray(landmark_world_position).flatten()# The result is a 3D point: [X, Y, Z]. Z should be very close to 0.
    return landmark_world_position

for i in range(num_photos):
    ##extrac csv instance
    Drone_pose, Rbw = extract_pose_from_csv_row(pose_data[i])

    #open photos and store in list
    photo_path = os.path.join(photo_folder, f'image_{i}.jpg')
    photo = cv2.imread(photo_path)

    ##apply camera distorion parameters and greyscale to photos
    photo = cv2.undistort(photo, K, d)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    photo = cv2.GaussianBlur(photo, (5, 5), 0)
    undistorted_photos.append(photo)
    # After Gaussian Blur, create a binary mask using calibrated threshold
    ret, thresh = cv2.threshold(photo, 120, 255, cv2.THRESH_BINARY)

    # Now find contours on the 'thresh' image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##find contours in the heirarchy (contour inside a contour) to identify the inner dot of the landmark
    if hierarchy is not None and len(contours) > 0:
        for j, contour in enumerate(contours):
            
            # hierarchy[0][i] contains an array: [Next, Previous, First_Child, Parent]
            # We want to look at the 'Parent' index (which is at index 3)
            parent_idx = hierarchy[0][j][3]
            
            # 4. If parent_idx is NOT -1, it means this contour is INSIDE another contour
            if parent_idx != -1:
                
                # 5. Filter out tiny specks of noise that might be inside the square
                if cv2.contourArea(contour) > 10: 
                    
                    # We found the inner dot! Calculate its circle.
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    landmark_px.append([i, center, radius]) ##store the frame index, center, and radius of target
                    
                    ##transform pixel coordinates to world frame coordinates
                    landmark_w = transform_pixel_to_body_frame(center, Drone_pose, Rbw)
                    landmark_world.append(landmark_w) #append to list of world coordinates

##save list of landmark world coordinates to .csv file
landmark_world_array = np.array(landmark_world)
np.savetxt('Lab3/landmark_world_coordinates.csv', landmark_world_array, delimiter=',', header='X,Y,Z', comments='')
