import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

##landmark pixel coordinates
##must be within x=y= {-2, 2} [m] in body frame
landmark_px = []
landmark_body = []

for i in range(num_photos):
    #open photos and store in list
    photo_path = os.path.join(photo_folder, f'image_{i}.jpg')
    photo = cv2.imread(photo_path)
    photos.append(photo)

    ##apply camera distorion parameters and greyscale to photos
    photo = cv2.undistort(photo, K, d)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    photo = cv2.GaussianBlur(photo, (5, 5), 0)
    undistorted_photos.append(photo)
    # After Gaussian Blur, create a binary mask
    # Adjust 200 based on your lighting; pixels above 200 become white (255)
    ret, thresh = cv2.threshold(photo, 120, 255, cv2.THRESH_BINARY)

    # Now find contours on the 'thresh' image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##find contours in the heirarchy (contour inside a contour) to identify the inner dot of the landmark
    if hierarchy is not None and len(contours) > 0:
        for i, contour in enumerate(contours):
            
            # hierarchy[0][i] contains an array: [Next, Previous, First_Child, Parent]
            # We want to look at the 'Parent' index (which is at index 3)
            parent_idx = hierarchy[0][i][3]
            
            # 4. If parent_idx is NOT -1, it means this contour is INSIDE another contour
            if parent_idx != -1:
                
                # 5. Filter out tiny specks of noise that might be inside the square
                if cv2.contourArea(contour) > 10: 
                    
                    # We found the inner dot! Calculate its circle.
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)

##algorithm to identify pixels


##transform pixels from camera frame to body frame

##match body frame to .csv file with pose and timestep
##solvePnP as possible openCV function
##assign location in body frame

##