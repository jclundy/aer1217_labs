import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

from scipy.spatial.transform import Rotation as Rot

"""
inputs - folder of images
output1 - csv of detections
    - img name
    - target i x, target i y
alternatively, a python data struct

output2 - folder of annotated images
 """


folder = "data/image_folder"

def initial_detection(current_img, filter):
    convolved = cv.filter2D(current_img/255.0, -1, circle_template, borderType=cv.BORDER_DEFAULT)
    normalized = (convolved - np.min(convolved.flatten())) / (np.max(convolved) - np.min(convolved))
    normalized = (normalized*255).astype(np.uint8)
    ret,filtered_thresholded = cv.threshold(normalized,200,255,cv.THRESH_BINARY)

    im = filtered_thresholded/255.0 * normalized
    image_max = ndi.maximum_filter(im, size=20, mode='constant')
    coordinates = peak_local_max(im, min_distance=20)
    return coordinates

def refine_detection(current_img, initial_x, initial_y, initial_radius=24, radius_delta=10, position_delta=20):
    
    min_radius = initial_radius - radius_delta
    max_radius = initial_radius + radius_delta
    min_x = initial_x - position_delta
    max_x = initial_x + position_delta + 1
    
    min_y = initial_y - position_delta
    max_y = initial_y + position_delta + 1
    
    min_error = np.inf
    best_parameters = (0,0,0)
    
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            for r in range(min_radius, max_radius):
                w = r * 2 + 1
                if x <= 0:
                    continue
                if x >= current_img.shape[1] - w:
                    continue
                if y <= 0:
                    continue
                if y >= current_img.shape[0] - w:
                    continue
                blank_new = np.ones((w,w), np.uint8)
                circle_template = cv.circle(blank_new,(r,r),r, 0, -1)
                img_slice = current_img[y:y+w, x:x+w]
                error = np.sum(np.abs(img_slice - circle_template))
    
                if(error < min_error):
                    min_error = error
                    best_parameters = (x,y,r)
    return best_parameters

def detect_all_targets(current_img):
    # create filter template
    blank_new = np.zeros((75,75), np.uint8)
    circle_template_0 = cv.circle(blank_new,(37,37),np.round(75/2).astype(np.uint8) , 1, -1)
    circle_template = cv.circle(circle_template_0,(37,37),25 , 0, -1)

    centers = initial_detection(current_img, circle_template)

    results = [] #x,y,r
    for coord in centers:
        initial_x = coord[1]
        initial_y = coord[0]
        best_parameters = refine_detection(current_img, initial_x, initial_y, initial_radius)
        results.append(best_parameters)
    
    return results

def compute_target_inertial_pose(pose_quad, pixels, camera_mtx, projection_mtx):
    # pose_quad - px, py, pz, qw, qx, qy, qz 
    (u,v) = pixels
    fy = camera_mtx[1,1]
    cy = camera_mtx[1,2]

    # Inertial Frame
    vehicle_position = np.array([pose_quad[0], pose_quad[1], pose_quad[2]]).reshape(3,1)
    vehicle_orientation_quat = np.array(pose_quad[3:7])
    r_obj = Rot.from_quat(vehicle_orientation_quat, scalar_first=True)
    # QUESTION! is this rotation from the body frame to inertial, or inertial to body frame?
    R = r_obj.as_matrix()
    # r.apply()
    zhat = [0,0,-1]
    Pw_hat = R @ zhat
    # 'Camera's World frame'
    cos_beta = np.dot(Pw_hat, zhat)
    Zq = vehicle_position[2] # pz

    beta = np.arccos(cos_beta)

    fx = camera_mtx[0,0]
    cx = camera_mtx[0,2]

    alpha1 = np.arctan2(u-cx, fx)
    alpha2 = np.arctan2(v-cy,fy)

    v1_hat = np.array([np.cos(alpha1), np.sin(alpha1)]).reshape(2,)
    v2_hat = np.array([np.cos(alpha2), np.sin(alpha2)]).reshape(2,)

    w1 = np.array([v-cy, 0, fy])
    w2 = np.array([0, u-cx, fx])
    w_hat = w1 + w2
    w_hat = w_hat / np.linalg.norm(w_hat)
    w_hat = w_hat.reshape(3,1)

    cos_alpha = np.dot(v1_hat, v2_hat)
    alpha= np.arccos(cos_alpha)

    L1 = Zq / np.cos(alpha + beta)
    Zw = L1 * np.cos(alpha)

    Yw = (u-cx)/fx * Zw
    Xw = (v-cy)/fy * Zw

    Pw = np.array([Xw,Yw, Zw])

    Pq = vehicle_position
    Pe =  Pq - R @ Pw
    # Pe2 = Pq - R @ w_hat * L1

    return Pe, Pw, alpha, beta