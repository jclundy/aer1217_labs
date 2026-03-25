"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor

    def triangulate(self, fc):
        # Camera intrinsics and baseline distance between left and right cameras
        fx, fy, cu, cv_p, b = self.cam.fx, self.cam.fy, self.cam.cu, self.cam.cv, self.cam.baseline

        def unproject(ul, vl, ur):
            # Disparity d = horizontal pixel difference between left and right images
            # Depth Z recovered from the stereo equation: Z = fx * baseline / d
            # X, Y recovered by inverting the pinhole projection: X = (u - cu) * Z / fx
            Z = fx * b / (ul - ur)
            return np.column_stack([(ul - cu) * Z / fx, (vl - cv_p) * Z / fy, Z])

        # pts_a: 3D point cloud from previous frame, pts_b: from current frame
        return unproject(fc[:,0], fc[:,1], fc[:,2]), unproject(fc[:,4], fc[:,5], fc[:,6])

    def svd_alignment(self, pts_a, pts_b):
        # Inverse-depth-squared weights: nearby points (small Z) are triangulated
        # more accurately so they contribute more to the alignment (eq. 2)
        w  = 1.0 / (pts_a[:,2]**2 + 1e-6)
        w /= w.sum()                          # normalise weights to sum to 1

        # Weighted centroids of each point cloud (eq. 2)
        mu_a, mu_b = w @ pts_a, w @ pts_b

        # Cross-covariance matrix W between the two centred point clouds (eq. 3)
        W = ((pts_b - mu_b) * w[:,None]).T @ (pts_a - mu_a)

        # SVD decomposition of W (eq. 4) — note: numpy gives W = U S Vt
        U, _, Vt = np.linalg.svd(W)

        # Rotation C_ba: det correction ensures proper rotation (no reflection) (eq. 5)
        C = U @ np.diag([1., 1., np.linalg.det(Vt.T) * np.linalg.det(U)]) @ Vt
        # Translation r: displacement of centroid after applying rotation (eq. 5)
        r = mu_b - C @ mu_a
        return C, r

    def ransac(self, pts_a, pts_b, fc, n_iter=500, thresh=0.03):
        N = len(pts_a)
        best_mask, best_count = np.ones(N, dtype=bool), 0

        for _ in range(n_iter):
            # Randomly sample 3 point pairs — minimum needed to fit a rigid-body model
            idx = np.random.choice(N, 3, replace=False)
            try:
                C_c, r_c = self.svd_alignment(pts_a[idx], pts_b[idx])
            except np.linalg.LinAlgError:
                continue

            # Apply candidate model to all points and compute depth-normalised residual
            # Dividing by Z makes the threshold scale-invariant across the depth range
            pred = (C_c @ pts_a.T).T + r_c
            err  = np.linalg.norm(pts_b - pred, axis=1) / np.maximum(pts_a[:,2], 1.0)

            # Points within threshold are inliers — consistent with the candidate motion
            mask = err < thresh
            if mask.sum() > best_count:
                best_count, best_mask = mask.sum(), mask

        if best_count < 3:
            best_mask = np.ones(N, dtype=bool)

        # Final re-estimation using all inliers (not just the 3-point sample)
        C, r = self.svd_alignment(pts_a[best_mask], pts_b[best_mask])
        # Return inlier right-image pixel coords for visualisation on right frame
        return C, r, fc[best_mask, 2:4], fc[best_mask, 6:8]

    def pose_estimation(self, features_coor):
        # dummy C and r
        C = np.eye(3)
        r = np.array([0,0,0])
        # feature in right img (without filtering)
        f_r_prev, f_r_cur = features_coor[:,2:4], features_coor[:,6:8]
        # ------------- start your code here -------------- #

        # 1: inverse stereo camera model — pixel matches -> 3D point clouds
        pts_a, pts_b = self.triangulate(features_coor)

        # 2: RANSAC outlier rejection + 3: SVD point cloud alignment
        C, r_b, f_r_prev, f_r_cur = self.ransac(pts_a, pts_b, features_coor)

        # r_b is expressed in the current frame b; rotate into previous frame a
        r = C.T @ r_b

        # If estimate exceeds 2m, reuse previous frame to avoid corrupting trajectory
        if np.linalg.norm(r) > 2.0:
            C, r = self.C.copy(), self.r.flatten().copy()

        # replace (1) the dummy C and r to the estimated C and r. 
        #         (2) the original features to the filtered features
        return C, r, f_r_prev, f_r_cur
    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 
