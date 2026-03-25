FILES
lab4.py            - Main entry point. Runs the VO loop and plots results.
stereo_vo_base.py  - Core VO class: RANSAC, triangulation, SVD pose estimation.
ground_truth_pose.mat - RTK GPS ground truth from the KITTI dataset.

DEPENDENCIES
pip install numpy opencv-contrib-python scipy matplotlib

DATASET SETUP
Place the KITTI city sequence data at:
/CityData/2011_09_26/2011_09_26_drive_0005_sync/image_00/data/
/CityData/2011_09_26/2011_09_26_drive_0005_sync/image_01/data/
Place ground_truth_pose.mat in the working directory.

HOW TO RUN
python lab4.py
Outputs: video.avi, VO_T.npy, a 3D trajectory plot, and RMSE printed to console.