"""
AER1217 Lab 3 - Georeferencing Using UAV payload Data

for each image captured by the drone, detect circular targets
and compute their (x, y) position on the ground in the Vicon frame.

  1. For each image we'll find circular tagets and get their centroids
  2. For each centroid pixel, back-project it to a ray in 3D space
  3. Intersect that ray with the ground plane (z = 0) to get world coords
  4. Collect all world coord detections across all images, then cluster
     them into 6 targets and average each cluster for a final position
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans


K = np.array([[698.86, 0, 306.91],
              [0, 699.13, 150.34],
              [0, 0,      1.0   ]])
dist = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0.0])

# Rotation part of T_CB ( n lab handout)
# T_CB is the 4x4 transform from body frame to camera frame
# R_CB is just top left 3x3 blk of that matrix
# We use R_CB.T to go camera -> body
R_CB = np.array([[0, -1,  0],
                 [-1,  0,  0],
                 [0,  0, -1]], dtype=float)

IMAGE_DIR  = '/Users/capcom/Documents/aer1217/lab3/images'
POSE_CSV   = '/Users/capcom/Documents/aer1217/lab3/lab3_pose.csv'
OUTPUT_DIR = '/Users/capcom/Documents/aer1217/lab3/output'

def pixel_to_world(u, v, pose):
    """
    Step 1 — Undo K and lens distortion
    Step 2 — Rotate the ray from camera frame to inertial (Vicon) frame
    Step 3 — Intersect the ray with the ground plane z = 0
    """

    r_c = np.append(
        cv2.undistortPoints(np.array([[[u, v]]], dtype=float), K, dist)[0][0],
        1.0   # append z=1 to form a direction vector [xn, yn, 1]
    )

    R_IB = Rotation.from_quat([pose.q_x, pose.q_y, pose.q_z, pose.q_w]).as_matrix()
    r_i  = R_IB @ R_CB.T @ r_c   # camera -> body -> inertial

    if abs(r_i[2]) < 1e-6 or -pose.p_z / r_i[2] < 0:
        return None
    t = -pose.p_z / r_i[2]                   # how far along the ray to the ground
    return pose.p_x + t * r_i[0], pose.p_y + t * r_i[1]   # (wx, wy) in metres

def detect_circles(img_path):
    """
    Params mostly through trial and error
    """
    img  = cv2.undistort(cv2.imread(str(img_path)), K, dist)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=30, param1=50, param2=30,
                               minRadius=8, maxRadius=60)
    if circles is None:
        return []

    # Return just the (x, y) centres, I don't want the radius
    return [(x, y) for x, y, _ in circles[0]]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
ann_dir = Path(OUTPUT_DIR) / 'annotated_images'
ann_dir.mkdir(exist_ok=True)

poses = pd.read_csv(POSE_CSV)

# Sort images by frame number so idx lines up with pose rows
images = sorted(Path(IMAGE_DIR).glob('image_*.jpg'),
                key=lambda p: int(p.stem.split('_')[1]))

# MAIN
world_points = []   # accumulates all (wx, wy) detections across all frames

for img_path in images:
    idx  = int(img_path.stem.split('_')[1])
    if idx >= len(poses):
        continue

    pose = poses.iloc[idx]

    # skip frames where the drone is on or near the ground
    if pose.p_z < 0.3:
        continue

    img = cv2.undistort(cv2.imread(str(img_path)), K, dist)
    centroids = detect_circles(img_path)

    for (cx, cy) in centroids:
        # back-project this pixel to a world coordinate
        result = pixel_to_world(cx, cy, pose)
        if result is None:
            continue
        wx, wy = result

        # Discard anything outside expected target area
        if not (-2.5 < wx < 2.5 and -2.5 < wy < 2.5):
            continue

        # save world coordinate for clustering later
        world_points.append([wx, wy])

        cv2.circle(img, (int(cx), int(cy)), 18, (0, 255, 0), 2)
        cv2.putText(img, f'({wx:.2f},{wy:.2f})m', (int(cx)+8, int(cy)-8),
                    cv2.ARIAL, 0.4, (255, 255, 0), 1)

    cv2.imwrite(str(ann_dir / f'ann_{img_path.name}'), img)
    print(f'{img_path.name}: {len(centroids)} circle(s) detected')

pts = np.array(world_points)
targets = KMeans(n_clusters=6, n_init=10).fit(pts).cluster_centers_

print('  TARGET POS (Vicon frame)')
for i, t in enumerate(targets):
    print(f'  Target {i+1}:  x={t[0]:+.3f},  y={t[1]:+.3f}  m')

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pts[:,0], pts[:,1], s=15, alpha=0.4, label='Individual detections')
for i, t in enumerate(targets):
    ax.scatter(*t, s=200, marker='*', zorder=5, label=f'T{i+1} ({t[0]:.2f},{t[1]:.2f})')
ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel='Vicon X (m)',
       ylabel='Vicon Y (m)', title='Lab 3 — Est Target Positions', aspect='equal')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/target_map.png', dpi=150)
plt.close()
print(f'Outputs saved to {OUTPUT_DIR}')
