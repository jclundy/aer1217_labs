# import rosbag
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

def extract_positions(bag_path, topic):
    positions = []
    bagpath = Path(bag_path)
    
    # with rosbag.Bag(bag_path, 'r') as bag:
    #     for _, msg, _ in bag.read_messages(topics=[topic]):
    #         try:
    #             p = msg.pose.position
    #             positions.append([p.x, p.y, p.z])
    #         except AttributeError:
    #             continue
    
    with AnyReader([bagpath]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                # print(msg.header.frame_id)
                p = msg.pose.position
                positions.append([p.x, p.y, p.z])
            except AttributeError:
                continue
    return np.array(positions)

def resample_path(path, num_points):
    path = np.array(path)
    
    if len(path) == 0:
        return np.zeros((num_points, 3))

    distances = np.zeros(len(path))
    distances[1:] = np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1)
    )

    if distances[-1] == 0:
        return np.repeat(path[0:1], num_points, axis=0)

    distances /= distances[-1]
    new_distances = np.linspace(0, 1, num_points)

    resampled = np.zeros((num_points, 3))
    for i in range(3):
        resampled[:, i] = np.interp(new_distances, distances, path[:, i])

    return resampled


def compute_rmse(path1, path2, num_points=200):
    p1 = resample_path(path1, num_points)
    p2 = resample_path(path2, num_points)

    diff = p1 - p2
    mse = np.mean(np.sum(diff**2, axis=1))
    return np.sqrt(mse), p1, p2


def plot_trajectories(path1, path2):
    fig = plt.figure(figsize=(12, 6))

    # Left plot: path1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(path1[:, 0], path1[:, 1], path1[:, 2])
    ax1.set_title("Trajectory: /cf9/pose")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Right plot: path2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(path2[:, 0], path2[:, 1], path2[:, 2])
    ax2.set_title("Trajectory: /cf9/cmd_full_state")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.show()

def plot_trajectories_overlay(path1, path2):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], label='cf9/pose')
    ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], linestyle='--', label='cf9/cmd_full_state')

    ax.set_title("Trajectory Comparison (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_trajectories_overlay_2d(path1, path2):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111) # , projection='3d')

    ax.plot(path1[:, 0], path1[:, 1], label='cf9/pose') # path1[:, 2], label='cf9/pose')
    ax.plot(path2[:, 0], path2[:, 1], linestyle='--', label='cf9/cmd_full_state') # path2[:, 2], linestyle='--', label='cf9/cmd_full_state')

    ax.set_title("Trajectory Comparison (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bag_file = "Group_2_trial_3_2026-04-22-18-05-54.bag"

    topic_pose = "/cf9/pose"
    topic_cmd  = "/cf9/cmd_full_state"

    path_pose = extract_positions(bag_file, topic_pose)
    path_cmd  = extract_positions(bag_file, topic_cmd)

    print(f"Loaded {len(path_pose)} pose points")
    print(f"Loaded {len(path_cmd)} cmd points")

    rmse, p1_resampled, p2_resampled = compute_rmse(path_pose, path_cmd)

    print("RMSE:", rmse)

    plot_trajectories_overlay(p1_resampled, p2_resampled)