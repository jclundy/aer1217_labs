# import rosbag
import numpy as np

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


def compute_rmse(path1, path2, num_points=200, dim=3):
    """
    dim = 3 → use x,y,z
    dim = 2 → use x,y only
    """
    p1 = resample_path(path1, num_points)
    p2 = resample_path(path2, num_points)

    if dim == 2:
        p1 = p1[:, :2]
        p2 = p2[:, :2]
    elif dim != 3:
        raise ValueError("dim must be 2 or 3")

    diff = p1 - p2
    mse = np.mean(np.sum(diff**2, axis=1))
    return np.sqrt(mse)


if __name__ == "__main__":
    bag_file = "Group_2_trial_3_2026-04-22-18-05-54.bag"

    topic_pose = "/cf9/pose"
    topic_cmd  = "/cf9/cmd_full_state"

    path_pose = extract_positions(bag_file, topic_pose)
    path_cmd  = extract_positions(bag_file, topic_cmd)

    print(f"Loaded {len(path_pose)} pose points")
    print(f"Loaded {len(path_cmd)} cmd points")

    rmse_3d = compute_rmse(path_pose, path_cmd, dim=3)
    rmse_2d = compute_rmse(path_pose, path_cmd, dim=2)

    print("RMSE (3D):", rmse_3d)
    print("RMSE (2D, ignoring z):", rmse_2d)