import numpy as np

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

    ax.set_zlim(0,1)

    ax.legend()
    plt.tight_layout()
    plt.show()

sym_data = np.loadtxt("simulation_data.csv", delimiter=' ')
data = sym_data.reshape(-1,7)
print(data[0])
print(data[1])

# first column is iteration
x = data[:,1]
y = data[:,2]
z = data[:,3] + 1

rx = data[:,4]
ry = data[:,5]
rz = data[:,6]

print("x.shape", x.shape)
print("y.shape", y.shape)
print("z.shape", z.shape)

N = data.shape[0]

ref = np.hstack([rx, ry, rz]).reshape((3,-1)).T
pose = np.hstack([x,y,z]).reshape((3,-1)).T

print(ref.shape)
print(pose.shape)

diff = ref - pose

print(diff.shape)

mse3d = np.mean(np.sum(diff**2, axis=1))
rmse3d = np.sqrt(mse3d)

print("N=", N)
print("rmse3d = {:.4f}".format(rmse3d))

diff2d = ref[:,0:2] - pose[:,0:2]

mse2d = np.mean(np.sum(diff2d**2, axis=1))
rmse2d = np.sqrt(mse2d)
print("rmse2d = {:.4f}".format(rmse2d))

plot_trajectories_overlay(ref, pose)
