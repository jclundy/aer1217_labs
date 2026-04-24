import numpy as np

sym_data = np.loadtxt("simulation_error.csv", delimiter=',')
data = sym_data.reshape(-1,4)

# first column is iteration
x_e = data[:,1]
y_e = data[:,2]
z_e = data[:,3]

N = data.shape[0]

diff = np.array([x_e, y_e, z_e])

mse3d = np.mean(np.sum(diff**2, axis=1))
rmse3d = np.sqrt(mse3d)

print("N=", N)
print("rmse3d = {:.4f}".format(rmse3d))

diff2d = np.array([x_e, y_e])
mse2d = np.mean(np.sum(diff2d**2, axis=1))
rmse2d = np.sqrt(mse2d)
print("rmse2d = {:.4f}".format(rmse2d))

