import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# from combined_discretized_casadi_solver import *
from fixed_number_of_subections_discretized_solver import *


npzfile = np.load("test_states.npz")
states = npzfile["states"]
waypoints =  npzfile["waypoints"]

print("states.shape", states.shape)

solve_dt = 0.1
ax0 = plt.figure().add_subplot(projection='3d')

# dts = np.ones_like(A4_vals) * solve_dt
wx = waypoints[:,0]
wy = waypoints[:,1]
wz = waypoints[:,2]

ax0.scatter(wx, wy, wz, marker='^')
# plot_times = np.ones_like(A0_vals) * plot_dt

x_vals = states[:,0]
y_vals = states[:,1]
z_vals = states[:,2]

print("states.shape", states.shape)
# print(x_vals)
# print(y_vals)
# print(z_vals)

ax0.scatter(x_vals,y_vals,z_vals)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")


plt.show()