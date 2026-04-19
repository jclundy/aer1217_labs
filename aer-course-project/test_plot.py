import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from combined_discretized_casadi_solver import *


npzfile = np.load("combined_coefficients.npz")
A4_vals = npzfile["A4"]
B4_vals = npzfile["B4"]
C4_vals = npzfile["C4"]
dts = npzfile["times"]


solve_dt = 0.1

# dts = np.ones_like(A4_vals) * solve_dt

waypoints =  npzfile["waypoints"]
Nw = waypoints.shape[0]


B0, B1, B2, B3, B4 = unroll_coefficients(np.array(B4_vals),waypoints[0,1],dts)
A0, A1, A2, A3, A4 = unroll_coefficients(np.array(A4_vals),waypoints[0,0],dts)
C0, C1, C2, C3, C4 = unroll_coefficients(np.array(C4_vals),waypoints[0,2],dts)

ax0 = plt.figure().add_subplot(projection='3d')

wx = waypoints[:,0]
wy = waypoints[:,1]
wz = waypoints[:,2]

ax0.scatter(wx, wy, wz, marker='o')

# plot trajectory of quadrotor evalutaed at every timestep
N_segments = A0.shape[0]
plot_freq = 60.0
plot_dt = 1/plot_freq
plot_times = np.ones_like(A0) * solve_dt

# N_plot = np.round(waypoint_start_times[-1] / plot_dt).astype(np.int64) - 1
states = evalute_polynomials_over_control_time_step(plot_times, plot_dt, N_segments, plot_freq, 
                                                                        A0, A1, A2, A3, A4_vals, 
                                                                        B0, B1, B2, B3, B4_vals, 
                                                                        C0, C1, C2, C3, C4_vals)

x_array = states[:,0]
y_array = states[:,1]
z_array = states[:,2]

ax0.plot(x_array,y_array,z_array)
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")

plt.show()