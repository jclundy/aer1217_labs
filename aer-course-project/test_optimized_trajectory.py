from combined_discretized_casadi_solver import *

def generate_waypoints():
    poses = [[-1.,-3.,1.],
        [-0.574,-2.748,0.986],
        [0.05,-2.5,1.],
        [0.5,-2.5,1.],
        [0.95,-2.5,1.],
        [1.085,-1.914,1.029],
        [1.11,-1.738,1.035],
        [1.184,-1.053,1.],
        [0.941,-0.617,1.005],
        [0.781,0.029,1.004],
        [0.45,0.5,1.],
        [0.,0.5,1.],
        [-0.45,0.5,1.],
        [-0.5,1.5,1.],
        [-0.5,1.95,1.],
        [-0.233,1.853,0.973],
        [-0.057,1.555,0.94,],
        [-0.211,1.027,0.941],
        [-0.496,0.595,0.964],
        [-0.311,0.167,0.986],
        [-0.225,-0.275,1.],
        [-0.224,-0.678,1.004],
        [-0.161,-1.371,1.021],
        [-0.149,-2.021,1.006],
        [0.05,-2.5,1.],
        [0.5,-2.5,1.],
        [0.95,-2.5,1.],
        [1.113,-1.965,0.999],
        [1.175,-1.42,1.],
        [1.17,-0.898,0.989],
        [0.942,-0.372,0.98,],
        [0.712,0.146,0.983],
        [0.45,0.5,1.],
        [0.,0.5,1.],
        [-0.45,0.5,1.],
        [-0.344,0.233,1.048],
        [-0.089,0.063,1.029],
        [0.303,-0.089,1.032],
        [0.775,-0.275,1.],
        [1.14,-0.468,1.006],
        [1.534,-0.655,0.988],
        [2.,-1.05,1.],
        [2.,-1.5,1.],
        [2.,-1.95,1.],
        [1.641,-1.89,1.009],
        [1.275,-1.396,0.981],
        [1.188,-1.011,0.985],
        [1.055,-0.603,0.989],
        [0.75,0.025,1.],
        [0.491,0.562,1.],
        [0.222,1.098,0.98,],
        [0.038,1.624,0.996],
        [-0.5,2.,1.]]

    return np.array(poses).reshape(-1,3)

def main():
    all_waypoints = generate_waypoints()
    waypoints = all_waypoints
    print("waypoints.shape=", waypoints.shape)


    Nw = waypoints.shape[0]
    max_time = 30.0

    p_prev = waypoints[0:Nw-1,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    print("waypoint_min_lengths=",waypoint_min_lengths.reshape(1,-1))

    total_length = np.sum(waypoint_min_lengths)
    segment_durations = waypoint_min_lengths / total_length * max_time

    maxSpeed = 2 * total_length / max_time


    dt = 0.1

    waypoint_desired_velocities = np.zeros(waypoints.shape)
    waypoint_desired_velocities[1:Nw-2,:] = np.inf

    waypoint_desired_accelerations = np.zeros(waypoints.shape)
    waypoint_desired_accelerations[1:Nw-2,:] = np.inf

    waypoint_desired_jerks = np.zeros(waypoints.shape)
    waypoint_desired_jerks[1:Nw-2,:] = np.inf

    waypoint_start_times = np.zeros((Nw,))
    for i in range(0,segment_durations.size):
        waypoint_start_times[i+1] = waypoint_start_times[i] + segment_durations[i]

    print("waypoint_start_times", waypoint_start_times)
    ax0 = plt.figure().add_subplot(projection='3d')

    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    ax0.scatter(wx, wy, wz, marker='o')

    desired_derivatives = np.zeros((Nw, 3, 3))
    desired_derivatives[1:Nw-2,:,:] = np.inf
    desired_derivatives[11,:,:] = 0

    solver = SegmentCasadiSolver(waypoints,desired_derivatives, waypoint_start_times, dt, maxSpeed)

    n = solver.N
    a3_start = 0
    a3_end = a3_start + n
    b3_start = a3_end
    b3_end = b3_start + n
    c3_start = b3_end
    c3_end = c3_start + n 

    num_decision_variables = 3 * n

    X0 = np.zeros((num_decision_variables,))
    X0[a3_start:a3_end] = 0 
    # B3 lower bound
    X0[b3_start:b3_end] = 0
    # C3 lower bound
    X0[c3_start:c3_end] = 0
    
    A4, B4, C4, solution = solver.solve_coefficients(X0)

    A0, A1, A2, A3, A4 = unroll_coefficients(np.array(A4),waypoints[0,0],solver.dts)
    B0, B1, B2, B3, B4 = unroll_coefficients(np.array(B4),waypoints[0,1],solver.dts)
    C0, C1, C2, C3, C4 = unroll_coefficients(np.array(C4),waypoints[0,2],solver.dts)

    tn = solver.dts[n-1]
    xn, xdn, xddn, xdddn = compute_state_1d(A0[n-1],A1[n-1],A2[n-1],A3[n-1],A4[n-1], tn)
    yn, ydn, yddn, ydddn = compute_state_1d(B0[n-1],B1[n-1],B2[n-1],B3[n-1],B4[n-1], tn)
    zn, zdn, zddn, zdddn = compute_state_1d(C0[n-1],C1[n-1],C2[n-1],C3[n-1],C4[n-1], tn)

    # ax0.scatter(A0_vals, B0_vals, C0_vals, marker='^')

    # plot trajectory of quadrotor evalutaed at every timestep

    plot_freq = 60.0
    plot_dt = 1/plot_freq
    # plot_times = np.ones_like(A0_vals) * plot_dt
    np.savez("combined_coefficients.npz", 
             A4=A4.reshape(-1,1), 
             B4=B4.reshape(-1,1), 
             C4=C4.reshape(-1,1),
             times=solver.dts,
             waypoints=waypoints)

    # N_sections = A0.shape[0]
    x_array, y_array, z_array = evalute_polynomials_over_control_time_step(segment_durations, plot_dt, Nw-1, plot_freq, 
                                                                           A0, A1, A2, A3, A4, 
                                                                           B0, B1, B2, B3, B4, 
                                                                           C0, C1, C2, C3, C4)

    ax0.plot(A0[0:-1], B0[0:-1], C0[0:-1])

    
    midpoint_idx = np.floor(waypoint_start_times[1:-1] /  max_time * A0.shape[0])

    print("A0.shape", A0.shape)
    print("sum(solver.dts)=", np.sum(solver.dts))

    A0_vals = np.concatenate([A0[0],A0[midpoint_idx], A0[-1]])
    B0_vals = np.concatenate([B0[0],B0[midpoint_idx], B0[-1]])
    C0_vals = np.concatenate([C0[0],C0[midpoint_idx], C0[-1]])

    print("A0_vals", A0_vals.reshape(1,-1))
    print("B0_vals", B0_vals.reshape(1,-1))
    print("C0_vals", C0_vals.reshape(1,-1))

    ax0.scatter(A0_vals, B0_vals, C0_vals, marker="^")
    # ax0.plot(x_array,y_array,z_array)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    # plt.show()
    plt.savefig("smoothed_trajectory.png")



if __name__ == "__main__":
    main()