use_discretized = True
if use_discretized:
    from fixed_number_of_subections_discretized_solver import *
else:
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
    waypoints = all_waypoints[0:3,:]
   
    # total_time = 30.3

    average_speed = 0.5
    ctrl_freq = 60

    if use_discretized:
        numSubsections = 51  
        states, total_duration = generate_trajectory(waypoints, average_speed, numSubsections, ctrl_freq)
    else:
        discretization_dt = 1/30.0
        states, total_duration = generate_trajectory(waypoints, average_speed, discretization_dt, ctrl_freq)

    ax0 = plt.figure().add_subplot(projection='3d')

    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    ax0.scatter(wx, wy, wz, marker='o')
    # plot_times = np.ones_like(A0_vals) * plot_dt

    print("states.shape",states.shape)
    print("total duration", total_duration)

    np.savez("test_states.npz", 
             states=states,
             waypoints=waypoints)
    
    x_vals = states[:,0]
    y_vals = states[:,1]
    z_vals = states[:,2]
    
    
    ax0.plot(x_vals,y_vals,z_vals)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    # plt.show()
    plt.savefig("smoothed_trajectory.png")



if __name__ == "__main__":
    main()