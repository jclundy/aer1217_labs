"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np
from collections import deque

from example_custom_utils import path, generate_trajectory

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

# from piecewise_trajectory_solver import generate_trajectory
# from combined_discretized_casadi_solver import generate_trajectory
import os
#########################
# REPLACE THIS (START) ##
#########################

##GATE ORDER

GATE_ORDER = [1, 3, 4, 2, 1, 4]

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""
        #########################
        # REPLACE THIS (START) ##
        #########################
        use_interpolation = True

        def interpolate_path(path, points_per_metre=10):
            dense = [path[0]]
            for i in range(len(path) - 1):
                a = np.array(path[i],   dtype=float)
                b = np.array(path[i+1], dtype=float)
                dist = np.linalg.norm(b - a)
                n = max(2, int(dist * points_per_metre))
                for k in range(1, n + 1):
                    dense.append(a + (b - a) * k / n)
            return dense
        
        def remove_duplicates(path):
            trimed = []
            for i in range(0, len(path) - 1):
                a = np.array(path[i],   dtype=float)
                b = np.array(path[i+1], dtype=float)
                
                if np.linalg.norm(a - b) <= 1e-6:
                    print("duplicate found at index ", i)
                    continue
                else:
                    trimed.append(a)
            trimed.append(path[-1])
            return trimed

        # ref_state = hardcoded_trajectory_generator(
        #     self.initial_obs, initial_info, self.CTRL_FREQ, self.total_duration,
        #     waypoints=self.waypoints
        # )
        discretization_dt = 0.1
        averageSpeed = 0.4 #20 cm /s

        ref_state = None
        save_file = "test_states.npz" # "trajectory_states.npz"
        recompute_trajectory = False
        if(os.path.exists(save_file) and not recompute_trajectory):
            print("loading saved trajectory")
            npzfile = np.load(save_file)
            ref_state = npzfile["ref_state"]
            total_time = npzfile["total_time"]
            self.waypoints = npzfile["waypoints"]
        else:
            print("planning path")

            full_path, keypts = path(GATE_ORDER)
            if(use_interpolation):
                trimed_full_path = remove_duplicates(full_path)
                dense_path = interpolate_path(trimed_full_path, points_per_metre=5)
                trimed_full_dense_path = remove_duplicates(full_path)
                self.waypoints = np.array(trimed_full_dense_path)
            else:
                trimed_full_path = remove_duplicates(full_path)
                self.waypoints = np.array(trimed_full_path)
            print("generating minimum-snap trajectory")

            waypointsPerGroup = 5
            ref_state, total_time = generate_trajectory(self.waypoints, averageSpeed, discretization_dt, self.CTRL_FREQ, waypointsPerGroup)
            # ref_state, total_time = generate_trajectory(self.waypoints, averageSpeed, discretization_dt, self.CTRL_FREQ)
            np.savez(save_file, ref_state=ref_state, total_time=total_time, waypoints=self.waypoints)

        self.total_duration = total_time

        print("ref_state.shape", ref_state.shape)

        self.ref_x = ref_state[:, 0]
        self.ref_y = ref_state[:, 1]
        self.ref_z = ref_state[:, 2]

        ##for traj planner
        self.ref_x, self.ref_y, self.ref_z = ref_state[:,0], ref_state[:,1], ref_state[:,2]
        self.ref_vel, self.ref_acc         = ref_state[:,3:6], ref_state[:,6:9]
        self.ref_euler, self.ref_euler_rates = ref_state[:,9:12], ref_state[:,12:15]

        t_scaled = np.linspace(0, total_time, ref_state[:,0].size)
        # t_scaled = np.arange(ref_state[:,0].size) * 


        #########################
        # REPLACE THIS (END) ####
        #########################

        return t_scaled

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):

        if self.ctrl is not None:
                        raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
        iteration = int(time * self.CTRL_FREQ)
        #########################
        # REPLACE THIS (START) ##
        #########################
        landDuration = 4
        takeOffTime  = 4

        initial_stop_iteration = (self.total_duration+takeOffTime)*self.CTRL_FREQ
        stop_iteration = initial_stop_iteration + 2*self.CTRL_FREQ

        land_iteration = stop_iteration + 1 * self.CTRL_FREQ
        end_iteration =  land_iteration +  landDuration*self.CTRL_FREQ


        if iteration == 0:
            command_type, args = Command(2), [1, takeOffTime]  # takeoff

        elif iteration >= takeOffTime*self.CTRL_FREQ and iteration < initial_stop_iteration:
            step = min(iteration - takeOffTime*self.CTRL_FREQ, len(self.ref_x)-1)
            command_type = Command(1)  # cmdFullState
            # print("sending command full state")
            # print("step=",step)
            # print("step=",step)
            # print("ref pos", self.ref_x[step],self.ref_z[step],self.ref_y[step])
            # print("ref vel", self.ref_vel[step].flatten())
            # print("ref acc", self.ref_acc[step].flatten())
            args = [np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]]),
                    self.ref_vel[step].flatten(),
                    self.ref_acc[step].flatten(),
                    self.ref_euler[step, 2],
                    self.ref_euler_rates[step]]
            xe = self.ref_x[step] - obs[0]
            ye = self.ref_y[step] - obs[2]
            ze = self.ref_z[step] - obs[4]
            print(step, xe, ye, ze)
        elif iteration >= initial_stop_iteration and iteration < stop_iteration:
            command_type = Command(1)  # cmdFullState
            print("sending command full state, zero derivatives")
            args = [np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]]),
                    np.zeros((3,)),
                    np.zeros((3,)),
                    0.0,
                    np.zeros((3,))]
        elif iteration >= stop_iteration and iteration < land_iteration:
            command_type, args = Command(6), []  # notify setpoint stop
            print("sending setpoint stop")

        elif iteration >= land_iteration and iteration < land_iteration + 1:
            print("sending land command")
            command_type, args = Command(3), [0., landDuration]  # land
        # elif iteration >= end_iteration:
        #     print("sending exit command")
        #     command_type, args = Command(4), []  # exit
        else:
            command_type, args = Command(0), []
        #########################
        # REPLACE THIS (END) ####
        #########################
        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
