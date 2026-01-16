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

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

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
    
        ## generate waypoints for planning
        max_climb = 2 * 0.25 #m/s
        max_forward = 3.5 * 0.1#m/s
        max_descent = 0.1 #m/s
        climb_height = 1
        radius = 1
        circle_center = [0,-3,1]

        waypoints = []


        wp0 = (-1,-3,0)
        wp1 = (-1,-3,climb_height)
        waypoints.append(wp0)
        waypoints.append(wp1)
        waypoints.append((0, -4, 1))
        waypoints.append((1,-3,1))
        waypoints.append((0,-2,1))
        waypoints.append(wp1)
        waypoints.append(wp0)

        # Polynomial fit.
        self.waypoints = np.array(waypoints)

        climb_duration = (climb_height)/max_climb
        curve1_z = np.linspace(0,climb_height, int(climb_duration*self.CTRL_FREQ))
        curve1_x = np.ones((len(curve1_z),)) * wp0[0]
        curve1_y = np.ones((len(curve1_z),)) * wp0[1]

        curve1_zd = max_climb * np.ones((len(curve1_z),))
        curve1_xd = 0 * np.ones((len(curve1_z),))
        curve1_yd = 0 * np.ones((len(curve1_z),))

        circumference = 2*np.pi*radius
        circle_duration = circumference / max_forward
        curve2_th = np.linspace(np.pi, -np.pi, int(circle_duration*self.CTRL_FREQ))
        curve2_x = np.cos(curve2_th) * radius + circle_center[0]
        curve2_y = np.sin(curve2_th) * radius + circle_center[1]
        curve2_z = np.ones((len(curve2_th)),) * climb_height

        curve2_zd = 0 * np.ones((len(curve2_th),))
        curve2_xd = max_forward * np.ones((len(curve2_th),))
        curve2_yd = 0 * np.ones((len(curve2_th),))

        descent_duration = climb_height / max_descent
        curve3_z = np.linspace(climb_height,0, int(descent_duration*self.CTRL_FREQ))
        curve3_x = np.ones((len(curve3_z),)) * wp0[0]
        curve3_y = np.ones((len(curve3_z),)) * wp0[1]

        curve3_zd = -max_descent * np.ones((len(curve3_z),))
        curve3_xd = 0 * np.ones((len(curve3_z),))
        curve3_yd = 0 * np.ones((len(curve3_z),))

        rx = np.concatenate((curve1_x, curve2_x, curve3_x),axis=0)
        ry = np.concatenate((curve1_y, curve2_y, curve3_y),axis=0)
        rz = np.concatenate((curve1_z, curve2_z, curve3_z),axis=0)


        rxd = np.concatenate((curve1_xd, curve2_xd, curve3_xd),axis=0)
        ryd = np.concatenate((curve1_yd, curve2_yd, curve3_yd),axis=0)
        rzd = np.concatenate((curve1_zd, curve2_zd, curve3_zd),axis=0)

        # total_duration = climb_duration +  circle_duration + descent_duration
        t_scaled = np.linspace(0, len(rx)/self.CTRL_FREQ, len(rx))
        yaw = np.zeros((len(rx),))
        yaw_start = int(climb_duration*self.CTRL_FREQ)
        yaw_end = yaw_start + int(circle_duration*self.CTRL_FREQ)
        print("yaw len", len(yaw))
        print("th len", len(curve2_th))
        # yaw[yaw_start:yaw_end] = th + np.pi / 2
        print("yaw_start - yaw_end", yaw_end - yaw_start)
        print(len(yaw[yaw_start:yaw_end]))
        yaw[yaw_start:yaw_end] = curve2_th + np.pi / 2
        
        self.ref_x = rx
        self.ref_y = ry
        self.ref_z = rz

        self.ref_xd = rxd
        self.ref_yd = ryd
        self.ref_zd = rzd
        self.ref_yaw = yaw

        return t_scaled

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        # if iteration >= len(self.ref_x):
        #     command_type = Command(0)  # None.
        #     args = []
        # else :
        #     x = self.ref_x[iteration]
        #     y = self.ref_y[iteration]
        #     z = self.ref_z[iteration]
        #     yaw = 0.
        #     duration = 1/self.CTRL_FREQ

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        command_type = Command(0)  # None.
        args = []

        # if iteration == 0:
        #     height = 1
        #     duration = 2

        #     command_type = Command(2)  # Take-off.
        #     args = [height, duration]
        # elif iteration == 2 * self.CTRL_FREQ:
        #     x = -1 #self.ref_x[-1]
        #     y = -3 #self.ref_y[-1]
        #     z = 1 
        #     yaw = 0.
        #     duration = 2.5

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]
        # elif iteration == 4.5 * self.CTRL_FREQ:
        #     height = 0.
        #     duration = 3

        #     command_type = Command(3)  # Land.
        #     args = [height, duration]
        # elif iteration == 7.5 * self.CTRL_FREQ:
        #     command_type = Command(4) # STOP command
        #     args = []

        # elif iteration == 8 * self.CTRL_FREQ:
        #     command_type = Command(0) # None
        #     args = []

        if iteration < len(self.ref_x):
            target_pos = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
            target_vel = np.array([self.ref_xd[iteration], self.ref_yd[iteration], self.ref_zd[iteration]])
            target_acc = np.zeros(3)
            target_yaw = self.ref_yaw[iteration]
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration == len(self.ref_x):
            command_type = Command(4) # STOP command
            args = []
        # else:
        #     command_type = Command(0)  # None.
        #     args = []


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
