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

from example_custom_utils import gate_normal, gate_via_points, RRTStar

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

##GATE ORDER

GATE_ORDER = [1, 2, 3, 4]

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
        ## generate waypoints for planning

        self.waypoints = []
        BOUNDS   = [[-3.5, 3.5], [-3.5, 3.5], [0.10, 1.95]]
        Z_BOUNDS = (BOUNDS[2][0], BOUNDS[2][1])

        ###start position    
        if use_firmware:
            start_pos = np.array([self.initial_obs[0], self.initial_obs[2],
                           initial_info["gate_dimensions"]["tall"]["height"]]) ##need to check this from initial info
        else:
            start_pos = np.array([self.initial_obs[0], self.initial_obs[2], self.initial_obs[4]])
        
        ##final hover 
        goal_pose = np.array([ -0.5, 2.0, 1.0])
        
        pts = [start_pos]
        for gid in GATE_ORDER:
            pts.append(np.array([self.NOMINAL_GATES[gid-1][:3], gid]))
            x, y, z = self.NOMINAL_GATES[gid-1][:3]
            pts.append(np.array([x,y,z,gid]))
        pts.append(goal_pose)

        buffer_pts = [pts[0]]
        ###add entry and exit buffer to gate
        for i in range(1, len(pts)-1):
            approach, centre, departure = gate_via_points(self.NOMINAL_GATES[i-1], buffer_pts[-1], pts[i+1], buf=0.40, z_bounds=Z_BOUNDS)
            buffer_pts.append(approach)
            buffer_pts.append(centre)
            buffer_pts.append(departure)
        buffer_pts.append(pts[-1])

        #set seed
        seed = 42
        full_path = [buffer_pts[0]] #add start point to full path

        ##run RRT* for each segment between waypoints
        for segid in range(len(buffer_pts)-1):
            a = buffer_pts[segid]
            b = buffer_pts[segid+1]

            #pad bounding to speed up (***)
            PAD    = 0.8
            lo     = np.minimum(a, b) - PAD
            hi     = np.maximum(a, b) + PAD
            bounds = [
                [max(lo[0], BOUNDS[0][0]), min(hi[0], BOUNDS[0][1])],
                [max(lo[1], BOUNDS[1][0]), min(hi[1], BOUNDS[1][1])],
                [max(lo[2], BOUNDS[2][0]), min(hi[2], BOUNDS[2][1])],
            ]
            ##RRT* call
            planner = RRTStar(a, b, obstacles=self.NOMINAL_OBSTACLES, bounds=bounds, rng=np.random.default_rng(seed))
            # seg_path = planner.plan()
            seg_path = planner.plan(gate_id=GATE_ORDER[segid] if segid < len(GATE_ORDER) else None) ##make sure gate being passed
            
            ##add to path
            full_path.extend(seg_path[1:])

        print(f"Planned path, total waypoints {len(full_path)} ")


        ##append waypoints and return time scalling for plotting
        self.waypoints = np.array(full_path)
        self.ref_x = self.waypoints[:, 0]
        self.ref_y = self.waypoints[:, 1]
        self.ref_z = self.waypoints[:, 2]
        t_scaled = np.arange(len(full_path), dtype=float)

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

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == 20*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

       # [INSTRUCTIONS] Example code for using goTo interface 
        elif iteration == 20*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5 
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 23*self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
            z = 1.5
            yaw = 0.
            duration = 6

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 30*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == 33*self.CTRL_FREQ-1:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

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
