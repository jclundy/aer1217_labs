import numpy as np
from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################
from trajectory_generators import hardcoded_trajectory_generator
from optimize_trajectory import optimize_trajectory
try:
    import example_custom_utils as ecu
except ImportError:
    from . import example_custom_utils as ecu
#########################
# REPLACE THIS (END) ####
#########################


class Controller():

    def __init__(self, initial_obs, initial_info, use_firmware=False, buffer_size=100, verbose=False):
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self.ctrl = None if use_firmware else PIDController()
        if not use_firmware:
            self.KF = initial_info["quadrotor_kf"]
        self.reset()
        self.interEpisodeReset()
        t_scaled = self.planning(use_firmware, initial_info)
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

    def planning(self, use_firmware, initial_info):
        #########################
        # REPLACE THIS (START) ##
        #########################
        gate_sequence = [0, 1, 2, 3]
        gates = initial_info["nominal_gates_pos_and_type"]
        tall_h = initial_info["gate_dimensions"]["tall"]["height"]
        low_h  = initial_info["gate_dimensions"]["low"]["height"]
        start_z = tall_h if use_firmware else self.initial_obs[4]
        self.total_duration = 20

        obstacles = [(ob[0], ob[1], 0.55) for ob in self.NOMINAL_OBSTACLES]
        planner = ecu.RRTStar(obstacles, bounds=(-3.5, 3.5, -3.5, 3.5))

        waypoints = [[self.initial_obs[0], self.initial_obs[2], start_z]]
        current = [self.initial_obs[0], self.initial_obs[2]]

        # Build key targets: approach + gate center + exit for each gate, then final target
        targets = []
        for idx in gate_sequence:
            g = gates[idx]
            goal_z = tall_h if g[6] == 0 else low_h
            normal = np.array([np.sin(g[5]), np.cos(g[5])])
            if np.dot(normal, np.array([g[0], g[1]]) - np.array(current)) > 0:
                normal = -normal
            targets.append(([g[0] + 0.70*normal[0], g[1] + 0.70*normal[1]], goal_z))  # approach
            targets.append(([g[0], g[1]], goal_z))                                    # gate center
            targets.append(([g[0] - 0.70*normal[0], g[1] - 0.70*normal[1]], goal_z))  # exit
            current = [g[0], g[1]]
        t = initial_info["x_reference"]
        targets.append(([t[0], t[2]], t[4]))  # final target

        # For each target, use direct path or RRT* if blocked
        current = [self.initial_obs[0], self.initial_obs[2]]
        # for (goal, goal_z) in targets:
        #     if not planner._edge_free(ecu.Node(*current), ecu.Node(*goal)):
        #         print(f"RRT* triggered: {current} -> {goal}")
        #         path = planner.plan(current, goal)
        #         path[-1] = goal
        #         n = len(path)
        #         for i, pt in enumerate(path[1:], 1):
        #             z = waypoints[-1][2] + (goal_z - waypoints[-1][2]) * i / (n - 1)
        #             waypoints.append([pt[0], pt[1], z])
        #     else:
        #         if np.hypot(goal[0]-current[0], goal[1]-current[1]) > 2.0:
        #             waypoints.append([(current[0]+goal[0])/2, (current[1]+goal[1])/2, goal_z])
        #         waypoints.append([goal[0], goal[1], goal_z])
        #     current = goal
        # self.waypoints = np.array(waypoints)
        # np.savez("waypoints.npz", wps = self.waypoints)

        # ref_state = hardcoded_trajectory_generator(
        #     self.initial_obs, initial_info, self.CTRL_FREQ, self.total_duration,
        #     waypoints=self.waypoints
        # )

        npzfile = np.load("waypoints.npz")
        self.waypoints = npzfile["wps"]

        ref_time_state = optimize_trajectory(self.waypoints)
        ref_time = ref_time_state[:,0]
        ref_state = ref_time_state[:,1:]
        self.ref_x, self.ref_y, self.ref_z = ref_state[:,0], ref_state[:,1], ref_state[:,2]
        self.ref_vel, self.ref_acc         = ref_state[:,3:6], ref_state[:,6:9]
        self.ref_euler, self.ref_euler_rates = ref_state[:,9:12], ref_state[:,12:15]

        # return np.linspace(0, self.total_duration, len(ref_state))
        return ref_time
        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
        iteration = int(time * self.CTRL_FREQ)
        #########################
        # REPLACE THIS (START) ##
        #########################
        if iteration == 0:
            command_type, args = Command(2), [1, 2]  # takeoff

        elif iteration >= 3*self.CTRL_FREQ and iteration < (self.total_duration+3)*self.CTRL_FREQ:
            step = min(iteration - 3*self.CTRL_FREQ, len(self.ref_x)-1)
            command_type = Command(1)  # cmdFullState
            args = [np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]]),
                    self.ref_vel[step].flatten(),
                    self.ref_acc[step].flatten(),
                    self.ref_euler[step, 2],
                    self.ref_euler_rates[step]]

        elif iteration == (self.total_duration+3)*self.CTRL_FREQ:
            command_type, args = Command(6), []  # notify setpoint stop

        elif iteration == (self.total_duration+3)*self.CTRL_FREQ + 1:
            command_type = Command(5)  # goTo
            args = [[self.ref_x[-1], self.ref_y[-1], 1.0], 0., 2.5, False]

        elif iteration == (self.total_duration+6)*self.CTRL_FREQ:
            command_type, args = Command(3), [0., 3]  # land

        elif iteration == (self.total_duration+9)*self.CTRL_FREQ:
            command_type, args = Command(4), []  # stop

        else:
            command_type, args = Command(0), []
        #########################
        # REPLACE THIS (END) ####
        #########################
        return command_type, args

    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")
        iteration = int(time * self.CTRL_FREQ)
        step = min(iteration, len(self.ref_x)-1)
        return np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]]), np.zeros(3)

    def reset(self):
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer    = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer   = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer   = deque([], maxlen=self.BUFFER_SIZE)
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0