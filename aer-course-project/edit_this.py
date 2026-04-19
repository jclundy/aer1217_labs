import numpy as np
from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################
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
        gate_sequence = [3, 0, 1, 2]  
        FLIGHT_Z      = 1.0           # fixed fligjt height
        APPROACH_DIST = 0.75          # offset before/after gate center along gate normal
        N_INTERP      = 4             # nmbr of interpolated points between each key point
        SPEED         = 0.35          # constant mission speed (m/s)
        OBS_RADIUS    = 0.75          # safety radius around cylindrical obstacles
        GATE_RADIUS   = 0.35          # safety radius around gate frames

        gates = initial_info["nominal_gates_pos_and_type"]

        # build obstacle list, optionally excluding the target gate
        def get_obstacles(exclude_gate_idx=None):
            obs = [(ob[0], ob[1], OBS_RADIUS) for ob in self.NOMINAL_OBSTACLES]
            for i, g in enumerate(gates):
                if i != exclude_gate_idx:
                    obs.append((g[0], g[1], GATE_RADIUS))
            return obs

        # inearly interpolate N_INTERP points from p0 to p1
        def interp(p0, p1):
            return [[p0[0] + (p1[0]-p0[0])*i/N_INTERP,
                     p0[1] + (p1[1]-p0[1])*i/N_INTERP,
                     FLIGHT_Z] for i in range(1, N_INTERP+1)]

        # return list of [x,y] nodes from p0 to p1,
        #         using RRT* only if direct path is blocked
        def connect(p0, p1, exclude_gate_idx=None):
            obstacles = get_obstacles(exclude_gate_idx)
            planner   = ecu.RRTStar(obstacles, bounds=(-3.5, 3.5, -3.5, 3.5))
            if planner._edge_free(ecu.Node(*p0), ecu.Node(*p1)):
                return [p1]
            path = planner.plan(p0, p1)
            path[-1] = p1
            return path[1:]

        # build ordered list of 3D waypoints
        waypoints = [[self.initial_obs[0], self.initial_obs[2], FLIGHT_Z]]
        current   = [self.initial_obs[0], self.initial_obs[2]]

        for idx in gate_sequence:
            g = gates[idx]
            cx, cy = g[0], g[1]
            normal = np.array([np.sin(g[5]), np.cos(g[5])])
            if np.dot(normal, np.array([cx, cy]) - np.array(current)) > 0:
                normal = -normal

            approach = [cx + APPROACH_DIST*normal[0], cy + APPROACH_DIST*normal[1]]
            center   = [cx, cy]
            exit_pt  = [cx - APPROACH_DIST*normal[0], cy - APPROACH_DIST*normal[1]]

            # current -> approach: exclude current gate from obstacles
            nodes = connect(current, approach, exclude_gate_idx=idx)
            prev  = current
            for node in nodes:
                for pt in interp(prev, node):
                    waypoints.append(pt)
                prev = node

            # approach -> center -> exit: always direct, current gate excluded
            for pt in interp(approach, center):
                waypoints.append(pt)
            for pt in interp(center, exit_pt):
                waypoints.append(pt)

            current = exit_pt

        # exit of last gate -> final target: all gates are obstacles
        target = [initial_info["x_reference"][0], initial_info["x_reference"][2]]
        nodes  = connect(current, target, exclude_gate_idx=None)
        prev   = current
        for node in nodes:
            for pt in interp(prev, node):
                waypoints.append(pt)
            prev = node

        waypoints = np.array(waypoints)  # (N, 3)
        self.waypoints = waypoints

        # convert waypoints to dense reference at CTRL_FREQ
        # timing is purely distance / sped
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        times = np.concatenate([[0.0], np.cumsum(dists / SPEED)])
        self.total_duration = times[-1]

        t_dense = np.linspace(0, self.total_duration, int(self.total_duration * self.CTRL_FREQ))
        self.ref_x = np.interp(t_dense, times, waypoints[:, 0])
        self.ref_y = np.interp(t_dense, times, waypoints[:, 1])
        self.ref_z = np.interp(t_dense, times, waypoints[:, 2])

        dt = 1.0 / self.CTRL_FREQ
        self.ref_vel = np.column_stack([
            np.gradient(self.ref_x, dt),
            np.gradient(self.ref_y, dt),
            np.gradient(self.ref_z, dt)
        ])

        return t_dense
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

        elif iteration >= 3*self.CTRL_FREQ and iteration < int((self.total_duration+3)*self.CTRL_FREQ):
            step = min(iteration - 3*self.CTRL_FREQ, len(self.ref_x)-1)
            command_type = Command(1)  # cmdFullState
            args = [np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]]),
                    self.ref_vel[step].flatten(),
                    np.zeros(3),
                    0.0,
                    np.zeros(3)]

        elif iteration == int((self.total_duration+3)*self.CTRL_FREQ):
            command_type, args = Command(6), []  # notify setpoint stop

        elif iteration == int((self.total_duration+3)*self.CTRL_FREQ) + 1:
            command_type = Command(5)  # goTo final position
            args = [[self.ref_x[-1], self.ref_y[-1], 1.0], 0., 2.5, False]

        elif iteration == int((self.total_duration+6)*self.CTRL_FREQ):
            command_type, args = Command(3), [0., 3]  # land

        elif iteration == int((self.total_duration+9)*self.CTRL_FREQ):
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