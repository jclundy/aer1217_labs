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
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Gate order for the run — change to whatever sequence is announced on demo day
GATE_ORDER = [4, 3, 2, 1, 2, 3]

#########################
# REPLACE THIS (END) ####
#########################


def interpolate_path(path, points_per_metre=10):
    """Densify a waypoint list so draw_trajectory always has enough points.

    Inserts linearly interpolated points between each consecutive pair so that
    there is approximately `points_per_metre` samples per metre of path length.
    Minimum 2 points between any two waypoints so very short segments are fine.
    """
    dense = [path[0]]
    for i in range(len(path) - 1):
        a = np.array(path[i],   dtype=float)
        b = np.array(path[i+1], dtype=float)
        dist = np.linalg.norm(b - a)
        n = max(2, int(dist * points_per_metre))
        for k in range(1, n + 1):
            dense.append(a + (b - a) * k / n)
    return dense


class Controller():
    """Template controller class."""

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ     = initial_info["ctrl_freq"]
        self.initial_obs   = initial_obs
        self.VERBOSE       = verbose
        self.BUFFER_SIZE   = buffer_size

        self.NOMINAL_GATES     = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        if use_firmware:
            self.ctrl = None
        else:
            self.ctrl = PIDController()
            self.KF   = initial_info["quadrotor_kf"]

        self.reset()
        self.interEpisodeReset()

        t_scaled = self.planning(use_firmware, initial_info)

        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


    def planning(self, use_firmware, initial_info):
        """Trajectory planning using RRT* through gate sequence."""

        #########################
        # REPLACE THIS (START) ##
        #########################

        BOUNDS   = np.array([[-3.5, 3.5], [-3.5, 3.5], [0.10, 1.95]])
        Z_BOUNDS = (BOUNDS[2, 0], BOUNDS[2, 1])

        # Start position
        if use_firmware:
            start_pos = np.array([self.initial_obs[0],
                                  self.initial_obs[2],
                                  self.initial_obs[4]])
        else:
            start_pos = np.array([self.initial_obs[0],
                                  self.initial_obs[2],
                                  self.initial_obs[4]])

        goal_pos = np.array([-0.5, 2.0, 1.0])

        # Build ordered list of key positions: start → gate centres → goal
        key_positions = [start_pos]
        for gid in GATE_ORDER:
            key_positions.append(np.array(self.NOMINAL_GATES[gid - 1][:3], dtype=float))
        key_positions.append(goal_pos)

        # Expand each gate into approach / centre / departure sub-points
        buffer_pts = [key_positions[0]]
        for i, gid in enumerate(GATE_ORDER):
            gate_raw = self.NOMINAL_GATES[gid - 1]
            prev_pos = buffer_pts[-1]
            next_pos = key_positions[i + 2]
            approach, centre, departure = gate_via_points(
                gate_raw, prev_pos, next_pos, buf=0.40, z_bounds=Z_BOUNDS)
            buffer_pts.append(approach)
            buffer_pts.append(centre)
            buffer_pts.append(departure)
        buffer_pts.append(goal_pos)

        # Map segment index to which gate (0-based) is being targeted.
        # Layout per gate: [approach, centre, departure] → indices 0,1,2
        # Only centre segments (index%3==1) pass through the gate opening.
        def gate_id_for_segment(seg_idx):
            gate_block = seg_idx // 3
            seg_within = seg_idx % 3
            if seg_within == 1 and gate_block < len(GATE_ORDER):
                return GATE_ORDER[gate_block] - 1   # 0-based
            return None

        # Skip RRT* for very short segments (approach→centre distance is ~buf=0.4 m
        # but departure is also ~0.4 m; straight line is collision-free there).
        SHORT_SEG_THRESH = 0.50   # metres

        seed = 42
        sparse_path = [buffer_pts[0]]

        for seg_idx in range(len(buffer_pts) - 1):
            a = np.array(buffer_pts[seg_idx],   dtype=float)
            b = np.array(buffer_pts[seg_idx+1], dtype=float)
            dist = np.linalg.norm(b - a)

            # Very short segment — straight line is fine
            if dist < SHORT_SEG_THRESH:
                sparse_path.append(b)
                continue

            PAD = 0.8
            lo  = np.minimum(a, b) - PAD
            hi  = np.maximum(a, b) + PAD
            seg_bounds = np.array([
                [max(lo[0], BOUNDS[0, 0]), min(hi[0], BOUNDS[0, 1])],
                [max(lo[1], BOUNDS[1, 0]), min(hi[1], BOUNDS[1, 1])],
                [max(lo[2], BOUNDS[2, 0]), min(hi[2], BOUNDS[2, 1])],
            ])

            planner = RRTStar(
                start     = a,
                goal      = b,
                obstacles = self.NOMINAL_OBSTACLES,
                gates     = self.NOMINAL_GATES,
                bounds    = seg_bounds,
                max_it    = 500,
                step_size = 0.4,
                goal_r    = 0.15,
                rewire_r  = 0.6,
                rng       = np.random.default_rng(seed),
            )
            seg_path = planner.plan(target_gate_idx=gate_id_for_segment(seg_idx))

            if seg_path is None or len(seg_path) == 0:
                print(f"[WARN] RRT* failed for segment {seg_idx} (len={dist:.2f}m), using straight line.")
                sparse_path.append(b)
            else:
                sparse_path.extend(seg_path[1:])

        print(f"Sparse path: {len(sparse_path)} waypoints")

        # Densify so draw_trajectory never gets a zero step argument
        dense_path = interpolate_path(sparse_path, points_per_metre=10)
        print(f"Planning complete — total waypoints (dense): {len(dense_path)}")

        self.waypoints = np.array(dense_path)
        self.ref_x = self.waypoints[:, 0]
        self.ref_y = self.waypoints[:, 1]
        self.ref_z = self.waypoints[:, 2]
        t_scaled   = np.arange(len(dense_path), dtype=float)

        #########################
        # REPLACE THIS (END) ####
        #########################

        return t_scaled


    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):

        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration == 0:
            command_type = Command(2)
            args = [1, 2]   # height=1, duration=2

        elif iteration >= 3 * self.CTRL_FREQ and iteration < 20 * self.CTRL_FREQ:
            step = min(iteration - 3 * self.CTRL_FREQ, len(self.ref_x) - 1)
            target_pos       = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel       = np.zeros(3)
            target_acc       = np.zeros(3)
            target_yaw       = 0.
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == 20 * self.CTRL_FREQ:
            command_type = Command(6)
            args = []

        elif iteration == 20 * self.CTRL_FREQ + 1:
            command_type = Command(5)
            args = [[self.ref_x[-1], self.ref_y[-1], 1.5], 0., 2.5, False]

        elif iteration == 23 * self.CTRL_FREQ:
            command_type = Command(5)
            args = [[self.initial_obs[0], self.initial_obs[2], 1.5], 0., 6, False]

        elif iteration == 30 * self.CTRL_FREQ:
            command_type = Command(3)
            args = [0., 3]

        elif iteration == 33 * self.CTRL_FREQ - 1:
            command_type = Command(4)
            args = []

        else:
            command_type = Command(0)
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args


    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):

        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time * self.CTRL_FREQ)

        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])

        return target_p, np.zeros(3)


    def reset(self):
        self.action_buffer        = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer           = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer        = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer          = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer          = deque([], maxlen=self.BUFFER_SIZE)
        self.interstep_counter    = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        self.interstep_learning_time        = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time     = 0
