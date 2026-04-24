import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##constants
GATES = [  # x, y, z, r, p, y, type 
      [ 0.5, -2.5, 1.0, 0, 0, -1.57, 0],      # gate 1
      [ 2.0, -1.5, 1.0, 0, 0, 0,     0],      # gate 2
      [ 0.0,  0.5, 1.0, 0, 0, 1.57,  0],      # gate 3
      [-0.5,  1.5, 1.0, 0, 0, 0,     0]       # gate 4
    ]
OBSTACLES =[  # x, y, z, r, p, y
      [ 1.5, -2.5, 1.3, 0.35, 0, 0],             # obstacle 1
      [ 0.5, -1.0, 1.3, 0.35, 0, 0],             # obstacle 2
      [ 1.5,    0, 1.3, 0.35, 0, 0],             # obstacle 3
      [-1.0,    0, 1.3, 0.35, 0, 0]              # obstacle 4  
    ]
##start and end points
START = [-1.0, -3.0, 1.0]  # start
GOAL = [-0.5,  2.0, 1.0]  # goal
##bounds
BOUNDS   = np.array([[-3.5, 3.5], [-3.5, 3.5], [0.10, 1.95]])
Z_BOUNDS = (BOUNDS[2, 0], BOUNDS[2, 1])
PADDING = 0.5 #for sample selection

##eg gate order
GATE_ORDER = [1,3,2,4,3,1]

##RRT Variables
ITERATION = 1000
REWIRE = 0.7
GOAL_R = 0.15
STEP_SIZE = 0.3
MAX_SEG = 1.25 

# Collision radii — inflated to account for the 0.2 m obstacle uncertainty
OBS_RADIUS  = 0.40  # pillar radius 0.06 m + 0.20 m noise + 0.09 m drone body
GATE_RADIUS = 0.40  # gate half-width 0.20 m + 0.20 m noise

##random gen
rng = np.random.default_rng(1)

##waypoint functions
def gate_normal(yaw):
    ##return vector that points through gate
    return np.array([np.sin(yaw), -np.cos(yaw), 0.0])

def gate_points(last_pt, gate_id, z_bounds=(0.10, 1.95)):
    ##buffer
    buf = 0.45

    ##get points
    centre = np.array(GATES[gate_id][:3], dtype=float)
    last_pt = np.array(last_pt, dtype=float)
    yaw    = GATES[gate_id][5]
    nrm    = gate_normal(yaw)   # unit vector along fly-through axis

    # Determine which side to approach from using the incoming direction
    pt_a = buf*nrm + centre 
    pt_b = -buf*nrm + centre
    ##get diff to find the fastest route
    diff_a = pt_a - last_pt
    diff_b = pt_b - last_pt
    ##determine direction
    if np.linalg.norm(diff_a) < np.linalg.norm(diff_b):
        approach = pt_a
        departure = pt_b
    else:
        approach = pt_b
        departure = pt_a

    return approach, centre, departure ##entrance waypoint, centre of gate, exit waypoint

# ── RRT* ─────────────────────────────────────────────────────────────────────
def weighted_dist(a, b):
    """Euclidean distance with extra cost for Z changes."""
    Z_PENALTY = 5.0
    diff = b - a
    return np.sqrt(diff[0]**2 + diff[1]**2 + (diff[2] * Z_PENALTY)**2)

def split_segment(a, b, gate):
    """Return a midpoint between a and b, nudged away from collisions."""
    a, b = np.array(a), np.array(b)
    mid = (a + b) / 2.0 
    mid[2] = 1 ##flat z

    MIN_CLEARANCE = OBS_RADIUS + 0.2   

    ##add clearance buffer
    def has_clearance(pt):
        if in_collision(pt, gate):
            return False
        for obs in OBSTACLES:
            d = np.sqrt((pt[0]-obs[0])**2 + (pt[1]-obs[1])**2)
            if d < MIN_CLEARANCE:
                    return False
        return True

    if has_clearance(mid):
        return mid

    print('midpoint not easily found')
    # midpoint is in collision — sample random offsets until we find a free one
    for _ in range(50):
        noise = rng.uniform(-0.5, 0.5, size=3)
        candidate = mid + noise
        candidate[2] = 1
        if not in_collision(candidate, gate):
            return candidate

    # fallback — return midpoint anyway and let the planner deal with it
    # print(f"  [warn] split_segment could not find free midpoint between {a} and {b}")
    return mid

##collision functions
def in_collision(pt, target_gate):
    """Return True if *pt* is inside any obstacle or non-target gate."""
    x, y, z = pt[:3]

    # Obstacle pillars
    for obs in OBSTACLES:
        px, py = obs[0], obs[1]
        if (x - px)**2 + (y - py)**2 < OBS_RADIUS**2:
            return True

    # Other gates
    for i, gate in enumerate(GATES):
        if i == target_gate:
            continue   # allowed to pass through this gate
        gx, gy, gz = gate[:3]
        if (x - gx)**2 + (y - gy)**2 + (z - gz)**2 < GATE_RADIUS**2:
            return True

    return False ##no collision at pt

def segment_free(a, b, target_gate):
    """Return True if the straight segment a→b is collision-free."""
    dist = np.linalg.norm(b - a)
    n    = max(20, int(dist / 0.05))   # check every ~5 cm
    for t in np.linspace(0.0, 1.0, n):
        pt = a + t * (b - a)
        if in_collision(pt, target_gate):
            return False
    return True

##sampling and steering
def _sample(seg_start, seg_goal):
    """Random sample with goal bias, clipped to a corridor around the segment."""
    if rng.random() < 0.15:
        return seg_goal.copy()
    
    # bounding box for sampling
    lo = np.minimum(seg_start, seg_goal) - PADDING
    hi = np.maximum(seg_start, seg_goal) + PADDING

    # clip to arena bounds
    lo = np.maximum(lo, BOUNDS[:, 0])
    hi = np.minimum(hi, BOUNDS[:, 1])

    ##+/- 20 cm from z for sample
    zlo = 0.8
    zhi = 1.2
    lo[2] = np.maximum(lo[2], zlo)
    hi[2] = np.minimum(hi[2], zhi)
    pt = rng.uniform(lo, hi)
    # pt[2] = 1
    return pt

def nearest(sample, nodes):
    """Index of the tree node nearest to *sample*."""
    dists = np.linalg.norm(np.array(nodes) - sample, axis=1)
    return int(np.argmin(dists))

def steer(from_pt, to_pt):
    """Move from node *from_idx* towards *to_pt* by at most step_size."""
    vec     = to_pt - from_pt
    dist    = np.linalg.norm(vec)
    if dist < 1e-9:
        return from_pt.copy()
    return from_pt + (vec / dist) * min(STEP_SIZE, dist)

def near_id(pt, nodes):
    """Indices of nodes within rewire_r of *pt*."""
    dists = np.linalg.norm(np.array(nodes) - pt, axis=1)
    return np.where(dists < REWIRE)[0]

##last step
def extract_path(idx, nodes, parents):
    """Walk parent pointers back to root and return ordered path."""
    path, idx = [], idx
    while idx != -1:
        path.append(np.array(nodes[idx]))
        idx = parents[idx]
    path.reverse()
    return path

##PLAN segment
def plan(start, goal, target_gate):
    """
    Run RRT* from *start* to *goal* while treating *target_gate* (0-indexed)
    as passable.  Returns the smoothed waypoint list, or [start, goal] on
    failure.
    """
    nodes   = [np.array(start, dtype=float)]
    parents = [-1]
    costs   = [0.0]
    goal_idx = None

    for _ in range(ITERATION):
        sample = _sample(start, goal)
        nearest_idx = nearest(sample, nodes)
        new_pt = steer(np.array(nodes[nearest_idx]), sample)
        # Clamp z to valid bounds
        new_pt[2] = np.clip(new_pt[2], BOUNDS[2, 0], BOUNDS[2, 1])

        ## Best parent
        neighbours = near_id(new_pt, nodes)
        best_parent = nearest_idx
        # Cost to get to new_pt via the initial nearest node
        best_cost = costs[nearest_idx] + weighted_dist(nodes[nearest_idx], new_pt)

        for nb in neighbours:
            # Calculate potential cost via this neighbor
            c = costs[nb] + weighted_dist(nodes[nb], new_pt)
            if c < best_cost:
                # Check if the connection is actually clear
                if segment_free(nodes[nb], new_pt, target_gate):
                    best_cost = c
                    best_parent = nb
        
        #check for clear path
        if not segment_free(nodes[best_parent], new_pt, target_gate):
            continue

        ## add node
        new_idx = len(nodes)
        nodes.append(new_pt.copy())
        parents.append(best_parent)
        costs.append(best_cost)

        ## rewire neighbours
        for nb in neighbours:
            d_to_nb = weighted_dist(new_pt, nodes[nb])#dist to nc
            ##rewire if clear
            if costs[new_idx] + d_to_nb < costs[nb]:
                if segment_free(new_pt, nodes[nb], target_gate):
                    parents[nb] = new_idx
                    costs[nb] = costs[new_idx] + d_to_nb

        # Check if goal is reachable
        goal_arr = np.array(goal)
        if np.linalg.norm(new_pt - goal_arr) < GOAL_R:
            if segment_free(new_pt, goal_arr, target_gate):
                # total_c = costs[]
                g_idx = len(nodes)
                nodes.append(goal_arr.copy())
                parents.append(new_idx)
                costs.append(best_cost + weighted_dist(goal_arr, new_pt))
                goal_idx = g_idx

    
    if goal_idx is None:
        print(f"!!!! RRT* did not reach goal {goal}; using straight line")
        return [np.array(start), np.array(goal)]
  
    return extract_path(goal_idx, nodes, parents)

def path(GATE_ORDER):
    ##Get waypoints
    key_pts = [START.copy()]
    # gate_ids = [-1] ##O index
    gate_ids = []

    prev = START.copy()
    for gid in GATE_ORDER:
        gid0 = gid-1
        gate_ids.extend([gid0, gid0, gid0]) ##for obstacle detection o indexed
        ## add pts to ensure flying in and out normal to gate
        approach, centre, departure = gate_points(prev, gid0)
        key_pts.extend([approach, centre, departure])
        # key_pts.extend([approach, departure])
        prev = departure
    key_pts.append(GOAL.copy())
    gate_ids.append(-1) ##O index
   

    ###run rrt*
    full_path = [START]
    path_total_len = len(key_pts)-1
    ##iterate through all the key points
    for i in range(path_total_len):
        seg_start = key_pts[i]
        seg_goal   = key_pts[i + 1]
        tgt_gate   = gate_ids[i]
        
        ##print update
        print(f"Segment {i+1}/{path_total_len}:"
              f"{np.round(seg_start, 2)} → {np.round(seg_goal, 2)} "
              f"(gate mask: {tgt_gate})")

        ##check for long segs
        dist = np.linalg.norm(np.array(seg_start) - np.array(seg_goal))
        if dist >= MAX_SEG:
            # print('long segment, splitting')
            half_pt = split_segment(seg_start, seg_goal, tgt_gate)
            seg1 = plan(seg_start, half_pt, tgt_gate)
            seg2 = plan(half_pt, seg_goal, tgt_gate)
            full_path.extend(seg1[1:])
            full_path.extend(seg2[1:])
        elif dist < 0.55:
            ##straight line 
            full_path.append(np.array(seg_goal))
        else:
            ##get segment
            seg = plan(seg_start, seg_goal, tgt_gate)
            if full_path:
                seg = seg[1:] 
            full_path.extend(seg)
        
        ##get segment
        # seg = plan(seg_start, seg_goal, tgt_gate)
        # full_path.extend(seg[1:])


    return full_path, key_pts

##plot 
def plot_path(full_path, key_waypoints):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ── Path ──────────────────────────────────────────────────────────────────
    path_arr = np.array(full_path)
    ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
            color='royalblue', linewidth=1.5, label='RRT* path', zorder=3)
    ax.scatter(*START, color='green',  s=80, zorder=5, label='Start')
    ax.scatter(*GOAL,  color='red',    s=80, zorder=5, label='Goal')

    # ── Gates ─────────────────────────────────────────────────────────────────
    GATE_W = 0.4   # half-width of gate opening
    for i, gate in enumerate(GATES):
        gx, gy, gz = gate[:3]
        yaw = gate[5]
        # gate opening is perpendicular to the fly-through normal
        perp = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
        top  = np.array([0, 0, GATE_W])
        # four corners of the gate rectangle
        centre = np.array([gx, gy, gz])
        corners = [
            centre + perp * GATE_W + np.array([0, 0, -GATE_W]),
            centre - perp * GATE_W + np.array([0, 0, -GATE_W]),
            centre - perp * GATE_W + np.array([0, 0,  GATE_W]),
            centre + perp * GATE_W + np.array([0, 0,  GATE_W]),
        ]
        poly = Poly3DCollection([corners], alpha=0.25, facecolor='gold', edgecolor='darkorange', linewidth=1.2)
        ax.add_collection3d(poly)
        ax.text(gx, gy, gz + GATE_W + 0.1, f'G{i+1}',
                fontsize=8, color='darkorange', ha='center')

    # ── Obstacles ─────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 30)
    z_cyl = np.linspace(0.0, 1.95, 2)
    for obs in OBSTACLES:
        ox, oy, r = obs[0], obs[1], OBS_RADIUS
        X = ox + r * np.outer(np.cos(theta), np.ones_like(z_cyl))
        Y = oy + r * np.outer(np.sin(theta), np.ones_like(z_cyl))
        Z =       np.outer(np.ones_like(theta), z_cyl)
        ax.plot_surface(X, Y, Z, color='tomato', alpha=0.25, linewidth=0)
        # top cap
        cap_x = ox + r * np.cos(theta)
        cap_y = oy + r * np.sin(theta)
        cap_corners = list(zip(cap_x, cap_y, np.full_like(cap_x, 1.95)))
        cap_poly = Poly3DCollection([cap_corners], alpha=0.3, facecolor='tomato', linewidth=0)
        ax.add_collection3d(cap_poly)
    # ── Gate order waypoints ───────────────────────────────────────────────────
    kw = np.array(key_waypoints)
    ax.scatter(kw[:, 0], kw[:, 1], kw[:, 2],
               color='orange', s=25, zorder=4, label='Key waypoints')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(BOUNDS[0])
    ax.set_ylim(BOUNDS[1])
    ax.set_zlim(BOUNDS[2])
    ax.set_title('RRT* path — gate sequence ' + str(GATE_ORDER))
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    gates = [1,3,4,1,3,2]
    path, key_waypoints = path(gates)
    print("Key waypoints (gates + start/goal):")
    for i, pt in enumerate(key_waypoints):
        print(f"  {i:2d}: {np.round(pt, 3)}")

    print("\nFull path:")
    for i, pt in enumerate(path):
        print(f"  {i:3d}: {np.round(pt, 3)}")

    ##plot path
    plot_path(path, key_waypoints)

#___ Trajectory optimization _____________________________________________
# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class SegmentCasadiSolver:
    def __init__(self, waypoints, segmentStartDerivatives, segmentEndDerivatives, waypoint_start_times, dt=1/60.0, maxSpeed = 2):
        self.waypoints = waypoints
        self.Nw = waypoints.shape[0]
        total_time = waypoint_start_times[-1] - waypoint_start_times[0]
        self.total_time = total_time
        self.dt = dt
        self.maxSpeed = maxSpeed

        # Number of decsion variables is total time / dt
        N = np.floor(total_time / self.dt).astype(np.int64)

        print("total_time=",total_time)
        print("self.dt=",self.dt)
        print("N=",N)
        # discretized times
        self.dts = np.ones((N,1)) * dt
        # last_dt = total_time - (N-1)*dt
        # self.dts[N-1] = last_dt

        self.N = N
        self.A4 = ca.SX.sym('A4', N)
        self.B4 = ca.SX.sym('B4', N)
        self.C4 = ca.SX.sym('C4', N)

        self.max_accel_xy = maxSpeed / 4
        self.max_accel_z = maxSpeed * 0.2 / 4

        self.max_jerk_xy = self.max_accel_xy / dt
        self.max_jerk_z = self.max_accel_z / dt

        # self.max_snap = self.max_jerk_xy / dt

        snap_integral = 24**2 * ca.sum2(ca.sum1((self.A4 * self.dts)**2)) + 24**2 * ca.sum2(ca.sum1((self.B4 * self.dts)**2)) + 24**2 * ca.sum2(ca.sum1((self.C4 * self.dts)**2))


        A0, A1, A2, A3, A4 = unroll_coefficients(self.A4,waypoints[0,0],self.dts)
        B0, B1, B2, B3, B4 = unroll_coefficients(self.B4,waypoints[0,1],self.dts)
        C0, C1, C2, C3, C4 = unroll_coefficients(self.C4,waypoints[0,2],self.dts)

        cost = snap_integral

        g = []
        lb_vals = []
        ub_vals = []

        adjusted_start_times = waypoint_start_times - waypoint_start_times[0]
        print("adjusted_start_times.shape", adjusted_start_times.shape)
        print("waypoint_start_times.shape", waypoint_start_times.shape) 
        x, xd, xdd, xddd = compute_state_1d(A0,A1,A2,A3,A4, self.dts)
        y, yd, ydd, yddd = compute_state_1d(B0,B1,B2,B3,B4, self.dts)
        z, zd, zdd, zddd = compute_state_1d(C0,C1,C2,C3,C4, self.dts)

        # velocity and acceleration constraint
        # g.append(ca.sqrt(xd**2 + yd**2 + zd**2).max())
        # lb_vals.append(0); ub_vals.append(self.maxSpeed)
        
        # g.append(ca.sqrt(xdd**2 + ydd**2 + zdd**2))
        # lb_vals.append(0); ub_vals.append(self.max_accel_xy)
        
        # g.append(ca.sqrt(xddd**2 + yddd**2 + zddd**2))
        # lb_vals.append(0); ub_vals.append(self.max_jerk_xy)

        # Start point equality
        x0_error = x[0] - waypoints[0,0]
        y0_error = y[0] - waypoints[0,1]
        z0_error = z[0] - waypoints[0,2]

        x0d_error = xd[0] - segmentStartDerivatives[0,0]
        y0d_error = yd[0] - segmentStartDerivatives[0,1]
        z0d_error = zd[0] - segmentStartDerivatives[0,2]

        x0dd_error = xdd[0] - segmentStartDerivatives[1,0]
        z0dd_error = zdd[0] - segmentStartDerivatives[1,1]
        y0dd_error = ydd[0] - segmentStartDerivatives[1,2]

        x0ddd_error = xddd[0] - segmentStartDerivatives[2,0]
        y0ddd_error = yddd[0] - segmentStartDerivatives[2,1]
        z0ddd_error = zddd[0] - segmentStartDerivatives[2,2]

        g.append(x0_error)
        g.append(y0_error)
        g.append(z0_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # equality constraint for speed
        g.append(x0d_error)
        g.append(y0d_error)
        g.append(z0d_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # equality constraints for acceleration
        g.append(x0dd_error)
        g.append(y0dd_error)
        g.append(z0dd_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # # inequality constraints for jerk
        g.append(x0ddd_error)
        g.append(y0ddd_error)
        g.append(z0ddd_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # End waypoint equality
        final_dt = adjusted_start_times[-1] - N * dt
        # final_dt = self.dts[-1]
        xN, xdN, xddN, xdddN = compute_state_1d(A0[-1],A1[-1],A2[-1],A3[-1],A4[-1], final_dt)
        yN, ydN, yddN, ydddN = compute_state_1d(B0[-1],B1[-1],B2[-1],B3[-1],B4[-1], final_dt)
        zN, zdN, zddN, zdddN = compute_state_1d(C0[-1],C1[-1],C2[-1],C3[-1],C4[-1], final_dt)      
        
        xN_error = xN - waypoints[-1,0]
        yN_error = yN - waypoints[-1,1]
        zN_error = zN - waypoints[-1,2]

        g.append(xN_error)
        g.append(yN_error)
        g.append(zN_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        if (np.isfinite(segmentEndDerivatives[0,:]).all()):
            print("applying endpoint velocity condition")
            xNd_error = xdN - segmentEndDerivatives[0,0]
            yNd_error = ydN - segmentEndDerivatives[0,1]
            zNd_error = zdN - segmentEndDerivatives[0,2]
            # equality constraint for speed
            g.append(xNd_error)
            g.append(yNd_error)
            g.append(zNd_error)
            for j in range(0,3): lb_vals.append(0); ub_vals.append(0)
        else:
            print("applying bounds on endpoint velocity")
            g.append(xdN)
            g.append(ydN)
            g.append(zdN)
            for j in range(0,3): lb_vals.append(-self.maxSpeed); ub_vals.append(self.maxSpeed)


        if (np.isfinite(segmentEndDerivatives).all()):
            print("applying endpoint derivative condition")
            xNd_error = xdN - segmentEndDerivatives[0,0]
            yNd_error = ydN - segmentEndDerivatives[0,1]
            zNd_error = zdN - segmentEndDerivatives[0,2]

            xNdd_error = xddN - segmentEndDerivatives[1,0]
            yNdd_error = yddN - segmentEndDerivatives[1,1]
            zNdd_error = zddN - segmentEndDerivatives[1,2]

            xNddd_error = xdddN - segmentEndDerivatives[2,0]
            yNddd_error = ydddN - segmentEndDerivatives[2,1]
            zNddd_error = zdddN - segmentEndDerivatives[2,2]

            # equality constraints for acceleration
            g.append(xNdd_error)
            g.append(yNdd_error)
            g.append(zNdd_error)
            for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            # # inequality constraints for jerk
            g.append(xNddd_error)
            g.append(yNddd_error)
            g.append(zNddd_error)
            for j in range(0,3): lb_vals.append(0); ub_vals.append(0)
        else:
            print("applying bounds in endpoint derivatives")
            g.append(xddN)
            g.append(yddN)
            g.append(zddN)
            for j in range(0,3): lb_vals.append(-self.max_accel_xy); ub_vals.append(self.max_accel_xy)
            g.append(xdddN)
            g.append(xdddN)
            g.append(xdddN)
            for j in range(0,3): lb_vals.append(-self.max_jerk_xy); ub_vals.append(self.max_jerk_xy)

        for i in range(1,self.Nw-1):
            
            index = np.floor(adjusted_start_times[i] / self.dt).astype(np.int64) - 1
            if(index < 0):
                index = 0
            # waypoint_dt = self.dts[index]
            waypoint_dt = adjusted_start_times[i] - (index) * dt

            print("Waypoint i=", i)
            print("discretized index=", index)

            print("waypoint time:", adjusted_start_times[i])
            print("polynomial coefficient time:", index * dt)
            print("waypoint_dt", waypoint_dt)
            print("A0.shape", A0.shape)

            A0_i = A0[index]
            A1_i = A1[index]
            A2_i = A2[index]
            A3_i = A3[index]
            A4_i = A4[index]
            B0_i = B0[index]
            B1_i = B1[index]
            B2_i = B2[index]
            B3_i = B3[index]
            B4_i = B4[index]
            C0_i = C0[index]
            C1_i = C1[index]
            C2_i = C2[index]
            C3_i = C3[index]
            C4_i = C4[index]
            wp_x, wp_xd, wp_xdd, wp_xddd= compute_state_1d(A0_i,A1_i,A2_i,A3_i,A4_i, waypoint_dt)
            wp_y, wp_yd, wp_ydd, wp_yddd= compute_state_1d(B0_i,B1_i,B2_i,B3_i,B4_i, waypoint_dt)
            wp_z, wp_zd, wp_zdd, wp_zddd= compute_state_1d(C0_i,C1_i,C2_i,C3_i,C4_i, waypoint_dt)
            

            x_error = wp_x - waypoints[i,0]
            y_error = wp_y - waypoints[i,1]
            z_error = wp_z - waypoints[i,2]

            # equality constraint for waypoint positions
            g.append(x_error)
            g.append(y_error)
            g.append(z_error)

            for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            g.append(wp_xd)
            g.append(wp_yd)
            g.append(wp_zd)           

            lb_vals.append(-self.maxSpeed)
            lb_vals.append(-self.maxSpeed)
            lb_vals.append(-self.maxSpeed)

            ub_vals.append(self.maxSpeed)
            ub_vals.append(self.maxSpeed)
            ub_vals.append(self.maxSpeed)

            g.append(wp_xdd)
            g.append(wp_ydd)
            g.append(wp_zdd)           

            lb_vals.append(-self.max_accel_xy)
            lb_vals.append(-self.max_accel_xy)
            lb_vals.append(-self.max_accel_z)

            ub_vals.append(self.max_accel_xy)
            ub_vals.append(self.max_accel_xy)
            ub_vals.append(self.max_accel_z)

            # # inequality constraints for jerk
            # g.append(xNddd_error)
            # g.append(yNddd_error)
            # g.append(zNddd_error)
            # equality_constraints += 12


        opt_variables = ca.vertcat(
            ca.reshape(self.A4, -1, 1), 
            ca.reshape(self.B4, -1, 1), 
            ca.reshape(self.C4, -1, 1))

        opt_constraints = ca.vertcat(*g)

        print("opt_constraints.shape", opt_constraints.shape)

        self.lbg = np.array(lb_vals)
        self.ubg = np.array(ub_vals)

        print("self.lbg.shape", self.lbg.shape)
        print("self.ubg.shape", self.ubg.shape)

        # p - equality constraints
        nlp_prob = {'f': cost, 'x': opt_variables, 'g': opt_constraints}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-4,
            'jit': True,
            'compiler': 'shell',
            'jit_options': {
                'flags': ['-O3'],
                'verbose': False
            }
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    
    def solve_coefficients(self,initial_guess):
        solution = self.solver(x0 = initial_guess, lbg=self.lbg, ubg=self.ubg)
        X = solution['x']

        coeffs = X.reshape((-1,3))

        A4 = coeffs[:,0]
        B4 = coeffs[:,1]
        C4 = coeffs[:,2]

        return A4, B4, C4, solution
    

def compute_state_1d(P0, P1, P2, P3, P4, dt):
    # integrate position
    p = P0 + P1 * dt + P2*dt**2 + P3 * dt**3 + P4 * dt**4

    # integrate velocity
    pd = P1 + 2 * P2 * dt + 3 * P3 * dt**2 + 4 * P4 * dt**3

    # integrate acceleration
    pdd = 2 * P2 + 6 * P3 * dt + 12 * P4 * dt**2

    # integrate jerk
    pddd = 6 * P3 + 24 * P4 * dt

    return p, pd, pdd, pddd


def unroll_coefficients(P4, WP0, times):
    n = times.size

    M = np.zeros((n,n))
    M[1:,:] = np.tril(np.ones((n-1,n)))


    # x = A0 + A1 t + A2 t^2 + A3 t^3 + A4 t^4
    # dx (t) = A1 + 2 * A2 * t + 3 * A3 * t^2 + 4  * A4 * t^3
    #  A1_1 = A1_0 + 2* A2 * t + 3 * A3_0 * t^2 + 4 * T^3



    # DDX = 2 * A2 + 6 * A3 *t + 12 * A4 * t**2
    ### 2 * A2_1 = 2 * A2_0 + 6 * A3_0 * t0 + 12 * A4 * t**2

    # DDDX = 6 * A3 + 24 * A4  * t
    ###  6 A3_1 = 6*A3_0 + 24 A4 * t => A3_1 = A3_0

    #D4X = 24 * A4
    P3 = 4 * ca.mtimes(M, P4) * times

    # P2 = 3 * M @ (P3 * times)
    P2 = 3 * ca.mtimes(M, P3) * times + 6 * ca.mtimes(M,P4) * times**2

    # P1 = 2 * M @ (P2 * times) + 3 * M @ (P3 * times **2)

    P1 = 2 * ca.mtimes(M, P2) * times + 3* ca.mtimes(M, P3) * times **2 + 4 * ca.mtimes(M, P4) * times**3

    # P0 = M @ (P1 * times) + M @ (P2 * times **2) + M @ (P3 * times **3) + waypoints[0]
    P0 =WP0 +  ca.mtimes(M, P1) * times + ca.mtimes(M, P2) * times**2 + ca.mtimes(M, P3) * times**3 + ca.mtimes(M,P4) * times**4

    # p = P0 + P1 * times + P2 * times**2 + P3 * times**3
    # p = P0 + P1 * times + P2*times**2 + P3 * times**3

    return P0, P1, P2, P3, P4

def integrate_1d(P4,WP0, times):

    P0, P1, P2, P3, P4 = unroll_coefficients(P4, WP0, times)

    # integrate position
    p = P0 + P1 * times + P2*times**2 + P3 * times**3 + P4 * times**4

    # integrate velocity
    pd = P1 + 2 * P2 * times + 3 * P3 * times**2 + 4 * P4 * times**3

    # integrate acceleration
    pdd = 2 * P2 + 6 * P3 * times + 12 * P4 * times**2

    return p, pd, pdd


# def evalute_polynomials_over_control_time_step(dt,A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4):
#     x_vals = np.array([])
#     y_vals = np.array([])
#     z_vals = np.array([])
#     # t_vals = np.array([])

#     x_vals = A0 + A1 * dt + A2 * dt**2 + A3 * dt**3 + A4 * dt**4
#     y_vals = B0 + B1 * dt + B2 * dt**2 + B3 * dt**3 + B4 * dt**4
#     z_vals = C0 + C1 * dt + C2 * dt**2 + C3 * dt**3 + C4 * dt**4

#     x_array = np.array(x_vals).flatten()
#     y_array = np.array(y_vals).flatten()
#     z_array = np.array(z_vals).flatten()

#     return x_array, y_array, z_array

def evalute_polynomials_over_control_time_step(times, n, freq, A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4):
    # x_vals = np.array([])
    # y_vals = np.array([])
    # z_vals = np.array([])
    p_vals = np.array([])
    v_vals = np.array([])
    a_vals = np.array([])
    j_vals = np.array([])

    print("times.shape", times.shape)
    print("A0.shape", times.shape)

    for idx in range(0,n):
        duration = times[idx]

        nsample = int(duration * freq)
        ti = np.arange(nsample) * 1 / freq

        x, xd, xdd, xddd = compute_state_1d(A0[idx], A1[idx], A2[idx], A3[idx], A4[idx], ti)
        y, yd, ydd, yddd = compute_state_1d(B0[idx], B1[idx], B2[idx], B3[idx], B4[idx], ti)
        z, zd, zdd, zddd = compute_state_1d(C0[idx], C1[idx], C2[idx], C3[idx], C4[idx], ti)

        p = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
        v = np.hstack([xd.reshape(-1,1), yd.reshape(-1,1), zd.reshape(-1,1)])
        a = np.hstack([xdd.reshape(-1,1), ydd.reshape(-1,1), zdd.reshape(-1,1)])
        j = np.hstack([xddd.reshape(-1,1), yddd.reshape(-1,1), zddd.reshape(-1,1)])

        p_vals = p if(p_vals.size == 0) else np.concatenate([p_vals, p]).reshape(-1,3)
        v_vals = v if(v_vals.size == 0) else np.concatenate([v_vals, v]).reshape(-1,3)
        a_vals = a if(a_vals.size == 0) else np.concatenate([a_vals, a]).reshape(-1,3)
        j_vals = j if(j_vals.size == 0) else np.concatenate([j_vals, j]).reshape(-1,3)


    states = np.hstack([p_vals, v_vals, a_vals, j_vals])

    return states

def generate_trajectory(waypoints, averageSpeed, discretization_dt, ctrl_freq, numWaypointsPerGroup, usePlot = False):
    print("waypoints.shape=", waypoints.shape)

    Nw = waypoints.shape[0]

    p_prev = waypoints[0:Nw-1,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    total_length = np.sum(waypoint_min_lengths)
    total_time = total_length / averageSpeed
    segment_durations = waypoint_min_lengths / total_length * total_time

    maxSpeed = 1.2 *averageSpeed

    waypoint_start_times = np.zeros((Nw,))
    for i in range(0,segment_durations.size):
        waypoint_start_times[i+1] = waypoint_start_times[i] + segment_durations[i]

    desired_derivatives = np.ones((Nw, 3, 3)) * np.inf
    desired_derivatives[0,:,:] = 0
    desired_derivatives[Nw-1,:,:] = 0

    A0 = np.array([])
    B0 = np.array([])
    C0 = np.array([])

    A1 = np.array([])
    B1 = np.array([])
    C1 = np.array([])

    A2 = np.array([])
    B2 = np.array([])
    C2 = np.array([])

    A3 = np.array([])
    B3 = np.array([])
    C3 = np.array([])

    A4 = np.array([])
    B4 = np.array([])
    C4 = np.array([])

    durations = np.array([])

    if usePlot:
        ax0 = plt.figure().add_subplot(projection='3d')
        wx = waypoints[:,0]
        wy = waypoints[:,1]
        wz = waypoints[:,2]

        ax0.scatter(wx, wy, wz, marker='o')

    segment_derivatives = np.zeros((2,3,3))
    segment_derivatives[1,:,:] = np.inf

    segmentStartDerivatives = np.zeros((3,3))
    segmentEndDerivatives  = np.ones((3,3)) * np.inf

    print("Num waypoints = ", Nw)

    print("*************************************************")

    num3Groups = np.floor((Nw - 1)/(numWaypointsPerGroup-1)).astype(np.uint32)


    for groupIdx in range(0, num3Groups): 

        segmentStartIdx = groupIdx * (numWaypointsPerGroup-1)
        
        if(groupIdx == num3Groups- 1):
            segmentEndIdx = Nw-1
        else:
            segmentEndIdx = segmentStartIdx + numWaypointsPerGroup-1

        numSegmentsInGroup = (segmentEndIdx - segmentStartIdx)
        print("solving group ", groupIdx)
        print("segmentStartIdx ", segmentStartIdx)
        print("segmentEndIdx ", segmentEndIdx)
        print("numSegmentsInGroup ", numSegmentsInGroup)

        


        # if(segmentEndIdx < Nw-numWaypointsPerGroup):
        #     nextGroupWaypoint = waypoints[segmentEndIdx+1,:]
        #     # delta = (waypoints[segmentEndIdx+1,:] - waypoints[segmentEndIdx,:])
        #     # endVel = delta/np.linalg.norm(delta) * averageSpeed
        #     # segmentEndDerivatives = np.ones((3,3)) * np.inf
        #     # segmentEndDerivatives[0,:] = endVel
        # else:
        #     print("sending trajectory endpoint velocity to zero")
        if(groupIdx == num3Groups-1):
            print("setting trajectory endpoint velocity to zero")
            segmentEndDerivatives = np.zeros((3,3))
        else:
            # delta = (waypoints[segmentEndIdx+1,:] - waypoints[segmentEndIdx,:])
            # endVel = delta/np.linalg.norm(delta) * averageSpeed
            # segmentEndDerivatives = np.ones((3,3)) * np.inf
            # segmentEndDerivatives[0,:] = endVel
            segmentEndDerivatives  = np.ones((3,3)) * np.inf


        segment_waypoints = waypoints[segmentStartIdx:segmentEndIdx+1,:]

        groupDuration = np.sum(segment_durations[segmentStartIdx:segmentEndIdx])

        print("duration i", segment_durations[segmentStartIdx])
        print("duration i+1", segment_durations[segmentStartIdx])
        # groupDuration = segment_durations[segmentStartIdx] + segment_durations[segmentStartIdx+1]

        segment_waypoint_start_times = waypoint_start_times[segmentStartIdx:segmentEndIdx+1]

        # discretization_dt = groupDuration / (numSegmentsInGroup*numSubsections)
        print("solving group ", groupIdx)
        print("segmentStartIdx ", segmentStartIdx)
        print("segmentEndIdx ", segmentEndIdx)

        print("groupDuration ", groupDuration)
        print("discretization_dt ", discretization_dt)
        print("segment_waypoints.shape ", segment_waypoints.shape)
        print("segment_waypoint_start_times.shape ", segment_waypoint_start_times.shape)
        print("segment_waypoints ", segment_waypoints)
        print("segment_waypoint_start_times ", segment_waypoint_start_times)

        solver = SegmentCasadiSolver(segment_waypoints,segmentStartDerivatives, segmentEndDerivatives, segment_waypoint_start_times, discretization_dt, maxSpeed)

        num_decision_variables = 3 * solver.N

        X0 = np.zeros((num_decision_variables,))
        
        A4_i, B4_i, C4_i, solution = solver.solve_coefficients(X0)
        A0_i, A1_i, A2_i, A3_i, _ = unroll_coefficients(np.array(A4_i),segment_waypoints[0,0],solver.dts)
        B0_i, B1_i, B2_i, B3_i, _ = unroll_coefficients(np.array(B4_i),segment_waypoints[0,1],solver.dts)
        C0_i, C1_i, C2_i, C3_i, _ = unroll_coefficients(np.array(C4_i),segment_waypoints[0,2],solver.dts)

        A0 = np.concatenate([A0, np.array(A0_i).flatten()])
        B0 = np.concatenate([B0, np.array(B0_i).flatten()])
        C0 = np.concatenate([C0, np.array(C0_i).flatten()])

        A1 = np.concatenate([A1, np.array(A1_i).flatten()])
        B1 = np.concatenate([B1, np.array(B1_i).flatten()])
        C1 = np.concatenate([C1, np.array(C1_i).flatten()])

        A2 = np.concatenate([A2, np.array(A2_i).flatten()])
        B2 = np.concatenate([B2, np.array(B2_i).flatten()])
        C2 = np.concatenate([C2, np.array(C2_i).flatten()])

        A3 = np.concatenate([A3, np.array(A3_i).flatten()])
        B3 = np.concatenate([B3, np.array(B3_i).flatten()])
        C3 = np.concatenate([C3, np.array(C3_i).flatten()])

        A4 = np.concatenate([A4, np.array(A4_i).flatten()])
        B4 = np.concatenate([B4, np.array(B4_i).flatten()])
        C4 = np.concatenate([C4, np.array(C4_i).flatten()])
        durations = np.concatenate([durations, solver.dts.flatten()])
        
        # ax0.scatter(A0,B0,C0)

        
        tn = solver.dts[-1]
        xn, xdn, xddn, xdddn = compute_state_1d(A0_i[-1],A1_i[-1],A2_i[-1],A3_i[-1],A4_i[-1], tn)
        yn, ydn, yddn, ydddn = compute_state_1d(B0_i[-1],B1_i[-1],B2_i[-1],B3_i[-1],B4_i[-1], tn)
        zn, zdn, zddn, zdddn = compute_state_1d(C0_i[-1],C1_i[-1],C2_i[-1],C3_i[-1],C4_i[-1], tn)

        endpoint_speed = np.sqrt(xdn**2 + ydn**2 + zdn**2)

        # Update velocity constraint at end of segment
        print("updating segment endpoint velocity: ", xdn, ydn, zdn)
        print("endpoint speed: ", endpoint_speed)
        segmentStartDerivatives[0,0] = xdn
        segmentStartDerivatives[0,1] = ydn
        segmentStartDerivatives[0,2] = zdn

        segmentStartDerivatives[1, 0] = xddn
        segmentStartDerivatives[1, 1] = yddn
        segmentStartDerivatives[1, 2] = zddn

        segmentStartDerivatives[2, 1] = ydddn
        segmentStartDerivatives[2, 0] = xdddn
        segmentStartDerivatives[2, 2] = zdddn


        print("--------------------------------------------------------------------")

    if usePlot:
        plt.savefig("P0_coeffs.png")

    np.savez("combined_coefficients.npz", 
             A4=A4,
             B4=B4,
             C4=C4,
             times=durations,
             waypoints=waypoints)

    N_discretized_segments = A0.shape[0]

    print("A0.shape", A0.shape)
    print("durations.shape", durations.shape)

    states = evalute_polynomials_over_control_time_step(durations, N_discretized_segments, ctrl_freq, 
                                                                           A0, A1, A2, A3, A4, 
                                                                           B0, B1, B2, B3, B4, 
                                                                           C0, C1, C2, C3, C4)

    print("states.shape", states.shape)

    sx, wp_xd, wp_xdd, wp_xddd= compute_state_1d(A0,A1,A2,A3,A4, durations)
    sy, wp_yd, wp_ydd, wp_yddd= compute_state_1d(B0,B1,B2,B3,B4, durations)
    sz, wp_zd, wp_zdd, wp_zddd= compute_state_1d(C0,C1,C2,C3,C4, durations)

    if usePlot:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(sx,sy,sz)


    p_ref = states[:,0:3]
    v_ref = states[:,3:6]
    a_ref = states[:,6:9]
    j_ref = states[:,9:12]

    yaw_desired = np.zeros_like(states[:,0])
    yaw_rate_desired = np.zeros_like(states[:,0])
    
    euler_ref, body_rates_ref = evaluate_angular_states_over_trajectory(yaw_desired, yaw_rate_desired, a_ref, j_ref)

    states = np.column_stack([p_ref, v_ref, a_ref, euler_ref, body_rates_ref]).reshape(-1, 15)
    return states, total_time

def evaluate_angular_states_over_trajectory(yaw_vals, phi_dot_vals, a_ref, j_ref):
    ref_ax = a_ref[:,0]
    ref_ay = a_ref[:,1]
    ref_az = a_ref[:,2]

    ref_jx = j_ref[:,0]
    ref_jy = j_ref[:,1]
    ref_jz = j_ref[:,2]

    # euler values
    euler_values = []
    # body rates
    body_rates = []

    numStates = ref_ax.shape[0]
    for i in range(0, numStates):
        # desired yaw angle
        # vx = ref_vx[i]
        # vy = ref_vy[i]

        yaw_des = yaw_vals[i]
        phi_dot = phi_dot_vals[i]

        a = np.array([ref_ax[i], ref_ay[i], ref_az[i]])
        g = 9.8
        a_des = a + np.array([0.,0.,g])

        z_b = a_des / np.linalg.norm(a_des)
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        y_c = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0])
        x_b = np.cross(y_c, z_b)
        x_b = x_b / np.linalg.norm(x_b)
        y_b = np.cross(z_b, x_b)
        y_b = y_b / np.linalg.norm(y_b)
        R = np.column_stack([x_b, y_b, z_b])

        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=False).reshape(-1,1)
        T = np.linalg.norm(a_des)

        euler_values.append(euler)

        # jerk
        c = T
        j = np.array([ref_jx[i],ref_jy[i],ref_jz[i]]).T

        w_x = - y_b.T @ j / c
        w_y = x_b.T @ j / c
        w_z = phi_dot * x_c.T @ x_b + w_y * y_c.T @ z_b

        body_rates.append(np.array([w_x, w_y, w_z]))
    
    euler_ref = np.array(euler_values).reshape(-1,3)
    body_rates_ref = np.array(body_rates).reshape(-1,3)

    return euler_ref, body_rates_ref