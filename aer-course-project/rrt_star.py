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
    """Return (approach, centre, departure) waypoints for a gate."""

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

    return approach, centre, departure

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


