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
##eg gate order
GATE_ORDER = [1,3,2,4,3,1]
##RRT Variables
ITERATION = 300
REWIRE = 0.6
GOAL_R = 0.1
STEP_SIZE = 0.4
# Collision radii — inflated to account for the 0.2 m obstacle uncertainty
OBS_RADIUS  = 0.35   # pillar radius 0.06 m + 0.20 m noise + 0.09 m drone body
GATE_RADIUS = 0.20   # gate half-width 0.20 m + 0.20 m noise

##random gen
rng = np.random.default_rng(1)

##waypoint functions
def gate_normal(yaw):
    ##return vector that points through gate
    return np.array([np.sin(yaw), -np.cos(yaw), 0.0])

def gate_points(last_pt, gate_id, z_bounds=(0.10, 1.95)):
    """Return (approach, centre, departure) waypoints for a gate."""

    ##buffer
    buf = 0.3

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
##collision functions
def in_collision(pt, target_gate):
    """Return True if *pt* is inside any obstacle or non-target gate."""
    x, y, z = pt[:3]

    # Obstacle pillars
    for obs in OBSTACLES:
        px, py = obs[0], obs[1]
        if (x - px)**2 + (y - py)**2 < 0.35**2:
            return True

    # Other gates
    for i, gate in enumerate(GATES):
        if i == target_gate:
            continue   # allowed to pass through this gate
        gx, gy, gz = gate[:3]
        if (x - gx)**2 + (y - gy)**2 + (z - gz)**2 < 0.20**2:
            return True

    return False ##no collision at pt

def segment_free(a, b, target_gate):
    """Return True if the straight segment a→b is collision-free."""
    dist = np.linalg.norm(b - a)
    n    = max(8, int(dist / 0.05))   # check every ~5 cm
    for t in np.linspace(0.0, 1.0, n):
        pt = a + t * (b - a)
        if in_collision(pt, target_gate):
            return False
    return True

##sampling and steering
def _sample(goal):
    """Random sample with 15 % goal bias."""
    if rng.random() < 0.15:
        return goal.copy()
    lo = BOUNDS[:, 0]
    hi = BOUNDS[:, 1]
    return rng.uniform(lo, hi)

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
    return from_pt + (vec / dist) * min(0.4, dist)

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
        sample = _sample(goal)
        nearest_idx = nearest(sample, nodes)
        new_pt = steer(np.array(nodes[nearest_idx]), sample)
        # Clamp z to valid bounds
        new_pt[2] = np.clip(new_pt[2], BOUNDS[2, 0], BOUNDS[2, 1])

        ##chose best parents
        neighbours = near_id(new_pt, nodes)
        best_parent = nearest_idx
        best_cost = costs[nearest_idx] + np.linalg.norm(new_pt - nodes[nearest_idx])
        for nb in neighbours:
            c = costs[nb] + np.linalg.norm(new_pt - nodes[nb])
            if c<best_cost and segment_free(nodes[best_parent], new_pt, target_gate):
                best_cost = c
                best_parent = nb
        if not segment_free(nodes[best_parent], new_pt, target_gate):
            continue

        ##add node
        new_idx = len(nodes)
        nodes.append(new_pt.copy())
        parents.append(best_parent)
        costs.append(best_cost)

        ###rewire
        for nb in neighbours:
            c = best_cost + np.linalg.norm(np.array(nodes[nb]) - new_pt)
            if c < costs[nb] and segment_free(new_pt, np.array(nodes[nb]), target_gate):
                parents[nb] = new_idx
                costs[nb]   = c
        
        # Check if goal is reachable
        goal_arr = np.array(goal)
        if np.linalg.norm(new_pt - goal_arr) < STEP_SIZE:
            if segment_free(new_pt, goal_arr, target_gate):
                g_idx = len(nodes)
                nodes.append(goal_arr.copy())
                parents.append(new_idx)
                costs.append(best_cost + np.linalg.norm(goal_arr - new_pt))
                goal_idx = g_idx
    
    if goal_idx is None:
        print(f"!!!! RRT* did not reach goal {goal}; using straight line")
        return [np.array(start), np.array(goal)]
  
    return extract_path(goal_idx, nodes, parents)

def path():
    ##Get waypoints
    key_pts = [START.copy()]
    gate_ids = [] ##O index

    prev = START.copy()
    for gid in GATE_ORDER:
        gid0 = gid-1
        gate_ids.extend([gid0, gid0, gid0]) ##for obstacle detection o indexed
        ## add pts to ensure flying in and out normal to gate
        approach, centre, departure = gate_points(prev, gid0)
        key_pts.extend([approach, centre, departure])
        prev = departure
    key_pts.append(GOAL.copy())
    gate_ids.append(-1)   # final leg: no gate to pass through

    ###run rrt*
    full_path = []
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
        
        ##get segment
        seg = plan(seg_start, seg_goal, tgt_gate)
        full_path.extend(seg)

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
        corners = [
            [gx, gy, gz - GATE_W] + perp * GATE_W,
            [gx, gy, gz - GATE_W] - perp * GATE_W,
            [gx, gy, gz + GATE_W] - perp * GATE_W,
            [gx, gy, gz + GATE_W] + perp * GATE_W,
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
    path, key_waypoints = path()
    print("Key waypoints (gates + start/goal):")
    for i, pt in enumerate(key_waypoints):
        print(f"  {i:2d}: {np.round(pt, 3)}")

    print("\nFull path:")
    for i, pt in enumerate(path):
        print(f"  {i:3d}: {np.round(pt, 3)}")

    ##plot path
    plot_path(path, key_waypoints)


