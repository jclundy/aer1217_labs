import numpy as np


# ── Gate geometry helpers ─────────────────────────────────────────────────────

def gate_normal(yaw):
    """Unit vector that points through the gate opening (fly-through direction).

    The gate frame is a square in the plane perpendicular to this vector.
      yaw = 0       → gate plane is XZ → fly-through is ±Y → normal = [0,-1,0]
      yaw = ±pi/2   → gate plane is YZ → fly-through is ±X → normal = [±1,0,0]
    """
    return np.array([np.sin(yaw), -np.cos(yaw), 0.0])


def gate_via_points(gate_raw, prev_pos, next_pos, buf=0.40, z_bounds=(0.10, 1.95)):
    """Return (approach, centre, departure) waypoints for a gate.

    The approach and departure points are placed exactly along the gate normal
    (the fly-through axis), NOT along the direction of travel.  This guarantees
    the drone enters and exits perpendicular to the gate frame regardless of
    where it came from, preventing diagonal frame collisions.

    Args:
        gate_raw : [x, y, z, r, p, yaw, type]
        prev_pos : position before this gate (used only to pick the sign)
        next_pos : position after  this gate (used only to pick the sign)
        buf      : metres before/after gate centre to place sub-waypoints
        z_bounds : (z_lo, z_hi) to clamp sub-waypoints
    """
    centre = np.array(gate_raw[:3], dtype=float)
    yaw    = gate_raw[5]
    nrm    = gate_normal(yaw)   # unit vector along fly-through axis

    # Determine which side to approach from using the incoming direction
    diff = centre - np.array(prev_pos, dtype=float)
    sign = np.sign(np.dot(diff, nrm))
    if abs(sign) < 0.1:
        sign = 1.0

    approach  = centre - sign * buf * nrm
    departure = centre + sign * buf * nrm

    # Clamp z
    for pt in (approach, departure, centre):
        pt[2] = np.clip(pt[2], z_bounds[0], z_bounds[1])

    return approach, centre, departure


# ── RRT* ─────────────────────────────────────────────────────────────────────

class RRTStar:
    """RRT* path planning in 3-D.

    Parameters
    ----------
    start, goal   : array-like, shape (3,)
    obstacles     : list of [x, y, z, r, p, yaw] — pillar centres
    gates         : list of [x, y, z, r, p, yaw, type] — gate centres
    bounds        : array-like, shape (3, 2) — [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    max_it        : maximum RRT* iterations
    step_size     : max distance to extend per step  (metres)
    goal_r        : radius within which goal is considered reached (metres)
    rewire_r      : neighbourhood radius for rewiring (metres)
    rng           : numpy Generator (for reproducibility)
    """

    # Collision radii — inflated to account for the 0.2 m obstacle uncertainty
    OBS_RADIUS  = 0.35   # pillar radius 0.06 m + 0.20 m noise + 0.09 m drone body
    GATE_RADIUS = 0.30   # gate half-width 0.20 m + some margin

    def __init__(self, start, goal, obstacles, gates, bounds,
                 max_it=500, step_size=0.4, goal_r=0.15, rewire_r=0.6, rng=None):

        self.start     = np.array(start,  dtype=float)
        self.goal      = np.array(goal,   dtype=float)
        self.obstacles = obstacles          # list of obstacle descriptors
        self.gates     = gates              # list of gate descriptors
        self.bounds    = np.array(bounds,  dtype=float)   # shape (3,2)
        self.max_it    = max_it
        self.step_size = step_size
        self.goal_r    = goal_r
        self.rewire_r  = rewire_r
        self.rng       = rng if rng is not None else np.random.default_rng(1)

        # Tree storage
        self.nodes   = [self.start.copy()]
        self.parents = [-1]
        self.costs   = [0.0]

    # ── Collision ─────────────────────────────────────────────────────────────

    def _point_in_collision(self, pt, target_gate_idx=None):
        """Return True if *pt* is inside any obstacle or non-target gate.

        Args:
            pt               : (3,) position to test
            target_gate_idx  : 0-based index of the gate being navigated through
                               (that gate is skipped in collision checks); None = skip all gate checks
        """
        x, y, z = pt[:3]

        # Obstacle pillars — treat as infinite-height cylinders (z check optional)
        for obs in self.obstacles:
            px, py = obs[0], obs[1]
            if (x - px)**2 + (y - py)**2 < self.OBS_RADIUS**2:
                return True

        # Other gates — treated as spheres to avoid clipping their frames
        for i, gate in enumerate(self.gates):
            if i == target_gate_idx:
                continue   # allowed to pass through this gate
            gx, gy, gz = gate[:3]
            if (x - gx)**2 + (y - gy)**2 + (z - gz)**2 < self.GATE_RADIUS**2:
                return True

        return False

    def _segment_free(self, a, b, target_gate_idx=None):
        """Return True if the straight segment a→b is collision-free."""
        dist = np.linalg.norm(b - a)
        n    = max(8, int(dist / 0.05))   # check every ~5 cm
        for t in np.linspace(0.0, 1.0, n):
            pt = a + t * (b - a)
            if self._point_in_collision(pt, target_gate_idx):
                return False
        return True

    # ── Sampling & steering ───────────────────────────────────────────────────

    def _sample(self):
        """Random sample with 15 % goal bias."""
        if self.rng.random() < 0.15:
            return self.goal.copy()
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return self.rng.uniform(lo, hi)

    def _nearest(self, sample):
        """Index of the tree node nearest to *sample*."""
        dists = np.linalg.norm(np.array(self.nodes) - sample, axis=1)
        return int(np.argmin(dists))

    def _steer(self, from_idx, to_pt):
        """Move from node *from_idx* towards *to_pt* by at most step_size."""
        from_pt = np.array(self.nodes[from_idx])
        vec     = to_pt - from_pt
        dist    = np.linalg.norm(vec)
        if dist < 1e-9:
            return from_pt.copy()
        return from_pt + (vec / dist) * min(self.step_size, dist)

    def _near_indices(self, pt):
        """Indices of nodes within rewire_r of *pt*."""
        dists = np.linalg.norm(np.array(self.nodes) - pt, axis=1)
        return np.where(dists < self.rewire_r)[0]

    def _extract_path(self, idx):
        """Walk parent pointers back to root and return ordered path."""
        path = []
        while idx != -1:
            path.append(np.array(self.nodes[idx]))
            idx = self.parents[idx]
        path.reverse()
        return path

    # ── Main loop ─────────────────────────────────────────────────────────────

    def plan(self, target_gate_idx=None):
        """Run RRT* and return the path as a list of (3,) arrays.

        Args:
            target_gate_idx : 0-based gate index that collision checks should
                              ignore (the gate we are flying through).
                              Pass None if no gate should be ignored.

        Returns:
            list of np.ndarray (3,), or None if no path found within max_it.
        """
        for _ in range(self.max_it):

            sample      = self._sample()
            nearest_idx = self._nearest(sample)
            new_pt      = self._steer(nearest_idx, sample)

            # Reject if the new point itself is in collision
            if self._point_in_collision(new_pt, target_gate_idx):
                continue

            # Reject if the edge to the new point is in collision
            if not self._segment_free(self.nodes[nearest_idx], new_pt, target_gate_idx):
                continue

            # ── Choose best parent from neighbourhood ──────────────────────
            near_idxs   = self._near_indices(new_pt)
            best_parent = nearest_idx
            best_cost   = (self.costs[nearest_idx]
                           + np.linalg.norm(self.nodes[nearest_idx] - new_pt))

            for idx in near_idxs:
                edge_cost = np.linalg.norm(self.nodes[idx] - new_pt)
                cost_via  = self.costs[idx] + edge_cost
                if cost_via < best_cost and self._segment_free(self.nodes[idx], new_pt, target_gate_idx):
                    best_parent = idx
                    best_cost   = cost_via

            # ── Add node to tree ───────────────────────────────────────────
            new_idx = len(self.nodes)
            self.nodes.append(new_pt)
            self.parents.append(best_parent)
            self.costs.append(best_cost)

            # ── Rewire neighbours through new node if cheaper ──────────────
            for idx in near_idxs:
                edge_cost = np.linalg.norm(self.nodes[idx] - new_pt)
                cost_via  = self.costs[new_idx] + edge_cost
                if cost_via < self.costs[idx] and self._segment_free(new_pt, self.nodes[idx], target_gate_idx):
                    self.parents[idx] = new_idx
                    self.costs[idx]   = cost_via

            # ── Check goal ─────────────────────────────────────────────────
            if np.linalg.norm(new_pt - self.goal) < self.goal_r:
                # Connect directly to goal
                goal_cost = self.costs[new_idx] + np.linalg.norm(new_pt - self.goal)
                self.nodes.append(self.goal.copy())
                self.parents.append(new_idx)
                self.costs.append(goal_cost)
                return self._extract_path(len(self.nodes) - 1)

        # max_it exhausted without reaching goal
        return None
