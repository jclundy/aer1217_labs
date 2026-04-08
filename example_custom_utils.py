import numpy as np


##Gate positioning
##get fly through direction from gate yaw
def gate_normal(yaw):
    """Unit vector that points through the gate opening (fly-through direction).

    The gate frame is a square in the plane perpendicular to this vector.
      yaw = 0       → gate plane is XZ → fly-through is ±Y → normal = [0,-1,0]
      yaw = ±pi/2   → gate plane is YZ → fly-through is ±X → normal = [±1,0,0]
    """
    return np.array([np.sin(yaw), -np.cos(yaw), 0.0])

##get approach, center and depart points for a gate
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

###RRT* algo
class RRTStar:
    """RRT* path planning algorithm.

    """
    def __init__ (self, start, goal, obstacles, bounds, max_it=300, step_size=0.5, goal_r=0.1, rewire_r=0.5, rng=None):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_it = max_it
        self.step_size = step_size
        self.goal_r = goal_r
        self.rewire_r = rewire_r
        self.rng = rng if rng is not None else np.random.default_rng(1)


        ##tree storage
        self.nodes = [self.start.copy()] ##start tree with first node at start
        self.parents = [-1] 
        self.bounds = [0.0] ## no cost to start

    ##check if path will collide with any obstacles
    def collision(self, pt, objective):

        ##check if path between points will hit an obstacle
        for obs in self.obstacles:
            if obs[5] == objective[3]:
                ##ignore this obstacle
                continue
         
            px, py, pz = obs[:3]
            obs_r = obs[3]
            if obs_r == 0:
                obs_r = 0.5 ##assume big

            ##check if line between p1 and p2 intersects with sphere around obs
            x, y, z = pt[:3]
            if self.line_sphere_intersection(px, py, pz, obs_r, x, y, z):
                return True
            return False

    def segment_collision(self, pt):
        ##sample points in 5cm resolution to check
        pass

    
    ##RRT* steps

    ##pick sample
    def _sample(self):
        goal_bias = 0.15
        if self.rng.random() < goal_bias:
            sample = self.goal
        lo, hi = self.bounds [:,0], self.bounds[:,1]
        return self.rng.uniform(lo, hi)
    
    ##return idx of nearest node in tree
    def _nearest(self, sample):
        dists = np.linalg.norm(self.nodes - sample, axis=1)
        return np.argmin(dists)
    
    ##move to NN by step size
    def _steer(self, nearest_idx, sample):
        to_pt = self.nodes[nearest_idx]
        from_pt = sample
        vec = from_pt - to_pt
        dist = np.linalg.norm(vec)
        
        return from_pt + (vec/dist) * min(self.step_size, dist)

    ##idx for rewire
    def _near(self, new_node):
        dists = np.linalg.norm(self.nodes - new_node, axis=1)
        return np.where(dists < self.rewire_r)[0]
    
    def _extract_path(self, idx):
        path = []
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        path.reverse()
        return path
    ###RRT* main loop
    def plan(self, objective):

        for i in range(self.max_it):

            sample = self._sample() #pick pt
            nearest_idx = self._nearest(sample) #find NN
            new_pt = self._steer(nearest_idx, sample) #go to NN

            ##chech for collisions
            if self.collision(new_pt, objective):
                continue
            if not self.segment_collision(new_pt):
                continue

            ##choose parent
            near_idxs = self._near(new_pt)
            best_parent = nearest_idx
            best_cost = self.cost[nearest_idx] + np.linalg.norm(self.nodes[nearest_idx] - new_pt)

            for idx in near_idxs:
                cost_t = self.cost[idx] + np.linalg.norm(self.nodes[idx] - new_pt)
                if cost_t < best_cost and self.collision(new_pt, self.nodes[idx]):
                    best_parent = idx
                    best_cost = cost_t
            
            ##add to tree
            new_idx = len(self.nodes)
            self.nodes.append(new_pt)
            self.parents.append(best_parent)
            self.cost.append(best_cost)

            ##rewire
            for idx in near_idxs:
                cost_t = self.cost[new_idx] + np.linalg.norm(self.nodes[idx] - new_pt)
                if cost_t < self.cost[idx] and self.collision(new_pt, self.nodes[idx]):
                    self.parents[idx] = new_idx
                    self.cost[idx] = cost_t

            ##check if goal reached and extract path
            if np.linalg.norm(new_pt - self.goal) < self.goal_r:
                goal_idx = new_idx
                self.nodes.append(self.goal)
                self.parents.append(goal_idx)
                self.cost.append(self.cost[goal_idx] + np.linalg.norm(self.nodes[goal_idx] - self.goal))
                path = self._extract_path(len(self.nodes)-1)
                break
        return path


# """Example utility module.

# Please use a file like this one to add extra functions.

# """

# def exampleFunction():
#     """Example of user-defined function.

#     """
#     x = -1
#     return x
