import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1
"""
Optimize times
Inputs:
- waypoints: list of waypoints (Nx3) vector
- initial_info: dictionary of simulation parameters
Outputs:
- times: list of durations for each waypoint segment (N-1)
"""

class WaypointOptimizer():
    def __init__(self, waypoints, initial_info):
        self.ctrl_freq = initial_info["ctrl_freq"]
        self.max_speed_xy = initial_info["max_speed_xy"]
        self.max_speed_z = initial_info["max_speed_z"]
        self.max_acceleration_xy = initial_info["max_acceleration_xy"]
        self.max_acceleration_z = initial_info["max_acceleration_z"]
        self.dt = 1/self.ctrl_freq
        self.max_jerk_xy = 2*self.max_acceleration_xy/self.dt
        self.max_jerk_z = 2*self.max_acceleration_z/self.dt
        self.max_tilt = initial_info["max_tilt"]
        self.waypoints = waypoints

        num_waypoints = self.waypoints.shape[0]

        self.N = num_waypoints-1

    def optimize_segments(self, max_duration):
        t = np.arange(self.waypoints.shape[0])
        num_waypoints = self.waypoints.shape[0]

        N = num_waypoints-1

        objective_func = lambda x: self.objective_function(x, N)
        bounds = self.bounds()

        # eq_constraint_func = lambda x: self.eq_constraints(x,N)
        # nl_constraint_func = lambda x: self.ineq_constraints(x, N)

        # eq_conds = {'type': 'eq', 'fun': nl_constraint_func, 'jac':'2-point'}
        # ineq_conds = {'type': 'ineq', 'fun': eq_constraint_func, 'jac':'2-point'}


        p_prev = self.waypoints[0:num_waypoints-1,:]
        p_next = self.waypoints[1:,:]
        waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)
        total_length = np.sum(waypoint_min_lengths)
        t_initial = waypoint_min_lengths / total_length * max_duration

        A3_initial = np.zeros_like(t_initial)
        B3_initial = np.zeros_like(t_initial)
        C3_initial = np.zeros_like(t_initial)
        x0 = np.concatenate([t_initial, A3_initial, B3_initial, C3_initial])

        contraint_ub = self.constraint_ub(N)
        contraint_lb = self.constraint_lb(N)

        constraint_func = lambda x: self.nl_constraints(x,N)
        nonl_constraints = NonlinearConstraint(constraint_func,contraint_lb, contraint_ub, jac='3-point', hess=BFGS())

        res = minimize(objective_func, x0, method='SLSQP',  jac='3-point',
               constraints=nonl_constraints, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)


        # res = minimize(objective_func, x0, method='trust-constr',  jac='2-point', hess=SR1(),
        #        constraints=nonl_constraints, options={'disp': True},
        #        bounds=bounds)

        return res

    def bounds(self):
        # Bounds(lb, ub)
        # bounds = Bounds([0, -0.5], [1.0, 2.0])

        # compute waypoint distances
        num_waypoints = self.waypoints.shape[0]
        N = num_waypoints - 1
        p_prev = self.waypoints[0:num_waypoints-1,:]
        p_next = self.waypoints[1:,:]
        waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

        # times must also meet speed constraints
        t_min = waypoint_min_lengths / (self.max_speed_xy)
        # times must be gt eq zero
        t_max = np.ones_like(t_min) * np.inf

        # times must be gt eq zero
        A3_min = np.ones_like(t_min) * - np.inf
        B3_min = np.ones_like(t_min) * - np.inf
        C3_min = np.ones_like(t_min) * - np.inf
        A3_max = np.ones_like(t_min) * np.inf
        B3_max = np.ones_like(t_min) * np.inf
        C3_max = np.ones_like(t_min) * np.inf

        lb = np.concatenate([t_min, A3_min, B3_min, C3_min]).flatten()
        ub = np.concatenate([t_max, A3_max, B3_max, C3_max]).flatten()

        bounds = Bounds(lb, ub)
        return bounds

    def nl_constraints(self, X, n):
        times = X[0:n]
        A3 = X[n:2*n]
        B3 = X[2*n:3*n]
        C3 = X[3*n:4*n]

        Ai, Bi, Ci = self.unwind_coefficients(times, A3, B3, C3)
        A0 = Ai[:,0]
        A1 = Ai[:,0]
        A2 = Ai[:,0]

        B0 = Bi[:,0]
        B1 = Bi[:,0]
        B2 = Bi[:,0]

        C0 = Ci[:,0]
        C1 = Ci[:,0]
        C2 = Ci[:,0]
        # constraint 1: acceleration constraint
        xdd = 2 * A2 + 6 * A3 * times
        ydd = 2 * B2 + 6 * B3 * times
        zdd = 2 * C2 + 6 * C3 * times
        xydd = np.sqrt(xdd**2 + ydd**2)

        xy_acceleration_constraint = self.max_acceleration_xy - xydd
        z_acceleration_constraint = self.max_acceleration_z - zdd

        # constraint 2: velocity constraint
        xd = A1 + 2 * A2 * times + 3 * A3 * times**2
        yd = B1 + 2 * B2 * times + 3 * B3 * times**2
        zd = C1 + 2 * C2 * times + 3 * C3 * times**2
        xyd = np.sqrt(xd**2 + yd**2)

        xy_velocity_constraint = self.max_speed_xy - np.abs(xyd)
        z_velocity_constraint = self.max_speed_z - np.abs(zd)

        eq0x = A0[0] - self.waypoints[0,0]
        eq0y = B0[0] - self.waypoints[0,1]
        eq0z = C0[0] - self.waypoints[0,2]

        # rest of positions
        x = A0 + A1 * times + A2 * times**2 + A3 * times**3
        y = B0 + B1 * times + B2 * times**2 + B3 * times**3
        z = C0 + C1 * times + C2 * times**2 + C3 * times**3

        eqnx = x - self.waypoints[1:,0]
        eqny = y - self.waypoints[1:,1]
        eqnz = z - self.waypoints[1:,2]

        # last velocity is zero
        xd = A1 + 2 * A2 * times + 3 * A3 * times**2
        yd = B1 + 2 * B2 * times + 3 * B3 * times**2
        zd = C1 + 2 * C2 * times + 3 * C3 * times**2

        eqn_xd = xd[n-1]
        eqn_yd = yd[n-1]
        eqn_zd = zd[n-1]

        # eqnx = np.insert(eqnx, 0, eq0x)
        # eqny = np.insert(eqny, 0, eq0y)
        # eqnz =np.insert(eqny, 0, eq0z)

        # additional_eq_constraints = np.array([eq0x, eq0y, eq0z, eqn_xd, eqn_yd, eqn_zd]).reshape(-1,1)
        # position_eq_constraints = np.concatenate([ eqnx, eqny, eqnz]).reshape(-1,1)      

        return np.concatenate([xy_acceleration_constraint, z_acceleration_constraint, 
                               xy_velocity_constraint, z_velocity_constraint,
                               eqnx, eqny, eqnz]).flatten()

    def constraint_lb(self, n):
        num_constraints = 7*n

        lb = np.ones((1, num_constraints)) * 0
        return lb.flatten()

    def constraint_ub(self, n):
        num_constraints = 7*n

        ub = np.ones((1, num_constraints)) * np.inf
        # equality constraints
        ub[4*num_constraints:] = 0
        return ub.flatten()


    # def eq_constraints(self,X,n):
    #     times = X[0:n]
    #     A3 = X[n:2*n]
    #     B3 = X[2*n:3*n]
    #     C3 = X[3*n:4*n]

    #     Ai, Bi, Ci = self.unwind_coefficients(times, A3, B3, C3)
    #     A0 = Ai[:,0]
    #     A1 = Ai[:,0]
    #     A2 = Ai[:,0]

    #     B0 = Bi[:,0]
    #     B1 = Bi[:,0]
    #     B2 = Bi[:,0]

    #     C0 = Ci[:,0]
    #     C1 = Ci[:,0]
    #     C2 = Ci[:,0]

    #     constraints = []

    #     # Position constraints
    #     x = A0 + A1 * times + A2 * times**2 + A3 * times**3
    #     y = B0 + B1 * times + B2 * times**2 + B3 * times**3
    #     z = C0 + C1 * times + C2 * times**2 + C3 * times**3

    #     # first position
    #     eq0x = A0[0] - self.waypoints[0,0]
    #     eq0y = B0[0] - self.waypoints[0,1]
    #     eq0z = C0[0] - self.waypoints[0,2]

    #     # rest of positions
    #     eqnx = x - self.waypoints[1:,0]
    #     eqny = y - self.waypoints[1:,1]
    #     eqnz = z - self.waypoints[1:,2]

    #     # last velocity is zero
    #     xd = A1 + 2 * A2 * times + 3 * A3 * times**2
    #     yd = B1 + 2 * B2 * times + 3 * B3 * times**2
    #     zd = C1 + 2 * C2 * times + 3 * C3 * times**2

    #     eqn_xd = xd[n-1]
    #     eqn_yd = yd[n-1]
    #     eqn_zd = zd[n-1]

    #     additional_eq_constraints = np.array([eq0x, eq0y, eq0z, eqn_xd, eqn_yd, eqn_zd]).reshape(-1,1)
    #     position_eq_constraints = np.concatenate([ eqnx, eqny, eqnz]).reshape(-1,1)
    #     constraints = np.append(additional_eq_constraints,eq0x).reshape(-1,1)
    #     constraints = np.append(additional_eq_constraints,eq0y).reshape(-1,1)
    #     return constraints

    def unwind_coefficients(self, times, A3, B3, C3):
        
        n = times.shape[0]
        M = np.zeros((n,n))
        M[1:,:] = np.tril(np.ones((n-1,n)))

        A2 = 3 * M @ (A3 * times)
        B2 = 3 * M @ (B3 * times)
        C2 = 3 * M @ (C3 * times)

        A1 = 2 * M @ (A2 * times) + 3 * M @ (A3 * times **2)
        B1 = 2 * M @ (B2 * times) + 3 * M @ (B3 * times **2)
        C1 = 2 * M @ (C2 * times) + 3 * M @ (C3 * times **2)

        A1 = 2 * M @ (A2 * times) + 3 * M @ (A3 * times **2)
        B1 = 2 * M @ (B2 * times) + 3 * M @ (B3 * times **2)
        C1 = 2 * M @ (C2 * times) + 3 * M @ (C3 * times **2)

        A0 = M @ (A1 * times) + M @ (A2 * times **2) + M @ (A3 * times **3)
        A0 = A0 + self.waypoints[0][0]

        B0 = M @ (B1 * times) + M @ (B2 * times **2) + M @ (B3 * times **3)
        B0 = B0 + self.waypoints[0][1]

        C0 = M @ (C1 * times) + M @ (C2 * times **2) + M @ (C3 * times **3)
        C0 = C0 + self.waypoints[0][2]

        Ai = np.concatenate([A0, A1, A2, A3]).reshape(-1,4)
        Bi = np.concatenate([B0, B1, B2, B3]).reshape(-1,4)
        Ci = np.concatenate([C0, C1, C2, C3]).reshape(-1,4)                
        """
        inputs: 3rd order polynomials
        outputs:
        """
        return Ai, Bi, Ci


    def objective_function(self, X,n):
        times = X[0:n]
        A3 = X[n:2*n]
        B3 = X[2*n:3*n]
        C3 = X[3*n:4*n]
        return np.sum(times**2)

    # def equality_constraints(self, times, A, B,C):
    #     Ai0 = A[:,0]
    #     Ai1 = A[:,1]
    #     Ai2 = A[:,2]
    #     Ai3 = A[:,3]

    #     Bi0 = B[:,0]
    #     Bi1 = B[:,1]
    #     Bi2 = B[:,2]
    #     Bi3 = B[:,3]

    #     Ci0 = C[:,0]
    #     Ci1 = C[:,1]
    #     Ci2 = C[:,2]
    #     Ci3 = C[:,3]


    #     p_next_x = Ai0 + Ai1 * times + Ai2 * times**2 + Ai3 * times **3
    #     p_next_y = Bi0 + Bi1 * times + Bi2 * times**2 + Bi3 * times **3
    #     p_next_z = Ci0 + Ci1 * times + Ci2 * times**2 + Ci3 * times **3

    #     x_eq = self.waypoints[1:, 0] - p_next_x
    #     y_eq = self.waypoints[1:, 1] - p_next_y
    #     z_eq = self.waypoints[1:, 2] - p_next_z

    #     return np.concatenate(x_eq, y_eq, z_eq)

    # def inequality_constraints(self, times, A,B,C):
    #     return None

    # def linear_constraints(self, times, A,B,C):
    #     pass
    #     # linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])



def generate_waypoints(startPos, endPos):
    poses = []
    poses.append((startPos[0], startPos[1], startPos[2]))
    poses.append((-0.25, -1.0, 0.25))
    poses.append((-0.5, -2.0, 0.5))
    poses.append((-0.5, -3.0, 0.5))
    poses.append((-0.5, -4.0, 0.5))
    poses.append((-0.5, -5.0, 0.5))
    poses.append((-0.5, -3.0, 2.0))
    poses.append((-0.5, -2.0, 2.0))
    poses.append((-0.5, -1.0, 2.0))
    poses.append((-0.5,  0.0, 2.0))
    poses.append((-0.5,  1.0, 2.0))
    poses.append((-0.5,  2.0, 2.0))
    poses.append((-0.5,  1.0, 1.5))
    poses.append((-0.5,  0.5, 1.0))
    poses.append((-0.25,  0.25, 0.5))
    poses.append([endPos[0], endPos[1], endPos[2]])
    return np.array(poses)

def test():

    # Load configuration.

    initial_pos = [0,0,0]
    end_pos = [0,0,0]

    freq = 60
    dt = 1/freq
    data = {}
    data["ctrl_freq"] = freq
    data["max_speed_xy"] = 2
    data["max_acceleration_xy"] = 11.2
    data["max_speed_z"] = 2
    data["max_acceleration_z"] = 0.517
    data["max_jerk_xy"] = 2*data["max_acceleration_xy"] / dt
    data["max_jerk_z"] = 2*data["max_acceleration_z"]/ dt
    data["max_tilt"] = 1.46 * np.pi / 180.0 # radians
    waypoints = generate_waypoints(initial_pos, end_pos)

    optimizer = WaypointOptimizer(waypoints, data)
    max_time = 360
    res = optimizer.optimize_segments(max_time)

    print(res)

    print("------------------------------------")
    print(res.x.shape)
    print(res.x[0:optimizer.N])
    print(np.sum(res.x[0:optimizer.N]))

    return

if __name__ == "__main__":
    test()