import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
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

    def optimize_segment_times():
        pass

        # todo: add limits for torques - constraints body rates

    """
    Decision variables:
    - times per segment
    - polynomial coefficients at each segment 3xN array [Ai....An; Bi...Bn; Ci....Cn]
    - A coefficeints: N x 4
    - B coefficients: N x 4
    - C coefficients: N x 4
    """

    def unwind_coefficients(self, times, A3, B3, C3):
        """
        inputs: 3rd order polynomials
        outputs: 
        """
        

    def objective_function(self, times, A, B,C):
        return np.sum(times)

    def equality_constraints(self, times, A, B,C):
        Ai0 = A[:,0]
        Ai1 = A[:,1]
        Ai2 = A[:,2]
        Ai3 = A[:,3]

        Bi0 = B[:,0]
        Bi1 = B[:,1]
        Bi2 = B[:,2]
        Bi3 = B[:,3]

        Ci0 = C[:,0]
        Ci1 = C[:,1]
        Ci2 = C[:,2]
        Ci3 = C[:,3]


        p_next_x = Ai0 + Ai1 * times + Ai2 * times**2 + Ai3 * times **3
        p_next_y = Bi0 + Bi1 * times + Bi2 * times**2 + Bi3 * times **3
        p_next_z = Ci0 + Ci1 * times + Ci2 * times**2 + Ci3 * times **3

        x_eq = self.waypoints[1:, 0] - p_next_x
        y_eq = self.waypoints[1:, 1] - p_next_y
        z_eq = self.waypoints[1:, 2] - p_next_z

        return np.concatenate(x_eq, y_eq, z_eq)
    
    def inequality_constraints(self, times, A,B,C):
        return None

    def linear_constraints(self, times, A,B,C):
        pass
        # linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

    def bounds(self, times):
        # Bounds(lb, ub)
        # bounds = Bounds([0, -0.5], [1.0, 2.0])
        t0 = np.zeros_like(times)
        tmax = np.ones_like(times) * np.inf
        A_min = np.ones_like(times) * -self.max_jerk_xy
        A_max = np.ones_like(times) * self.max_jerk_xy
        B_min = np.ones_like(times) * -self.max_jerk_xy
        B_max = np.ones_like(times) * self.max_jerk_xy
        C_min = np.ones_like(times) * -self.max_jerk_z
        C_max = np.ones_like(times) * self.max_jerk_z

        x_lower = np.concatenate([t0, A_min, B_min, C_min])
        x_upper = np.concatenate([tmax, A_max, B_max, C_max])
        bounds = Bounds(x_lower, x_upper)
        return bounds

# # Example code
# def rosen_with_args(x, a, b):
#     """The Rosenbrock function with additional arguments"""
#     return sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + b
# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen_with_args, x0, method='nelder-mead',
#                args=(0.5, 1.), options={'xatol': 1e-8, 'disp': True})

# from scipy.optimize import Bounds
# bounds = Bounds([0, -0.5], [1.0, 2.0])
# from scipy.optimize import LinearConstraint
# linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

# def cons_f(x):
#     return [x[0]**2 + x[1], x[0]**2 - x[1]]
# def cons_J(x):
#     return [[2*x[0], 1], [2*x[0], -1]]
# def cons_H(x, v):
#     return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])
# from scipy.optimize import NonlinearConstraint
# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

# x0 = np.array([0.5, 0])
# res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
#                constraints=[linear_constraint, nonlinear_constraint],
#                options={'verbose': 1}, bounds=bounds)
# print(res.x)
# """Sequential least squares"""
# ineq_cons = {'type': 'ineq',
#              'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
#                                          1 - x[0]**2 - x[1],
#                                          1 - x[0]**2 + x[1]]),
#              'jac' : lambda x: np.array([[-1.0, -2.0],
#                                          [-2*x[0], -1.0],
#                                          [-2*x[0], 1.0]])}
# eq_cons = {'type': 'eq',
#            'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
#            'jac' : lambda x: np.array([2.0, 1.0])}


# x0 = np.array([0.5, 0])
# res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
#                constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
#                bounds=bounds)
# print(res.x)

# def cons_f(x):
#     return [x[0]**2 + x[1], x[0]**2 - x[1]]
# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=BFGS())