# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class SegmentCasadiSolver:
    def __init__(self, waypoints,waypoint_desired_velocities, waypoint_start_times, dt=1/60.0):
        self.waypoints = waypoints
        self.numWaypoints = waypoints.shape([0])
        total_time = waypoint_start_times[-1]
        self.total_time = total_time
        self.dt = dt

        # Number of decsion variables is total time / dt
        N = np.floor(total_time / self.dt).astype(np.int64)
        # discretized times
        self.dts = np.ones((N,1)) * dt
        last_dt = total_time - (N-1)*dt
        self.dts[N-1] = last_dt

        self.N = N
        self.A3 = ca.SX.sym('A3', N)
        self.B3 = ca.SX.sym('B3', N)
        self.C3 = ca.SX.sym('C3', N)

        self.max_accel_xy = 5
        self.max_accel_z = 0.517

        self.min_time = 5

        self.max_jerk_xy = 2 * self.max_accel_xy / self.dt * 0.01
        self.max_jerk_z = 2 * self.max_accel_z / self.dt * 0.01

        snap_integral = 24**2 * ca.sum((self.A4 * self.dts)**2) + 24**2 *ca.sum((self.B4 * self.dts)**2) + 24**2 * ca.sum((self.C4 * self.dts)**2)

        cost = snap_integral

        A0, A1, A2, A3, A4 = unroll_coefficients(self.A4,waypoints[0,0],self.dts)
        B0, B1, B2, B3, B4 = unroll_coefficients(self.B4,waypoints[0,1],self.dts)
        C0, C1, C2, C3, C4 = unroll_coefficients(self.C4,waypoints[0,2],self.dts)

        waypoint_discretized_indices = []
        waypoint_time_delta = []

        g = []
        # constraints on decision variables
        g.append(self.A4)
        g.append(self.B4)
        g.append(self.C4)

        for i in range(0,self.numWaypoints):
            index = np.floor(waypoint_start_times[i] / total_time * N).astype(np.int64) 
            waypoint_discretized_indices.append(index)
            waypoint_dt = waypoint_start_times[i] - index * dt
            waypoint_time_delta.append(waypoint_dt)

            A0_i = A0[index], A1_i = A1[index], A2_i = A2[index], A3_i = A3[index], A4_i = A4[index]
            B0_i = B0[index], B1_i = B1[index], B2_i = B2[index], B3_i = B3[index], B4_i = B4[index]
            C0_i = C0[index], C1_i = C1[index], C2_i = C2[index], C3_i = C3[index], C4_i = C4[index]
            wp_x, wp_xd, wp_xdd = compute_state_1d(A0_i,A1_i,A2_i,A3_i,A3_i, waypoint_dt)
            wp_y, wp_yd, wp_ydd = compute_state_1d(B0_i,B1_i,B2_i,B3_i,B3_i, waypoint_dt)
            wp_z, wp_zd, wp_zdd = compute_state_1d(C0_i,C1_i,C2_i,C3_i,C3_i, waypoint_dt)
            
            # equality constraint for waypoint positions

            x_error = wp_x - waypoints[i,0]
            y_error = wp_y - waypoints[i,1]
            z_error = wp_z - waypoints[i,2]

            g.append(x_error)
            g.append(y_error)
            g.append(z_error)
            
            # equality constraint for waypoint velocities
            xd_error = 0 if waypoint_desired_velocities[i,0] == np.inf else wp_xd - waypoint_desired_velocities[i,0]
            yd_error = 0 if waypoint_desired_velocities[i,1] == np.inf else wp_yd - waypoint_desired_velocities[i,1]
            zd_error = 0 if waypoint_desired_velocities[i,2] == np.inf else wp_zd - waypoint_desired_velocities[i,2]

            g.append(xd_error)
            g.append(yd_error)
            g.append(zd_error)

        opt_variables = ca.vertcat(
            ca.reshape(self.A3, -1, 1), 
            ca.reshape(self.B3, -1, 1), 
            ca.reshape(self.C3, -1, 1))

        a3_start = 0
        a3_end = a3_start + N
        b3_start = a3_end
        b3_end = b3_start + N
        c3_start = b3_end
        c3_end = c3_start + N 

        # number of constraints = num decision variables + num position constraints + num velocity constraints = 4 * N + 3*N + 3*N
        num_constraints = 4 * N + 3*N + 3*N
        lb = np.zeros(num_constraints)

        # A3 lower bound
        lb[a3_start:a3_end] = -self.max_jerk_xy 
        # B3 lower bound
        lb[b3_start:b3_end] = -self.max_jerk_xy
        # C3 lower bound
        lb[c3_start:c3_end] = -self.max_jerk_z
        # position and error lower bound
        lb[c3_end:] = 0

        ub = np.zeros(num_constraints)
        # A3 lower bound
        ub[a3_start:a3_end] = self.max_jerk_xy 
        # B3 lower bound
        ub[b3_start:b3_end] = self.max_jerk_xy
        # C3 lower bound
        ub[c3_start:c3_end] = self.max_jerk_z
        # position and velocity error upper bound
        ub[c3_end:] =  0

        self.lbg = lb
        self.ubg = ub

        opt_constraints = ca.vertcat(*g)

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
        t_total = X[0]

        coeffs = X[1:].reshape((-1,4))

        fracs = coeffs[:,0]
        A3 = coeffs[:,1]
        B3 = coeffs[:,2]
        C3 = coeffs[:,3]

        return t_total, fracs, A3, B3, C3, solution
    
    def solve_coefficients(self,initial_guess):
        solution = self.solver(x0 = initial_guess, lbg=self.lbg, ubg=self.ubg)
        X = solution['x']
        t_total = X[0]

        coeffs = X[1:].reshape((-1,4))

        fracs = coeffs[:,0]
        A3 = coeffs[:,1]
        B3 = coeffs[:,2]
        C3 = coeffs[:,3]

        return t_total, fracs, A3, B3, C3, solution
    

def compute_state_1d(P0, P1, P2, P3, P4, dt):
    # integrate position
    p = P0 + P1 * dt + P2*dt**2 + P3 * dt**3 + P4 * dt**4

    # integrate velocity
    pd = P1 + 2 * P2 * dt + 3 * P3 * dt**2 + 4 * P4 * dt**3

    # integrate acceleration
    pdd = 2 * P2 + 6 * P3 * dt + 12 * P4 * dt**2

    return p, pd, pdd


def unroll_coefficients(P4, WP0, times):
    n = times.shape[0] - 1

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