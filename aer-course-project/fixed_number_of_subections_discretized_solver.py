# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class SegmentCasadiSolver:
    def __init__(self, waypoints, waypoint_derivatives, waypoint_start_times, control_dt, maxSpeed, numSubsections):
        self.waypoints = waypoints
        self.Nw = waypoints.shape[0]
        total_time = waypoint_start_times[-1] - waypoint_start_times[0]
        self.total_time = total_time
        self.maxSpeed = maxSpeed
        self.numSubsections = numSubsections

        # Number of decision variables: number of segments times number of segment subsections
        N = (self.Nw - 1) * self.numSubsections

        print("total_time=",total_time)
        print("N=",N)
        # discretized times
        section_durations = waypoint_start_times[1:] - waypoint_start_times[0:-1]
        section_dts = (section_durations / numSubsections).reshape(-1,1)
        self.dts = (np.ones((self.Nw-1, numSubsections)) * section_dts).reshape(-1,)

        # section_dt_0145 = (section_durations * 0.1).reshape(-1,1)
        # section_dt_23 = (section_durations * 0.3).reshape(-1,1)

        subsection_dts_start = []
        subsection_dts_end = []
        # total = 0
        # multiplier = 1
        # while(total < 1):
            
        

        # section_dts = np.hstack([section_dt_0145,section_dt_0145,section_dt_23, section_dt_23, section_dt_0145, section_dt_0145]).reshape(-1,)
        # self.dts = section_dts

        print("section_dts", section_dts)
        print("self.dts", self.dts)

        self.N = N
        self.A4 = ca.SX.sym('A4', N)
        self.B4 = ca.SX.sym('B4', N)
        self.C4 = ca.SX.sym('C4', N)

        self.max_accel_xy = (2 * maxSpeed ) /( control_dt)
        self.max_accel_z = (2 * maxSpeed ) /( control_dt)

        self.min_time = 5

        self.max_jerk_xy = 2 * self.max_accel_xy / control_dt
        self.max_jerk_z = 2 * self.max_accel_z / control_dt

        self.max_snap = 2 * self.max_jerk_xy / control_dt


        print("self.dts.shape", self.dts.shape)
        print("self.A4.shape", self.A4.shape)

        snap_integral = 24**2 * ca.sum((self.A4 * self.dts)**2) + 24**2 *ca.sum((self.B4 * self.dts)**2) + 24**2 * ca.sum((self.C4 * self.dts)**2)



        A0, A1, A2, A3, A4 = unroll_coefficients(self.A4,waypoints[0,0],self.dts)
        B0, B1, B2, B3, B4 = unroll_coefficients(self.B4,waypoints[0,1],self.dts)
        C0, C1, C2, C3, C4 = unroll_coefficients(self.C4,waypoints[0,2],self.dts)

        # path_x = A0 + A1 * self.dts + A2 * self.dts**2 + A3 * self.dts**3 + A4 * self.dts**4 - A0[0]
        # path_y = B0 + B1 * self.dts + B2 * self.dts**2 + B3 * self.dts**3 + B4 * self.dts**4 - B0[0]
        # path_z = C0 + C1 * self.dts + C2 * self.dts**2 + C3 * self.dts**3 + C4 * self.dts**4 - C0[0]

        # path_sum = ca.sum(path_x**2 + path_y**2 + path_z**2)   
         



        g = []
        lb_vals = []
        ub_vals = []

        adjusted_start_times = waypoint_start_times - waypoint_start_times[0]
        print("adjusted_start_times.shape", adjusted_start_times.shape)
        print("waypoint_start_times.shape", waypoint_start_times.shape) 
        # x, xd, xdd, xddd = compute_state_1d(A0,A1,A2,A3,A4, self.dts)
        # y, yd, ydd, yddd = compute_state_1d(B0,B1,B2,B3,B4, self.dts)
        # z, zd, zdd, zddd = compute_state_1d(C0,C1,C2,C3,C4, self.dts)

        # velocity and acceleration constraint
        # g.append(ca.sqrt(xd**2 + yd**2 + zd**2))
        # g.append(ca.sqrt(xdd**2 + ydd**2 + zdd**2))

        total_waypoint_error = 0

        # Start point equality
        x0, xd0, xdd0, xddd0 = compute_state_1d(A0[0],A1[0],A2[0],A3[0],A4[0], 0)
        y0, yd0, ydd0, yddd0 = compute_state_1d(A0[0],A1[0],A2[0],A3[0],A4[0], 0)
        z0, zd0, zdd0, zddd0 = compute_state_1d(A0[0],A1[0],A2[0],A3[0],A4[0], 0)

        x0_error = x0 - A0[0]
        y0_error = y0 - B0[0]
        z0_error = z0 - C0[0]

        x0d_error = xd0 - A1[0]
        y0d_error = yd0 - B1[0]
        z0d_error = zd0 - C1[0]

        x0dd_error = xdd0 - 2*A2[0]
        z0dd_error = zdd0 - 2*B2[0]
        y0dd_error = ydd0 - 2*C2[0]

        x0ddd_error = xddd0 - 6*A3[0]
        y0ddd_error = yddd0 - 6*A3[0]
        z0ddd_error = zddd0 - 6*A3[0]

        g.append(x0_error)
        g.append(y0_error)
        g.append(z0_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        total_waypoint_error += x0_error**2 + y0_error**2 + z0_error**2

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

        # equality constraints for jerk
        g.append(x0ddd_error)
        g.append(y0ddd_error)
        g.append(z0ddd_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # End waypoint equality

        # End waypoint equality
        final_dt = self.dts[-1]
        xN, xdN, xddN, xdddN = compute_state_1d(A0[-1],A1[-1],A2[-1],A3[-1],A4[-1], final_dt)
        yN, ydN, yddN, ydddN = compute_state_1d(B0[-1],B1[-1],B2[-1],B3[-1],B4[-1], final_dt)
        zN, zdN, zddN, zdddN = compute_state_1d(C0[-1],C1[-1],C2[-1],C3[-1],C4[-1], final_dt)      
        
        xN_error = xN - waypoints[self.Nw-1,0]
        yN_error = yN - waypoints[self.Nw-1,1]
        zN_error = zN - waypoints[self.Nw-1,2]

        xNd_error = xdN - waypoint_derivatives[self.Nw-1, 0,0]
        yNd_error = ydN - waypoint_derivatives[self.Nw-1, 0,1]
        zNd_error = zdN - waypoint_derivatives[self.Nw-1, 0,2]

        xNdd_error = xddN - waypoint_derivatives[self.Nw-1, 1,0]
        yNdd_error = yddN - waypoint_derivatives[self.Nw-1, 1,1]
        zNdd_error = zddN - waypoint_derivatives[self.Nw-1, 1,2]

        xNddd_error = xdddN - waypoint_derivatives[self.Nw-1, 2,0]
        yNddd_error = ydddN - waypoint_derivatives[self.Nw-1, 2,1]
        zNddd_error = zdddN - waypoint_derivatives[self.Nw-1, 2,2]

        g.append(xN_error)
        g.append(yN_error)
        g.append(zN_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

        # equality constraint for speed
        g.append(xNd_error)
        g.append(yNd_error)
        g.append(zNd_error)
        for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

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

        for i in range(1,self.Nw-1):
            
            print("i=", i)
            print("numSubsections=", numSubsections)

            index = i * numSubsections - 1
            print("index=", index)
            print("self.dts.shape=", self.dts.shape)
            waypoint_dt = self.dts[index]

            print("Waypoint i=", i)
            print("discretized index=", index)

            print("waypoint time:", adjusted_start_times[i])
            print("polynomial coefficient time:", np.sum(self.dts[0:index+1]))
            print("waypoint_dt", waypoint_dt)
            print("A0.shape", A0.shape)

            # A0_i = A0[index]
            # A1_i = A1[index]
            # A2_i = A2[index]
            # A3_i = A3[index]
            # A4_i = A4[index]
            # B0_i = B0[index]
            # B1_i = B1[index]
            # B2_i = B2[index]
            # B3_i = B3[index]
            # B4_i = B4[index]
            # C0_i = C0[index]
            # C1_i = C1[index]
            # C2_i = C2[index]
            # C3_i = C3[index]
            # C4_i = C4[index]

            # wp_x, wp_xd, wp_xdd, wp_xddd= compute_state_1d(A0_i,A1_i,A2_i,A3_i,A4_i, waypoint_dt)
            # wp_y, wp_yd, wp_ydd, wp_yddd= compute_state_1d(B0_i,B1_i,B2_i,B3_i,B4_i, waypoint_dt)
            # wp_z, wp_zd, wp_zdd, wp_zddd= compute_state_1d(C0_i,C1_i,C2_i,C3_i,C4_i, waypoint_dt)
            

            # x_error = wp_x - waypoints[i,0]
            # y_error = wp_y - waypoints[i,1]
            # z_error = wp_z - waypoints[i,2]

            # total_waypoint_error += x_error**2 + y_error**2 + z_error**2

            # # equality constraint for waypoint positions
            # g.append(x_error)
            # g.append(y_error)
            # g.append(z_error)
            # for j in range(0,3): lb_vals.append(0); ub_vals.append(0)


            A0_i1 = A0[index+1]
            A1_i1 = A1[index+1]
            A2_i1 = A2[index+1]
            A3_i1 = A3[index+1]
            A4_i1 = A4[index+1]
            B0_i1 = B0[index+1]
            B1_i1 = B1[index+1]
            B2_i1 = B2[index+1]
            B3_i1 = B3[index+1]
            B4_i1 = B4[index+1]
            C0_i1 = C0[index+1]
            C1_i1 = C1[index+1]
            C2_i1 = C2[index+1]
            C3_i1 = C3[index+1]
            C4_i1 = C4[index+1]

            wp_x1, wp_xd1, wp_xdd1, wp_xddd1= compute_state_1d(A0_i1,A1_i1,A2_i1,A3_i1,A4_i1, 0)
            wp_y1, wp_yd1, wp_ydd1, wp_yddd1= compute_state_1d(B0_i1,B1_i1,B2_i1,B3_i1,B4_i1, 0)
            wp_z1, wp_zd1, wp_zdd1, wp_zddd1= compute_state_1d(C0_i1,C1_i1,C2_i1,C3_i1,C4_i1, 0)

            x_error_1 = wp_x1 - waypoints[i,0]
            y_error_1 = wp_y1 - waypoints[i,1]
            z_error_1 = wp_z1 - waypoints[i,2]

            g.append(x_error_1)
            g.append(y_error_1)
            g.append(z_error_1)
            for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            # g.append(wp_xd1 - wp_xd)
            # g.append(wp_yd1 - wp_yd)
            # g.append(wp_zd1 - wp_zd)
            # for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            # g.append(wp_xdd1 - wp_xdd)
            # g.append(wp_ydd1 - wp_ydd)
            # g.append(wp_zdd1 - wp_zdd)
            # for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            # g.append(wp_xddd1 - wp_xddd)
            # g.append(wp_yddd1 - wp_yddd)
            # g.append(wp_zddd1 - wp_zddd)
            # for j in range(0,3): lb_vals.append(0); ub_vals.append(0)

            g.append(wp_xd1)
            g.append(wp_yd1)
            g.append(wp_zd1)           

            lb_vals.append(-self.maxSpeed)
            lb_vals.append(-self.maxSpeed)
            lb_vals.append(-self.maxSpeed)

            ub_vals.append(self.maxSpeed)
            ub_vals.append(self.maxSpeed)
            ub_vals.append(self.maxSpeed)

            g.append(wp_xdd1)
            g.append(wp_ydd1)
            g.append(wp_zdd1)           

            lb_vals.append(-self.max_accel_xy)
            lb_vals.append(-self.max_accel_xy)
            lb_vals.append(-self.max_accel_z)

            ub_vals.append(self.max_accel_xy)
            ub_vals.append(self.max_accel_xy)
            ub_vals.append(self.max_accel_z)

            g.append(wp_xddd1)
            g.append(wp_yddd1)
            g.append(wp_zddd1)

            lb_vals.append(-self.max_jerk_xy)
            lb_vals.append(-self.max_jerk_xy)
            lb_vals.append(-self.max_jerk_xy)

            ub_vals.append(self.max_jerk_xy)
            ub_vals.append(self.max_jerk_xy)
            ub_vals.append(self.max_jerk_z)


        cost = snap_integral # + total_waypoint_error


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
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-5,
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

def evalute_polynomials_over_control_time_step(times, dt, n, freq, A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4):
    # x_vals = np.array([])
    # y_vals = np.array([])
    # z_vals = np.array([])
    p_vals = np.array([])
    v_vals = np.array([])
    a_vals = np.array([])
    j_vals = np.array([])

    for idx in range(0,n):
        duration = times[idx]
        nsample = int(duration * freq)
        ti = dt * np.arange(nsample)

        x, xd, xdd, xddd = compute_state_1d(A0[idx], A1[idx], A2[idx], A3[idx], A4[idx], ti)
        y, yd, ydd, yddd = compute_state_1d(B0[idx], B1[idx], B2[idx], B3[idx], B4[idx], ti)
        z, zd, zdd, zddd = compute_state_1d(C0[idx], C1[idx], C2[idx], C3[idx], C4[idx], ti)

        p = np.hstack([x, y, z])
        v = np.hstack([xd, yd, zd])
        a = np.hstack([xdd, ydd, zdd])
        j = np.hstack([xddd, yddd, zddd])


        # x_vals = np.concatenate([x_vals, np.array(x).flatten()])
        # y_vals = np.concatenate([y_vals, np.array(y).flatten()])
        # z_vals = np.concatenate([z_vals, np.array(z).flatten()])
        p_vals = p if(p_vals.size == 0) else np.concatenate([p_vals, p])
        v_vals = v if(v_vals.size == 0) else np.concatenate([v_vals, v])
        a_vals = a if(a_vals.size == 0) else np.concatenate([a_vals, a])
        j_vals = j if(j_vals.size == 0) else np.concatenate([j_vals, j])

    states = np.hstack([p_vals, v_vals, a_vals, j_vals])

    return states

def generate_trajectory(waypoints, averageSpeed, numSegmentSubsections, ctrl_freq):
    print("waypoints.shape=", waypoints.shape)

    Nw = waypoints.shape[0]

    p_prev = waypoints[0:Nw-1,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    total_length = np.sum(waypoint_min_lengths)
    total_time = total_length / averageSpeed
    segment_durations = waypoint_min_lengths / total_length * total_time

    maxSpeed = 4 *averageSpeed

    waypoint_desired_velocities = np.zeros(waypoints.shape)
    waypoint_desired_velocities[1:Nw-2,:] = np.inf

    waypoint_desired_accelerations = np.zeros(waypoints.shape)
    waypoint_desired_accelerations[1:Nw-2,:] = np.inf

    waypoint_desired_jerks = np.zeros(waypoints.shape)
    waypoint_desired_jerks[1:Nw-2,:] = np.inf

    waypoint_start_times = np.zeros((Nw,))
    for i in range(0,segment_durations.size):
        waypoint_start_times[i+1] = waypoint_start_times[i] + segment_durations[i]

    desired_derivatives = np.zeros((Nw, 3, 3))
    desired_derivatives[1:Nw-2,:,:] = np.inf
    desired_derivatives[-1,:,:] = 0

    solver = SegmentCasadiSolver(waypoints,desired_derivatives, waypoint_start_times, 1/60.0, maxSpeed, numSegmentSubsections)

    num_decision_variables = 3 * solver.N

    X0 = np.zeros((num_decision_variables,))
    
    A4, B4, C4, solution = solver.solve_coefficients(X0)

    print("A4=",A4)
    print("B4=",B4)
    print("C4=",C4)

    print(solution)

    A0, A1, A2, A3, A4 = unroll_coefficients(np.array(A4),waypoints[0,0],solver.dts)
    B0, B1, B2, B3, B4 = unroll_coefficients(np.array(B4),waypoints[0,1],solver.dts)
    C0, C1, C2, C3, C4 = unroll_coefficients(np.array(C4),waypoints[0,2],solver.dts)

    print("A4=",A4)
    print("B4=",B4)
    print("C4=",C4)

    ctrl_dt = 1/ctrl_freq

    N_discretized_segments = A0.shape[0]

    np.savez("combined_coefficients.npz", 
             A4=A4,
             B4=B4,
             C4=C4,
             times=solver.dts,
             waypoints=waypoints)

    print("A0.shape",A0.shape)

    states = evalute_polynomials_over_control_time_step( solver.dts, 0.001, N_discretized_segments, ctrl_freq, 
                                                                           A0, A1, A2, A3, A4, 
                                                                           B0, B1, B2, B3, B4, 
                                                                           C0, C1, C2, C3, C4)

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