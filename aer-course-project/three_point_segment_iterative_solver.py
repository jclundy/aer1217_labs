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
        self.max_accel_z = maxSpeed / 4

        self.max_jerk_xy = self.max_accel_xy / 4
        self.max_jerk_z = self.max_accel_z / 4

        self.max_snap = self.max_jerk_xy / 4

        snap_integral = 24**2 * ca.sum((self.A4 * self.dts)**2) + 24**2 *ca.sum((self.B4 * self.dts)**2) + 24**2 * ca.sum((self.C4 * self.dts)**2)


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
        # g.append(ca.sqrt(xd**2 + yd**2 + zd**2))
        # for j in range(0,3): lb_vals.append(0); ub_vals.append(self.maxSpeed)
        # g.append(ca.sqrt(xdd**2 + ydd**2 + zdd**2))
        # for j in range(0,3): lb_vals.append(0); ub_vals.append(self.max_accel_xy)
        # g.append(ca.sqrt(xddd**2 + yddd**2 + zddd**2))
        # for j in range(0,3): lb_vals.append(0); ub_vals.append(self.max_jerk_xy)

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
        # final_dt = adjusted_start_times[-1] - N * dt
        final_dt = self.dts[-1]
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
            waypoint_dt = self.dts[index]

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

def generate_trajectory(waypoints, averageSpeed, numSubsections, ctrl_freq, numWaypointsPerGroup, usePlot = False):
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
            delta = (waypoints[segmentEndIdx+1,:] - waypoints[segmentEndIdx,:])
            endVel = delta/np.linalg.norm(delta) * averageSpeed
            segmentEndDerivatives = np.ones((3,3)) * np.inf
            segmentEndDerivatives[0,:] = endVel
            # segmentEndDerivatives  = np.ones((3,3)) * np.inf


        segment_waypoints = waypoints[segmentStartIdx:segmentEndIdx+1,:]

        groupDuration = np.sum(segment_durations[segmentStartIdx:segmentEndIdx])

        print("duration i", segment_durations[segmentStartIdx])
        print("duration i+1", segment_durations[segmentStartIdx])
        # groupDuration = segment_durations[segmentStartIdx] + segment_durations[segmentStartIdx+1]

        segment_waypoint_start_times = waypoint_start_times[segmentStartIdx:segmentEndIdx+1]

        # discretization_dt = groupDuration / (numSegmentsInGroup*numSubsections)
        discretization_dt = 0.1

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