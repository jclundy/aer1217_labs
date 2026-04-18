# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class SegmentCasadiSolver:
    def __init__(self, waypoints, waypoint_derivatives, waypoint_start_times, dt=1/60.0, maxSpeed = 2):
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
        last_dt = total_time - (N-1)*dt
        self.dts[N-1] = last_dt

        self.N = N
        self.A4 = ca.SX.sym('A4', N)
        self.B4 = ca.SX.sym('B4', N)
        self.C4 = ca.SX.sym('C4', N)

        self.max_accel_xy = 2 * maxSpeed / self.dt
        self.max_accel_z = 0.517

        self.min_time = 5

        self.max_jerk_xy = 2 * self.max_accel_xy / self.dt
        self.max_jerk_z = 2 * self.max_accel_z / self.dt

        self.max_snap = 2 * self.max_jerk_xy / self.dt

        snap_integral = 24**2 * ca.sum((self.A4 * self.dts)**2) + 24**2 *ca.sum((self.B4 * self.dts)**2) + 24**2 * ca.sum((self.C4 * self.dts)**2)


        A0, A1, A2, A3, A4 = unroll_coefficients(self.A4,waypoints[0,0],self.dts)
        B0, B1, B2, B3, B4 = unroll_coefficients(self.B4,waypoints[0,1],self.dts)
        C0, C1, C2, C3, C4 = unroll_coefficients(self.C4,waypoints[0,2],self.dts)

        trajectory_integral = ca.sum(A0 * self.dts + 0.5 * A1 * self.dts**2 + 1/3 * A2 * self.dts**3 + 0.25 * A3 * self.dts**4 + 1/5 * A4 * self.dts**5)

        cost = snap_integral #trajectory_integral

        g = []
        lb_vals = []
        ub_vals = []
        # constraints on decision variables
        # g.append(self.A4)
        # g.append(self.B4)
        # g.append(self.C4)



        adjusted_start_times = waypoint_start_times - waypoint_start_times[0]
        print("adjusted_start_times.shape", adjusted_start_times.shape)
        print("waypoint_start_times.shape", waypoint_start_times.shape) 
        x, xd, xdd, xddd = compute_state_1d(A0,A1,A2,A3,A4, self.dts)
        y, yd, ydd, yddd = compute_state_1d(B0,B1,B2,B3,B4, self.dts)
        z, zd, zdd, zddd = compute_state_1d(C0,C1,C2,C3,C4, self.dts)

        # velocity and acceleration constraint
        # g.append(ca.sqrt(xd**2 + yd**2 + zd**2))
        # g.append(ca.sqrt(xdd**2 + ydd**2 + zdd**2))



        # Start point equality
        x0_error = x[0] - A0[0]
        y0_error = y[0] - B0[0]
        z0_error = z[0] - C0[0]

        x0d_error = xd[0] - A1[0]
        y0d_error = yd[0] - B1[0]
        z0d_error = zd[0] - C1[0]

        x0dd_error = xdd[0] - 2*A2[0]
        z0dd_error = zdd[0] - 2*B2[0]
        y0dd_error = ydd[0] - 2*C2[0]

        x0ddd_error = xddd[0] - 6*A3[0]
        y0ddd_error = yddd[0] - 6*A3[0]
        z0ddd_error = zddd[0] - 6*A3[0]

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
        # g.append(x0ddd_error)
        # g.append(y0ddd_error)
        # g.append(z0ddd_error)
        # equality_constraints += 12

        # End waypoint equality
        xN_error = x[-1] - waypoints[-1,0]
        yN_error = y[-1] - waypoints[-1,1]
        zN_error = z[-1] - waypoints[-1,2]

        print("waypoints[-1,:]",waypoints[-1,:])

        xNd_error = xd[-1] - waypoint_derivatives[-1, 0,0]
        yNd_error = yd[-1] - waypoint_derivatives[-1, 0,1]
        zNd_error = zd[-1] - waypoint_derivatives[-1, 0,2]

        xNdd_error = xdd[-1] - waypoint_derivatives[-1, 1,0]
        yNdd_error = ydd[-1] - waypoint_derivatives[-1, 1,1]
        zNdd_error = zdd[-1] - waypoint_derivatives[-1, 1,2]

        xNddd_error = xddd[-1] - waypoint_derivatives[-1, 2,0]
        yNddd_error = yddd[-1] - waypoint_derivatives[-1, 2,1]
        zNddd_error = zddd[-1] - waypoint_derivatives[-1, 2,2]

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

        for i in range(1,self.Nw-1):
            
            index = np.floor(adjusted_start_times[i] / self.dt).astype(np.int64) 
            waypoint_dt = adjusted_start_times[i] - index * dt

            print("Waypoint i=", i)
            print("discretized index=", index)

            print("waypoint time:", adjusted_start_times[i])
            print("polynomial coefficient time:", index * dt)
            print("waypoint_dt", waypoint_dt)

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

def evalute_polynomials_over_control_time_step(times, dt, n, freq, A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4):
    x_vals = np.array([])
    y_vals = np.array([])
    z_vals = np.array([])
    t_vals = np.array([])

    prev_duration = 0
    for idx in range(0,n):
        duration = times[idx]
        nsample = int(duration * freq)
        ti = dt * np.arange(nsample)

        xi = A0[idx] + A1[idx] * ti + A2[idx] * ti**2 + A3[idx] * ti**3 + A4[idx] * ti**4
        yi = B0[idx] + B1[idx] * ti + B2[idx] * ti**2 + B3[idx] * ti**3 + B4[idx] * ti**4
        zi = C0[idx] + C1[idx] * ti + C2[idx] * ti**2 + C3[idx] * ti**3 + C4[idx] * ti**4

        # ti_total = np.array(ti) + prev_duration

        # print("nsample",nsample)
        # print("ti.shape",ti.shape)
        # print("ti_total.shape",ti_total.shape)
        # print("t_vals.shape",t_vals.shape)

        # t_vals = np.concatenate([t_vals, ti_total.flatten()])

        # print("x_vals.shape", x_vals.shape)
        # print("xi.shape", xi.shape)

        x_vals = np.concatenate([x_vals, np.array(xi).flatten()])
        y_vals = np.concatenate([y_vals, np.array(yi).flatten()])
        z_vals = np.concatenate([z_vals, np.array(zi).flatten()])
        prev_duration = duration

    # t_array = t_vals.flatten()
    x_array = np.array(x_vals).flatten()
    y_array = np.array(y_vals).flatten()
    z_array = np.array(z_vals).flatten()

    # return t_array, x_array, y_array, z_array
    return x_array, y_array, z_array

def generate_waypoints():
    poses = [[-1.0, -3.0, 1.0], 
            [-0.09999980975910072, -2.49952220397356, 1.0], 
            [0.5, -2.5, 1.0], 
            [2.0, -2.1, 1.0], 
            [2.0, -1.5, 1.0], 
            [1.2999999048795503, -0.64976110198678, 1.0], 
            [0.5999998097591007, 0.20047779602643997, 1.0], 
            [0.0, 0.2, 1.0], 
            [-0.5, 0.9, 1.0], 
            [-0.5, 1.5, 1.0], 
            [-0.5, 2.0, 1.0]]

    # poses = [[-1.0, -3.0, 1.0], 
    #         [-0.09999980975910072, -2.49952220397356, 1.0], 
    #         [0.5, -2.5, 1.0], 
    #         [2.0, -2.1, 1.0]]

    # poses = [[-1.0, -3.0, 1.0], 
    #         [-0.09999980975910072, -2.49952220397356, 1.0], 
    #         [0.5, -2.5, 1.0]]

    # poses = [[-1.0, -3.0, 1.0], 
    #         [0.5, -2.5, 1.0]]


    return np.array(poses).reshape(-1,3)

def main():
    waypoints = generate_waypoints()
    print("waypoints=", waypoints.reshape(-1,3))


    Nw = waypoints.shape[0]
    max_time = 15.0

    p_prev = waypoints[0:Nw-1,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    print("waypoint_min_lengths=",waypoint_min_lengths.reshape(1,-1))

    total_length = np.sum(waypoint_min_lengths)
    segment_durations = waypoint_min_lengths / total_length * max_time

    maxSpeed = 2 * total_length / max_time


    dt = 1/5

    waypoint_desired_velocities = np.zeros(waypoints.shape)
    waypoint_desired_velocities[1:Nw-2,:] = np.inf

    waypoint_desired_accelerations = np.zeros(waypoints.shape)
    waypoint_desired_accelerations[1:Nw-2,:] = np.inf

    waypoint_desired_jerks = np.zeros(waypoints.shape)
    waypoint_desired_jerks[1:Nw-2,:] = np.inf

    waypoint_start_times = np.zeros((Nw,))
    for i in range(0,segment_durations.size):
        waypoint_start_times[i+1] = waypoint_start_times[i] + segment_durations[i]

    print("waypoint_start_times", waypoint_start_times)
    ax0 = plt.figure().add_subplot(projection='3d')

    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    ax0.scatter(wx, wy, wz, marker='o')

    desired_derivatives = np.zeros((Nw, 3, 3))
    desired_derivatives[1:Nw-2,:,:] = np.inf

    solver = SegmentCasadiSolver(waypoints,desired_derivatives, waypoint_start_times, dt, maxSpeed)

    n = solver.N
    a3_start = 0
    a3_end = a3_start + n
    b3_start = a3_end
    b3_end = b3_start + n
    c3_start = b3_end
    c3_end = c3_start + n 

    num_decision_variables = 3 * n

    X0 = np.zeros((num_decision_variables,))
    X0[a3_start:a3_end] = 0 
    # B3 lower bound
    X0[b3_start:b3_end] = 0
    # C3 lower bound
    X0[c3_start:c3_end] = 0
    
    A4, B4, C4, solution = solver.solve_coefficients(X0)

    A0, A1, A2, A3, A4 = unroll_coefficients(np.array(A4),waypoints[0,0],solver.dts)
    B0, B1, B2, B3, B4 = unroll_coefficients(np.array(B4),waypoints[0,1],solver.dts)
    C0, C1, C2, C3, C4 = unroll_coefficients(np.array(C4),waypoints[0,2],solver.dts)

    tn = solver.dts[n-1]
    xn, xdn, xddn, xdddn = compute_state_1d(A0[n-1],A1[n-1],A2[n-1],A3[n-1],A4[n-1], tn)
    yn, ydn, yddn, ydddn = compute_state_1d(B0[n-1],B1[n-1],B2[n-1],B3[n-1],B4[n-1], tn)
    zn, zdn, zddn, zdddn = compute_state_1d(C0[n-1],C1[n-1],C2[n-1],C3[n-1],C4[n-1], tn)

    # ax0.scatter(A0_vals, B0_vals, C0_vals, marker='^')

    # plot trajectory of quadrotor evalutaed at every timestep

    plot_freq = 60.0
    plot_dt = 1/plot_freq
    # plot_times = np.ones_like(A0_vals) * plot_dt
    np.savez("combined_coefficients.npz", 
             A4=A4.reshape(-1,1), 
             B4=B4.reshape(-1,1), 
             C4=C4.reshape(-1,1))

    # N_sections = A0.shape[0]
    x_array, y_array, z_array = evalute_polynomials_over_control_time_step(segment_durations, plot_dt, Nw-1, plot_freq, 
                                                                           A0, A1, A2, A3, A4, 
                                                                           B0, B1, B2, B3, B4, 
                                                                           C0, C1, C2, C3, C4)

    ax0.plot(A0[0:-1], B0[0:-1], C0[0:-1])

    
    midpoint_idx = np.floor(waypoint_start_times[1:-1] /  max_time * A0.shape[0])

    print("A0.shape", A0.shape)
    print("sum(solver.dts)=", np.sum(solver.dts))

    A0_vals = np.concatenate([A0[0],A0[midpoint_idx], A0[-1]])
    B0_vals = np.concatenate([B0[0],B0[midpoint_idx], B0[-1]])
    C0_vals = np.concatenate([C0[0],C0[midpoint_idx], C0[-1]])

    print("A0_vals", A0_vals.reshape(1,-1))
    print("B0_vals", B0_vals.reshape(1,-1))
    print("C0_vals", C0_vals.reshape(1,-1))

    ax0.scatter(A0_vals, B0_vals, C0_vals, marker="^")
    # ax0.plot(x_array,y_array,z_array)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    plt.save("smoothed_trajectory.png")



if __name__ == "__main__":
    main()