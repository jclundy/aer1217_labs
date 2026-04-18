# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class SegmentCasadiSolver:
    def __init__(self, waypoints,waypoint_desired_velocities, waypoint_start_times, dt=1/60.0, maxSpeed = 2):
        self.waypoints = waypoints
        self.Nw = waypoints.shape[0]
        total_time = waypoint_start_times[-1] - waypoint_start_times[0]
        self.total_time = total_time
        self.dt = dt
        self.maxSpeed = 10 * maxSpeed

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

        self.max_accel_xy = 5
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

        waypoint_discretized_indices = []
        waypoint_time_delta = []

        g = []
        # constraints on decision variables
        g.append(self.A4)
        g.append(self.B4)
        g.append(self.C4)

        equality_constraints = 0

        adjusted_start_times = np.zeros((N+1,))
        adjusted_start_times[1:] = waypoint_start_times[1:] - waypoint_start_times[0:-1]

        for i in range(0,self.Nw):
            index = np.floor(adjusted_start_times[i] / total_time * (N-1)).astype(np.int64) 
            waypoint_discretized_indices.append(index)
            waypoint_dt = adjusted_start_times[i] - index * dt
            waypoint_time_delta.append(waypoint_dt)

            print("i=",i)
            print("index=",index)

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
            wp_x, wp_xd, wp_xdd = compute_state_1d(A0_i,A1_i,A2_i,A3_i,A4_i, waypoint_dt)
            wp_y, wp_yd, wp_ydd = compute_state_1d(B0_i,B1_i,B2_i,B3_i,B4_i, waypoint_dt)
            wp_z, wp_zd, wp_zdd = compute_state_1d(C0_i,C1_i,C2_i,C3_i,C4_i, waypoint_dt)
            
            # equality constraint for waypoint positions

            x_error = wp_x - waypoints[i,0]
            y_error = wp_y - waypoints[i,1]
            z_error = wp_z - waypoints[i,2]

            g.append(x_error)
            g.append(y_error)
            g.append(z_error)

            # inequality constraint for speed
            # wp_v = ca.sqrt(wp_xd**2 + wp_yd**2 + wp_zd**2)
            g.append(wp_xd)
            g.append(wp_yd)
            g.append(wp_zd)          

            # equality constraint for waypoint velocities #TODO - add acceleration, jerk
            if (i == 0 or i == self.Nw-1):
                xd_error = wp_xd - waypoint_desired_velocities[i,0]
                yd_error = wp_yd - waypoint_desired_velocities[i,1]
                zd_error = wp_zd - waypoint_desired_velocities[i,2]

                if(waypoint_desired_velocities[i,0] != np.inf):
                    g.append(xd_error)
                    equality_constraints +=1
                if(waypoint_desired_velocities[i,0] != np.inf):
                    g.append(yd_error)
                    equality_constraints +=1
                if(waypoint_desired_velocities[i,0] != np.inf):
                    g.append(zd_error)
                    equality_constraints +=1

        opt_variables = ca.vertcat(
            ca.reshape(self.A4, -1, 1), 
            ca.reshape(self.B4, -1, 1), 
            ca.reshape(self.C4, -1, 1))

        opt_constraints = ca.vertcat(*g)

        print("opt_constraints.shape", opt_constraints.shape)

        a3_start = 0
        a3_end = a3_start + N
        b3_start = a3_end
        b3_end = b3_start + N
        c3_start = b3_end
        c3_end = c3_start + N 
        v_norm_start = c3_end
        v_norm_end = v_norm_start + 3 * self.Nw 

        # number of constraints = num decision variables + num position constraints + velocity bound + num velocity eq constraints = 4 * N + 3*N + 3*N
        num_constraints = 3 * N + 3*self.Nw + 3 * self.Nw + equality_constraints #3*2

        print("num_constraints=", num_constraints)
        lb = np.zeros(num_constraints)

        # A3 lower bound
        lb[a3_start:a3_end] = -self.max_snap 
        # B3 lower bound
        lb[b3_start:b3_end] = -self.max_snap
        # C3 lower bound
        lb[c3_start:c3_end] = -self.max_snap
        # velocity norm lower bound
        lb[v_norm_start:v_norm_end] = -self.maxSpeed
        # position and error lower bound
        lb[v_norm_end:] = 0


        ub = np.zeros(num_constraints)
        # A3 lower bound
        ub[a3_start:a3_end] = self.max_snap 
        # B3 lower bound
        ub[b3_start:b3_end] = self.max_snap
        # C3 lower bound
        ub[c3_start:c3_end] = self.max_snap
        # velocity norm upper bound
        ub[v_norm_start:v_norm_end] = self.maxSpeed
        # position and velocity error upper bound
        ub[v_norm_end:] =  0

        self.lbg = lb
        self.ubg = ub

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

    return p, pd, pdd


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


def evalute_polynomials_over_control_time_step(dt,A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4):
    x_vals = np.array([])
    y_vals = np.array([])
    z_vals = np.array([])
    # t_vals = np.array([])

    x_vals = A0 + A1 * dt + A2 * dt**2 + A3 * dt**3 + A4 * dt**4
    y_vals = B0 + B1 * dt + B2 * dt**2 + B3 * dt**3 + B4 * dt**4
    z_vals = C0 + C1 * dt + C2 * dt**2 + C3 * dt**3 + C4 * dt**4

    x_array = np.array(x_vals).flatten()
    y_array = np.array(y_vals).flatten()
    z_array = np.array(z_vals).flatten()

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

    return np.array(poses).reshape(-1,3)

def main():
    waypoints = generate_waypoints()
    print("waypoints=", waypoints.reshape(-1,3))


    Nw = waypoints.shape[0]
    max_time = 60

    p_prev = waypoints[0:Nw-1,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    print("waypoint_min_lengths=",waypoint_min_lengths.reshape(1,-1))

    total_length = np.sum(waypoint_min_lengths)
    durations = waypoint_min_lengths / total_length * max_time

    maxSpeed = 2 * total_length / max_time


    dt = 0.5

    waypoint_desired_velocities = np.zeros(waypoints.shape)
    waypoint_desired_velocities[1:Nw-2,:] = np.inf
    waypoint_start_times = np.zeros((Nw,))
    for i in range(0,durations.size):
        waypoint_start_times[i+1] = waypoint_start_times[i] + durations[i]

    print("waypoint_start_times", waypoint_start_times)
    ax0 = plt.figure().add_subplot(projection='3d')

    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    ax0.scatter(wx, wy, wz, marker='o')

    A0_vals = np.array([])
    B0_vals = np.array([])
    C0_vals = np.array([])

    A4_vals = np.array([])
    B4_vals = np.array([])
    C4_vals = np.array([])

    for i in range(0, Nw-1):

        print("Solving Segment ", i)

        segment_waypoints = waypoints[i:i+2,:]
        segment_velocities = waypoint_desired_velocities[i:i+2,:]
        segment_waypoint_times = waypoint_start_times[i:i+2]
        print(segment_waypoints)
        print(segment_velocities)
        print(segment_waypoint_times)
        solver = SegmentCasadiSolver(segment_waypoints,segment_velocities, segment_waypoint_times, dt, maxSpeed)

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

        A4_vals = np.concatenate([A4_vals, np.array(A4).flatten()])
        B4_vals = np.concatenate([B4_vals, np.array(B4).flatten()])
        C4_vals = np.concatenate([C4_vals, np.array(A4).flatten()])

        A0, A1, A2, A3, A4 = unroll_coefficients(np.array(A4),segment_waypoints[0,0],solver.dts)
        B0, B1, B2, B3, B4 = unroll_coefficients(np.array(B4),segment_waypoints[0,1],solver.dts)
        C0, C1, C2, C3, C4 = unroll_coefficients(np.array(C4),segment_waypoints[0,2],solver.dts)

        tn = solver.dts[n-1]
        xn, xdn, xddn = compute_state_1d(A0[n-1],A1[n-1],A2[n-1],A3[n-1],A4[n-1], tn)
        yn, ydn, yddn = compute_state_1d(B0[n-1],B1[n-1],B2[n-1],B3[n-1],B4[n-1], tn)
        zn, zdn, zddn = compute_state_1d(C0[n-1],C1[n-1],C2[n-1],C3[n-1],C4[n-1], tn)

        # Update velocity constraint at end of segment
        if(i < Nw-2):
            print("updating segment endpoint velocity: ", xdn, ydn, zdn)
            waypoint_desired_velocities[i+1,0] = xdn
            waypoint_desired_velocities[i+1,1] = ydn
            waypoint_desired_velocities[i+1,2] = zdn

        A0_plus = np.concatenate([np.array(A0).flatten(), np.array([xn]).flatten()])
        B0_plus = np.concatenate([np.array(B0).flatten(), np.array([yn]).flatten()])
        C0_plus = np.concatenate([np.array(C0).flatten(), np.array([zn]).flatten()])

        A0_vals = np.concatenate([A0_vals, np.array(A0).flatten()])
        B0_vals = np.concatenate([B0_vals, np.array(B0).flatten()])
        C0_vals = np.concatenate([C0_vals, np.array(C0).flatten()])

    ax0.plot(A0_vals, B0_vals, C0_vals)

    # plot trajectory of quadrotor evalutaed at every timestep

    # x_array, y_array, z_array = evalute_polynomials_over_control_time_step(1/60.0,A0, A1, A2, A3, A4, B0, B1, B2, B3,B4, C0, C1, C2, C3, C4)

    # ax0.plot(x_array,y_array,z_array)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    plt.show()

    # np.savez("coefficients.npz", A4=np.ndarray(A4_vals).reshape(-1,1), B4=np.ndarray(B4_vals).reshape(-1,1), C4=np.ndarray(C4_vals).reshape(-1,1))


if __name__ == "__main__":
    main()