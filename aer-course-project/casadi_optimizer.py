
# from casadi import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class CasadiSolver:
    def __init__(self, waypoints,durations):
        self.durations = durations
        self.waypoints = waypoints
        N = waypoints.shape[0] - 1
        self.N = N
        self.A3 = ca.SX.sym('A3', N)
        self.B3 = ca.SX.sym('B3', N)
        self.C3 = ca.SX.sym('C3', N)

        self.frac = ca.SX.sym('t_frac', N)
        self.total_duration = ca.SX.sym('T')

        self.max_accel_xy = 5
        self.max_accel_z = 0.517
        self.dt = 1/60

        self.min_time = 5

        self.max_jerk_xy = 2 * self.max_accel_xy / self.dt * 0.01
        self.max_jerk_z = 2 * self.max_accel_z / self.dt * 0.01

        max_duration = np.sum(durations)
        fractions = durations / max_duration

        computed_times = self.frac/(ca.sum(self.frac)) * self.total_duration
        x_error = position_error_1d(self.A3, waypoints[:,0], computed_times)

        y_error = position_error_1d(self.B3, waypoints[:,1], computed_times)

        z_error = position_error_1d(self.C3, waypoints[:,2], computed_times)

        error_cost = ca.sum(x_error **2) + ca.sum(y_error**2) + ca.sum(z_error**2)
        jerk_integral = 36 * ca.sum(self.A3**2 * computed_times) + ca.sum(self.B3**2 * computed_times) + ca.sum(self.C3**2 * computed_times)

        a1 = 0
        a2 = 0
        a3 = 5
        # cost = a1 * error_cost + a2 * jerk_integral + a3* ca.sum(computed_times)
        cost = a1 * error_cost + a2 * jerk_integral + a3* ca.sum(computed_times)**2


        opt_variables = ca.vertcat(
            self.total_duration,
            ca.reshape(self.frac, -1, 1), 
            ca.reshape(self.A3, -1, 1), 
            ca.reshape(self.B3, -1, 1), 
            ca.reshape(self.C3, -1, 1))

        g = []
        g.append(self.total_duration)
        g.append(ca.sum(fractions) - 1)

        g.append(self.frac)
        g.append(self.A3)
        g.append(self.B3)
        g.append(self.C3)
        g.append(x_error)
        g.append(y_error)
        g.append(z_error)

        # print("x_error.shape", x_error.shape)

        frac_start = 2
        frac_end = frac_start + N
        a3_start = frac_end
        a3_end = a3_start + N
        b3_start = a3_end
        b3_end = b3_start + N
        c3_start = b3_end
        c3_end = c3_start + N 

        lb = np.zeros(4 * N + 2 + 3*N)
        # total time lb
        lb[0] = self.min_time
        # fraction equality constraint
        lb[1] = 0

        # fraction lower bound
        lb[frac_start:frac_end] = 0
        # A3 lower bound
        lb[a3_start:a3_end] = -self.max_jerk_xy 
        # B3 lower bound
        lb[b3_start:b3_end] = -self.max_jerk_xy
        # C3 lower bound
        lb[c3_start:c3_end] = -self.max_jerk_z
        # position error lower bound
        lb[c3_end:] = -1e-3


        ub = np.zeros(4 * N + 2 + 3*N)
        # total time lb
        ub[0] = max_duration
        # fraction equality constraint
        ub[1] = 0
 
        # fraction upper bound
        ub[frac_start:frac_end] = 1
        # A3 lower bound
        ub[a3_start:a3_end] = self.max_jerk_xy 
        # B3 lower bound
        ub[b3_start:b3_end] = self.max_jerk_xy
        # C3 lower bound
        ub[c3_start:c3_end] = self.max_jerk_z
        # position error upper bound
        ub[c3_end:] =  1e-3

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



def position_error_1d(P3,waypoints, times):

    n = waypoints.shape[0] - 1

    M = np.zeros((n,n))
    M[1:,:] = np.tril(np.ones((n-1,n)))

    # P2 = 3 * M @ (P3 * times)
    P2 = 3 * ca.mtimes([M, P3]) * times

    # P1 = 2 * M @ (P2 * times) + 3 * M @ (P3 * times **2)
    P1 = 2 * ca.mtimes(M, P2) * times + 3* ca.mtimes(M, P3) * times **2

    # P0 = M @ (P1 * times) + M @ (P2 * times **2) + M @ (P3 * times **3) + waypoints[0]
    P0 = ca.mtimes(M, P1) * times + ca.mtimes(M, P2) * times**2 + ca.mtimes(M, P3) * times**3 + waypoints[0]

    # p = P0 + P1 * times + P2 * times**2 + P3 * times**3
    p = P0 + P1 * times + P2*times**2 + P3 * times**3

    # last velocity is zero
    # pd = P1 + 2 * P2 * times + 3 * P3 * times**2
    # pdd = 2 * P2 + 6 * P3 * times

    # Velocity equality constraint

    # pd1 = np.insert(P1[1:], n-1,0)

    eqn_p1 = p - waypoints[1:]

    # eqn_pd = pd - pd1

    # pdd1 = np.insert(2*P2[1:], n-1,0)

    # eqn_pdd = pdd - pdd1

    return eqn_p1

def generate_waypoints():
    # poses = [[-1.0, -3.0, 1.0], 
    #         [-0.09999980975910072, -2.49952220397356, 1.0], 
    #         [0.5, -2.5, 1.0], 
    #         [2.0, -2.1, 1.0], 
    #         [2.0, -1.5, 1.0], 
    #         [1.2999999048795503, -0.64976110198678, 1.0], 
    #         [0.5999998097591007, 0.20047779602643997, 1.0], 
    #         [0.0, 0.2, 1.0], 
    #         [-0.5, 0.9, 1.0], 
    #         [-0.5, 1.5, 1.0], 
    #         [-0.5, 2.0, 1.0]]

    poses = [[-1.0, -3.0, 1.0], 
            [-0.09999980975910072, -2.49952220397356, 1.0], 
            [0.5, -2.5, 1.0], 
            [2.0, -2.1, 1.0]]

    return np.array(poses).reshape(-1,3)

def unwind_coefficients(A3, B3, C3, times, waypoints):
    
    n = times.shape[0]

    M = np.zeros((n,n))
    M[1:,:] = np.tril(np.ones((n-1,n)))

    A2 = 3 * M @ (A3 * times)
    B2 = 3 * M @ (B3 * times)
    C2 = 3 * M @ (C3 * times)

    A1 = 2 * M @ (A2 * times) + 3 * M @ (A3 * times **2)
    B1 = 2 * M @ (B2 * times) + 3 * M @ (B3 * times **2)
    C1 = 2 * M @ (C2 * times) + 3 * M @ (C3 * times **2)

    A0 = M @ (A1 * times) + M @ (A2 * times **2) + M @ (A3 * times **3) + waypoints[0,0]
    # A0 = self.waypoints[0:n,0]       

    B0 = M @ (B1 * times) + M @ (B2 * times **2) + M @ (B3 * times **3) + waypoints[0,1]
    # B0 = self.waypoints[0:n,1]

    C0 = M @ (C1 * times) + M @ (C2 * times **2) + M @ (C3 * times **3) + waypoints[0,2]
    # C0 = self.waypoints[0:n,2]

    Ai = np.concatenate([A0, A1, A2, A3]).reshape(-1,4)
    Bi = np.concatenate([B0, B1, B2, B3]).reshape(-1,4)
    Ci = np.concatenate([C0, C1, C2, C3]).reshape(-1,4)                
    """
    inputs: 3rd order polynomials
    outputs:
    """
    A0 = np.array(A0).flatten()
    A1 = np.array(A1).flatten()
    A2 = np.array(A2).flatten()
    A3 = np.array(A3).flatten()
    B0 = np.array(B0).flatten()
    B1 = np.array(B1).flatten()
    B2 = np.array(B2).flatten()
    B3 = np.array(B3).flatten()
    C0 = np.array(C0).flatten()
    C1 = np.array(C1).flatten()
    C2 = np.array(C2).flatten()
    C3 = np.array(C3).flatten()
    return A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3

def evalute_polynomials_over_control_time_step(times, dt, n, freq, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3):
    x_vals = np.array([])
    y_vals = np.array([])
    z_vals = np.array([])
    t_vals = np.array([])

    prev_duration = 0
    for idx in range(0,n):
        duration = times[idx]
        nsample = int(duration * freq)
        ti = dt * np.arange(nsample)

        xi = A0[idx] + A1[idx] * ti + A2[idx] * ti**2 + A3[idx] * ti**3
        yi = B0[idx] + B1[idx] * ti + B2[idx] * ti**2 + B3[idx] * ti**3
        zi = C0[idx] + C1[idx] * ti + C2[idx] * ti**2 + C3[idx] * ti**3

        # ti_total = np.array(ti) + prev_duration

        # print("nsample",nsample)
        # print("ti.shape",ti.shape)
        # print("ti_total.shape",ti_total.shape)
        # print("t_vals.shape",t_vals.shape)

        # t_vals = np.concatenate([t_vals, ti_total.flatten()])
        x_vals = np.concatenate([x_vals, xi])
        y_vals = np.concatenate([y_vals, yi])
        z_vals = np.concatenate([z_vals, zi])
        prev_duration = duration

    t_array = t_vals.flatten()
    x_array = np.array(x_vals).flatten()
    y_array = np.array(y_vals).flatten()
    z_array = np.array(z_vals).flatten()

    return t_array, x_array, y_array, z_array

def main():
    waypoints = generate_waypoints()
    print("waypoints=", waypoints.reshape(-1,3))

    max_time = 60
    n = waypoints.shape[0]-1

    p_prev = waypoints[0:n,:]
    p_next = waypoints[1:,:]
    waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

    print("waypoint_min_lengths=",waypoint_min_lengths.reshape(1,-1))

    total_length = np.sum(waypoint_min_lengths)
    durations = waypoint_min_lengths / total_length * max_time

    fractions = durations / max_time

    print("fractions=", fractions)


    solver = CasadiSolver(waypoints,durations)


    frac_start = 1
    frac_end = frac_start + n
    a3_start = frac_end
    a3_end = a3_start + n
    b3_start = a3_end
    b3_end = b3_start + n
    c3_start = b3_end
    c3_end = c3_start + n 

    X0 = np.zeros(4 * n + 1)
    X0[0] = max_time
    # fractions
    X0[frac_start:frac_end] = fractions
    # A3 lower bound
    X0[a3_start:a3_end] = 0 
    # B3 lower bound
    X0[b3_start:b3_end] = 0
    # C3 lower bound
    X0[c3_start:c3_end] = 0


    
    t_total, fracs, A3, B3, C3, solution = solver.solve_coefficients(X0)


    print("solution", solution)
    print("t_total", t_total)
    print("fracs", fracs)
    print("A3", A3)
    print("B3", B3)
    print("C3", C3)

    durations_star = (fracs / np.sum(fracs) * t_total).reshape((-1,1))

    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3 = unwind_coefficients(A3, B3, C3, durations_star, waypoints)

    tn = durations_star[n-1]
    xn = A0[n-1] + A1[n-1] * tn + A2[n-1] * tn**2 + A3[n-1] * tn**3
    yn = B0[n-1] + B1[n-1] * tn + B2[n-1] * tn**2 + B3[n-1] * tn**3
    zn = C0[n-1] + C1[n-1] * tn + C2[n-1] * tn**2 + C3[n-1] * tn**3

    print("A0.shape",A0.shape)
    print("xn.shape",np.array([xn]).shape)

    print("durations_star.shape",durations_star.shape)

    A0_plus = np.concatenate([np.array(A0).flatten(), np.array([xn]).flatten()])
    B0_plus = np.concatenate([np.array(B0).flatten(), np.array([yn]).flatten()])
    C0_plus = np.concatenate([np.array(C0).flatten(), np.array([zn]).flatten()])

    print("A0_plus", A0_plus)
    print("B0_plus", B0_plus)
    print("C0_plus", C0_plus)
    print("waypoints=", waypoints.reshape(-1,3))

    ax0 = plt.figure().add_subplot(projection='3d')
    ax0.scatter(A0_plus, B0_plus, C0_plus, marker='^')
    ax0.scatter(wx, wy, wz, marker='o')

    # plot trajectory of quadrotor evalutaed at every timestep
    ctrl_freq = 60.0
    dt = 1/ctrl_freq
    t_array, x_array, y_array, z_array = evalute_polynomials_over_control_time_step(durations_star, dt, n, ctrl_freq, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3)

    ax0.plot(x_array,y_array,z_array)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    plt.show()


if __name__ == "__main__":
    main()