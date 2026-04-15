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
        self.dt = 1.0/float(self.ctrl_freq)
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

        p_prev = self.waypoints[0:N,:]
        p_next = self.waypoints[1:,:]
        waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)
        print("waypoint_min_lengths=",waypoint_min_lengths.reshape(1,-1))
        total_length = np.sum(waypoint_min_lengths)
        t_initial = waypoint_min_lengths / total_length * max_duration
        print("t_initial=",t_initial.reshape(1,-1))

        self.times = t_initial
        # A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3 = self.initialize_coefficients(t_initial)

        x0 = np.concatenate(np.zeros((N*3,))).reshape(-1,1)

        print("x0=",x0.reshape(4,-1))

        print("x0.shape", x0.shape)

        contraint_ub = self.constraint_ub(N)
        contraint_lb = self.constraint_lb(N)

        constraint_func = lambda x: self.nl_constraints(x,N)
        nonl_constraints = NonlinearConstraint(constraint_func,contraint_lb, contraint_ub, jac='3-point', hess=BFGS())

        res = minimize(objective_func, x0, method='SLSQP',  jac='3-point',
               constraints=nonl_constraints, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)

        return res

    def bounds(self):
        # compute waypoint distances
        num_waypoints = self.waypoints.shape[0]
        N = num_waypoints - 1
        p_prev = self.waypoints[0:num_waypoints-1,:]
        p_next = self.waypoints[1:,:]
        waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)

        # times must also meet speed constraints
        print("self.dt", self.dt)

        # times must be gt eq zero
        A3_min = np.ones((N,1)) * - np.inf
        B3_min = np.ones((N,1)) * - np.inf
        C3_min = np.ones((N,1)) * - np.inf
        A3_max = np.ones((N,1)) * np.inf
        B3_max = np.ones((N,1)) * np.inf
        C3_max = np.ones((N,1)) * np.inf

        lb = np.concatenate([A3_min, B3_min, C3_min]).flatten()
        ub = np.concatenate([A3_max, B3_max, C3_max]).flatten()

        bounds = Bounds(lb, ub)
        return bounds

    def nl_constraints(self, X, n):
        times = X[0:n]
        A3 = X[n:2*n]
        B3 = X[2*n:3*n]
        C3 = X[3*n:4*n]

        A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3 = self.unwind_coefficients(times, A3, B3, C3)
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

        eqnx = A0 - self.waypoints[0:n,0]
        eqny = B0 - self.waypoints[0:n,1]
        eqnz = C0 - self.waypoints[0:n,2]

        eqnx1 = x - self.waypoints[1:,0]
        eqny1 = y - self.waypoints[1:,1]
        eqnz1 = z - self.waypoints[1:,2]

        # last velocity is zero
        xd = A1 + 2 * A2 * times + 3 * A3 * times**2
        yd = B1 + 2 * B2 * times + 3 * B3 * times**2
        zd = C1 + 2 * C2 * times + 3 * C3 * times**2

        # Velocity equality constraint
        xd0 = A1
        yd0 = B1
        zd0 = C1

        xd1 = np.insert(A1[1:], n-1,0)
        yd1 = np.insert(B1[1:], n-1,0)
        zd1 = np.insert(C1[1:], n-1,0)

        eqn_xd = xd - xd1
        eqn_yd = yd - yd1
        eqn_zd = zd - zd1

        xdd1 = np.insert(2*A2[1:], n-1,0)
        ydd1 = np.insert(2*B2[1:], n-1,0)
        zdd1 = np.insert(2*C2[1:], n-1,0)

        eqn_xdd = xdd - xdd1
        eqn_ydd = ydd - ydd1
        eqn_zdd = zdd - zdd1

        return np.concatenate([xy_acceleration_constraint, z_acceleration_constraint, 
                               xy_velocity_constraint, z_velocity_constraint,
                               eqnx, eqny, eqnz, 
                               eqnx1, eqny1, eqnz1, 
                               eqn_xd, eqn_yd, eqn_zd,
                               eqn_xdd, eqn_ydd, eqn_zdd]).flatten()

    def constraint_lb(self, n):
        num_constraints = 16*n

        lb = np.ones((1, num_constraints)) * 0
        lb[4*num_constraints:] = -1e-2
        return lb.flatten()

    def constraint_ub(self, n):
        num_constraints = 16*n

        ub = np.ones((1, num_constraints)) * np.inf
        # equality constraints
        ub[4*num_constraints:] = 1e-2
        return ub.flatten()

    def unwind_coefficients(self,A3, B3, C3):
        
        n = self.times.shape[0]

        times = self.times
        M = np.zeros((n,n))
        M[1:,:] = np.tril(np.ones((n-1,n)))

        A2 = 3 * M @ (A3 * times)
        B2 = 3 * M @ (B3 * times)
        C2 = 3 * M @ (C3 * times)

        A1 = 2 * M @ (A2 * times) + 3 * M @ (A3 * times **2)
        B1 = 2 * M @ (B2 * times) + 3 * M @ (B3 * times **2)
        C1 = 2 * M @ (C2 * times) + 3 * M @ (C3 * times **2)

        A0 = M @ (A1 * times) + M @ (A2 * times **2) + M @ (A3 * times **3) + self.waypoints[0,0]
        # A0 = self.waypoints[0:n,0]       

        B0 = M @ (B1 * times) + M @ (B2 * times **2) + M @ (B3 * times **3) + self.waypoints[0,1]
        # B0 = self.waypoints[0:n,1]

        C0 = M @ (C1 * times) + M @ (C2 * times **2) + M @ (C3 * times **3) + self.waypoints[0,2]
        # C0 = self.waypoints[0:n,2]

        Ai = np.concatenate([A0, A1, A2, A3]).reshape(-1,4)
        Bi = np.concatenate([B0, B1, B2, B3]).reshape(-1,4)
        Ci = np.concatenate([C0, C1, C2, C3]).reshape(-1,4)                
        """
        inputs: 3rd order polynomials
        outputs:
        """
        return A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3


    def initialize_coefficients(self,times):
        n = times.shape[0]
        A0 = self.waypoints[0:n,0]
        B0 = self.waypoints[0:n,1]
        C0 = self.waypoints[0:n,2]

        xd_hat = (self.waypoints[1:,0]-self.waypoints[0:n,0])/times
        xd_hat = np.insert(xd_hat, n, 0)
        xdd_hat = (xd_hat[1:] -xd_hat[0:n])/times
        xdd_hat = np.insert(xdd_hat, n, 0)
        xddd_hat = (xdd_hat[1:] -xdd_hat[0:n])/times

        A1 = np.zeros(times.shape)
        A1[1:] = xd_hat[0:n-1]      

        A2 = np.zeros(times.shape)
        A2[1:] = xdd_hat[0:n-1]

        A3 = np.zeros(times.shape)
        A3[1:] = 1/6 * xddd_hat[0:n-1]

        yd_hat = (self.waypoints[1:,1]-self.waypoints[0:n,1])/times
        yd_hat = np.insert(yd_hat, n, 0)
        ydd_hat = (yd_hat[1:] -yd_hat[0:n])/times
        ydd_hat = np.insert(ydd_hat, n, 0)
        yddd_hat = (ydd_hat[1:] -ydd_hat[0:n])/times

        B1 = np.zeros(times.shape)
        B1[1:] = yd_hat[0:n-1]      

        B2 = np.zeros(times.shape)
        B2[1:] = ydd_hat[0:n-1]

        B3 = np.zeros(times.shape)
        B3[1:] = 1/6 * yddd_hat[0:n-1]


        zd_hat = (self.waypoints[1:,2]-self.waypoints[0:n,2])/times
        zd_hat = np.insert(zd_hat, n, 0)
        zdd_hat = (zd_hat[1:] -zd_hat[0:n])/times
        zdd_hat = np.insert(zdd_hat, n, 0)
        zddd_hat = (zdd_hat[1:] - zdd_hat[0:n])/times

        C1 = np.zeros(times.shape)
        C1[1:] = zd_hat[0:n-1]      

        C2 = np.zeros(times.shape)
        C2[1:] = zdd_hat[0:n-1]

        C3 = np.zeros(times.shape)
        C3[1:] = 1/6 * zddd_hat[0:n-1]


        return A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3



    def objective_function(self, X,n):

        # data = X.reshape(4,n).T

        # A3 = data[:,0]
        # B3 = data[:,1]
        # C3 = data[:,2]

        A3 = X[0:n]
        B3 = X[n:2*n]
        C3 = X[2*n:3*n]
        
        jerk_integral = 36 * A3**2 * self.times

        return np.sum(jerk_integral)

    def compute_states(self, X,dt):
        n = self.N
        durations = X[0:n]
        A3 = X[n:2*n]
        B3 = X[2*n:3*n]
        C3 = X[3*n:4*n]

        A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3 = self.unwind_coefficients(durations, A3, B3, C3)

        total_duration = np.sum(durations)
        print("total duration", total_duration)
        print("dt", dt)
        # times = np.arange(durations.shape[0]) * total_duration * dt

        freq = 1/dt

        times = np.linspace(0, total_duration, int(total_duration*freq))

        # poly_indexes = 
        x = np.zeros((times.shape[0],))
        y = np.zeros((times.shape[0],))
        z = np.zeros((times.shape[0],))

        xd = np.zeros((times.shape[0],))
        yd = np.zeros((times.shape[0],))
        zd = np.zeros((times.shape[0],))

        xdd = np.zeros((times.shape[0],))
        ydd = np.zeros((times.shape[0],))
        zdd = np.zeros((times.shape[0],))

        segment_start_times = np.zeros((len(durations),1))
        segment_end_times = np.zeros((len(durations),1))
        # poly_indexes = []
        for j in range(1,len(durations)):
            segment_end_times[j-1] += durations[j-1]
            segment_start_times[j] = segment_end_times[j-1]

        for i in range(0, len(times)):
            # find which polynomial to use
            elapsed_time = i * dt
            poly_idx = 0
            for j in range(0,len(durations)):
                if( elapsed_time <= segment_end_times[j]):
                    break
                poly_idx +=1
            
            if poly_idx >= len(durations):
                poly_idx = len(durations)-1

            ti = elapsed_time - segment_start_times[poly_idx]

            xdd[i] = 2 * A2[poly_idx] + 6 * A3[poly_idx] * ti
            ydd[i] = 2 * B2[poly_idx] + 6 * B3[poly_idx] * ti
            zdd[i] = 2 * C2[poly_idx] + 6 * C3[poly_idx] * ti

            xd[i] = A1[poly_idx] + 2 * A2[poly_idx] * ti + 3 * A3[poly_idx] * ti**2
            yd[i] = B1[poly_idx] + 2 * B2[poly_idx] * ti + 3 * B3[poly_idx] * ti**2
            zd[i] = C1[poly_idx] + 2 * C2[poly_idx] * ti + 3 * C3[poly_idx] * ti**2

            x[i] = A0[poly_idx] + A1[poly_idx] * ti + A2[poly_idx] * ti**2 + A3[poly_idx] * ti**3
            y[i] = B0[poly_idx] + B1[poly_idx] * ti + B2[poly_idx] * ti**2 + B3[poly_idx] * ti**3
            z[i] = C0[poly_idx] + C1[poly_idx] * ti + C2[poly_idx] * ti**2 + C3[poly_idx] * ti**3

        print("xdd.shape", xdd.shape)
        print("xd.shape", xd.shape)
        print("x.shape", x.shape)

        return np.concatenate([times, x,y,z,xd,yd,zd,xdd,ydd,zdd]).reshape(-1,10)



def generate_waypoints():
    poses = []

    poses = [[-1.0, -3.0, 1.0], 
            [-0.09999980975910072, -2.49952220397356, 1.0], 
            [0.5, -2.5, 1.0], 
            [2.0, -2.1, 1.0]]
    return np.array(poses).reshape(-1,3)


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

    states = optimizer.compute_states(res.x,dt)
    print(states)

    state_times = states[:,0]
    print("state_times", state_times)

    return

def optimize_trajectory(waypoints):

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

    optimizer = WaypointOptimizer(waypoints, data)
    max_time = 120
    res = optimizer.optimize_segments(max_time)
    states = optimizer.compute_states(res.x,dt)

    return res, states # t, x, y,  z, xd, yd, zd, xdd, ydd, zdd

import matplotlib.pyplot as plt

def main():

    waypoints = generate_waypoints()
    print("waypoints=", waypoints.reshape(-1,3))

    freq = 60
    dt = 1/freq
    data = {}
    data["ctrl_freq"] = freq
    data["max_speed_xy"] = 1
    data["max_acceleration_xy"] = 5
    data["max_speed_z"] = 1
    data["max_acceleration_z"] = 0.517
    data["max_jerk_xy"] = 2*data["max_acceleration_xy"] / dt
    data["max_jerk_z"] = 2*data["max_acceleration_z"]/ dt
    data["max_tilt"] = 1.46 * np.pi / 180.0 # radians

    optimizer = WaypointOptimizer(waypoints, data)
    max_time = 120
    res = optimizer.optimize_segments(max_time)

    n = waypoints.shape[0] - 1

    data = res.x.reshape(4,n).T
    print("res.x", data)

    times = res.x[0:n]
    A3 = res.x[n:2*n]
    B3 = res.x[2*n:3*n]
    C3 = res.x[3*n:4*n]

    A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3 = optimizer.unwind_coefficients(times, A3, B3, C3)

    print("A0.shape", A0.shape)
    print("waypoints.shape", waypoints.shape)
    xdiff = A0 - waypoints[0:n, 0]
    ydiff = B0 - waypoints[0:n, 1]
    zdiff = C0 - waypoints[0:n, 2]
    computed_positions_diff = np.concatenate([xdiff, ydiff,zdiff]).reshape(-1,3)
    print("computed_positions_diff", computed_positions_diff)


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

        ti_total = ti+ prev_duration

        t_vals = np.concatenate([t_vals, ti_total])
        x_vals = np.concatenate([x_vals, xi])
        y_vals = np.concatenate([y_vals, yi])
        z_vals = np.concatenate([z_vals, zi])
        prev_duration = duration

    t_array = t_vals.flatten()
    x_array = np.array(x_vals).flatten()
    y_array = np.array(y_vals).flatten()
    z_array = np.array(z_vals).flatten()

    # [times, x,y,z,xd,yd,zd,xdd,ydd,zdd]
    # x = states[:,1]
    # y = states[:,2]
    # z = states[:,3]

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(A0, B0, C0, marker='^')
    ax.plot(x_array, y_array,z_array)
    # ax.legend()
    wx = waypoints[:,0]
    wy = waypoints[:,1]
    wz = waypoints[:,2]

    # ax1 = plt.figure().add_subplot(projection='3d')
    ax.scatter(wx, wy, wz, marker='o')

    plt.show()

    return


if __name__ == "__main__":
    main()