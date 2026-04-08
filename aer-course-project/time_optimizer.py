import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.spatial.transform import Rotation
from scipy.optimize import BFGS
from scipy.optimize import SR1

# from safe_control_gym.utils.configuration import ConfigFactory


def generate_waypoints(startPos, endPos):
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
    return np.array(poses).reshape(-1,3)

class TimeSegmentOptimizer():
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

    def optimize_time_segments(self, max_duration):
        # Example code: hardcode waypoints


        # Polynomial fit.
        deg = 4
        t = np.arange(self.waypoints.shape[0])

        num_waypoints = self.waypoints.shape[0]

        p_prev = self.waypoints[0:num_waypoints-1,:]
        p_next = self.waypoints[1:,:]

        print("p_next.shape", p_next.shape)
        waypoint_min_lengths = np.linalg.norm(p_next - p_prev, axis=1)
        print("waypoint_min_lengths.shape", waypoint_min_lengths.shape)
        total_length = np.sum(waypoint_min_lengths)
        t_initial = waypoint_min_lengths / total_length * max_duration
        print("t_initial", t_initial)

        print("t total", sum(t_initial))

        print(f"number of waypoints {num_waypoints}")
        print(f"number of time segments {len(t_initial)}")

        constraint_func = lambda x: self.constraints(x)
        objective_func = lambda x: self.objective_function(x)
        # bounds = self.bounds(t_initial)

        lb = np.zeros_like(t_initial)
        ub = np.ones_like(t_initial) * np.inf
        bounds = Bounds(lb, ub)

        print("Running minimization")

        vx_max = self.max_speed_xy * np.ones_like(t_initial)
        vy_max = self.max_speed_xy * np.ones_like(t_initial)
        vz_max = self.max_speed_z * np.ones_like(t_initial)

        ax_max = self.max_acceleration_xy * np.ones_like(t_initial)
        ay_max = self.max_acceleration_xy * np.ones_like(t_initial)
        az_max = self.max_acceleration_z * np.ones_like(t_initial)

        jx_max = self.max_jerk_xy * np.ones_like(t_initial)
        jy_max = self.max_jerk_xy * np.ones_like(t_initial)
        jz_max = self.max_jerk_z * np.ones_like(t_initial)

        nl_ub = [vx_max, vy_max,vz_max,ax_max, ay_max,az_max,jx_max, jy_max,jz_max]
        nl_lb = [-vx_max, -vy_max, -vz_max, -ax_max, -ay_max, -az_max, -jx_max, -jy_max, -jz_max]

        nonl_constraints = NonlinearConstraint(constraint_func,0, np.inf, jac='2-point', hess=BFGS())

        res = minimize(objective_func, t_initial, method='SLSQP', jac='2-point', hess=SR1(), constraints=nonl_constraints, bounds=bounds, options={'verbose': 1})
        return res


    def bounds(self, times):
        # Bounds(lb, ub)
        # bounds = Bounds([0, -0.5], [1.0, 2.0])
        t0 = np.zeros_like(times)
        tmax = np.ones_like(times) * np.inf

        bounds = Bounds(t0, tmax)
        return bounds

    def objective_function(self, times):
        return np.sum(times**2)

        # t_initial = np.linspace(t[0], t[-1], int(max_duration*ctrl_freq))

    def constraints(self, durations):
        # should return g.t or equal to zero
        deg = 4
        times = np.insert(durations,0,0)
        fx = np.poly1d(np.polyfit(times, self.waypoints[:,0], deg))
        fy = np.poly1d(np.polyfit(times, self.waypoints[:,1], deg))
        fz = np.poly1d(np.polyfit(times, self.waypoints[:,2], deg))

        # velocity polynomial
        dfx = fx.deriv()
        dfy = fy.deriv()
        dfz = fz.deriv()

        # acceleration polynomial
        ddfx = dfx.deriv()
        ddfy = dfy.deriv()
        ddfz = dfz.deriv()

        # jerk polynomial
        d3fx = dfx.deriv()
        d3fy = dfy.deriv()
        d3fz = dfz.deriv()

        # position
        ref_x = fx(times)
        ref_y = fy(times)
        ref_z = fz(times)

        # velocity
        ref_vx = dfx(times)
        ref_vy = dfy(times)
        ref_vz = dfz(times)

        vx_constraints = self.max_speed_xy - np.abs(ref_vx)
        vy_constraints = self.max_speed_xy - np.abs(ref_vy)
        vz_constraints = self.max_speed_z - np.abs(ref_vz)

        # accleration
        ref_ax = ddfx(times)
        ref_ay = ddfy(times)
        ref_az = ddfz(times)

        ax_constraints = self.max_acceleration_xy - np.abs(ref_ax)
        ay_constraints = self.max_acceleration_xy - np.abs(ref_ay)
        az_constraints = self.max_acceleration_z - np.abs(ref_az)

        # jerk
        ref_jx = d3fx(times)
        ref_jy = d3fy(times)
        ref_jz = d3fz(times)

        jx_constraints = self.max_jerk_xy - np.abs(ref_jx)
        jy_constraints = self.max_jerk_xy - np.abs(ref_jy)
        jz_constraints = self.max_jerk_z - np.abs(ref_jz)

        return np.concatenate([vx_constraints,
                               vy_constraints,
                               vz_constraints,
                               ax_constraints,
                               ay_constraints,
                               az_constraints,
                               jx_constraints,
                               jy_constraints,
                               jz_constraints])

        # return np.max(np.abs(np.concatenate([ref_vx,
        #                        ref_vy,
        #                        ref_vz,
        #                        ref_ax,
        #                        ref_ay,
        #                        ref_az,
        #                        ref_jx,
        #                        ref_jy,
        #                        ref_jz])),axis=0)

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

    optimizer = TimeSegmentOptimizer(waypoints, data)
    max_time = 360
    res = optimizer.optimize_time_segments(max_time)

    print(res)

    print("------------------------------------")
    print(res.x)
    print(np.sum(res.x))

    return

if __name__ == "__main__":
    test()