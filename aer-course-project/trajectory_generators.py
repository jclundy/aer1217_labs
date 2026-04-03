import numpy as np
from scipy.spatial.transform import Rotation

def circle_trajectory_state(initial_state, center, r_c, omega, t):
    alpha = 0
    beta = 0
    gamma = 0

    # Parameters
    g = 9.81

    # Disturbance estimate from UDE

    # xref
    origin = center
    px_0 = r_c * np.cos(omega * t)
    py_0 = r_c * np.sin(omega * t)
    pz_0 = 0
    p_vec0 = np.array([px_0, py_0, pz_0])

    R_incline = Rotation.from_euler('xyz',[alpha, beta, gamma], degrees=False).as_matrix().reshape(3,3)

    p_vec = R_incline @ p_vec0 + origin

    p = np.array([p_vec[0], p_vec[1], p_vec[2]]).reshape(-1,1)  # Position in inertial frame

    r_center =  (origin-p_vec) / np.linalg.norm((origin-p_vec))
    r_vertical =  R_incline @ np.array([0,0,1])
    r_tangent = np.cross(r_center, r_vertical)
    r_tangent = r_tangent / np.linalg.norm(r_tangent)

    V = omega * r_c * r_tangent
    v = np.array([V[0],V[1], V[2]]).reshape(-1,1)  # Velocity in inertial frame

    # centripedal acceleration
    ac_hat = (origin-p_vec) / np.linalg.norm((origin-p_vec))
    ac = ac_hat * omega**2*r_c

    # combined desired acceleration
    a_des = ac + np.array([0.,0.,g])
    T = np.linalg.norm(a_des)

    # desired yaw angle
    yaw_des = omega * t + initial_state[8] # psi
    phi_dot = 0

    z_b = a_des / np.linalg.norm(a_des)
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
    y_c = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0])
    x_b = np.cross(y_c, z_b)
    x_b = x_b / np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    y_b = y_b / np.linalg.norm(y_b)
    R = np.column_stack([x_b, y_b, z_b])

    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=False).reshape(-1,1)


    # jerk
    c = T
    j_x = omega**3 * r_c * np.sin(omega*t)
    j_y = -omega**3 * r_c * np.cos(omega * t)
    j_z = 0
    j = R_incline @ np.array([j_x,j_y,j_z]).T

    w_x = - y_b.T @ j / c
    w_y = x_b.T @ j / c
    w_z = phi_dot * x_c.T @ x_b + w_y * y_c.T @ z_b

    # ustar_t = np.array([T, w_x, w_y, w_z]).reshape(-1,1)
    body_rates = np.array([w_x, w_y, w_z]).reshape(-1,1)
    xstar_t = np.concatenate([p, v, ac.reshape(3,1), euler, body_rates]).reshape(-1, 1)

    return xstar_t

def circle_trajectory_generator(initial_obs, radius, duration, ctrl_freq):
    init_pos = np.array([initial_obs[0], initial_obs[2], initial_obs[4]])
    center = init_pos + np.array([-radius, 0.0, 1])

    omega = 2 * np.pi / duration

    nsample = int(duration * ctrl_freq)
    time = 1/ctrl_freq * np.arange(nsample)

    ref_state = []

    for i in range(nsample):
        t = time[i]
        xstar_t = circle_trajectory_state(initial_obs, center, radius, omega, t)
        ref_state.append(xstar_t)


    return np.array(ref_state).reshape(-1, 15)


def hardcoded_trajectory_generator(initial_obs, initial_info, ctrl_freq, duration):
        # Example code: hardcode waypoints
        poses = []
        poses.append((initial_obs[0], initial_obs[2], initial_obs[4]))
        poses.append((-0.5, -3.0, 2.0))
        poses.append((-0.5, -2.0, 2.0))
        poses.append((-0.5, -1.0, 2.0))
        poses.append((-0.5,  0.0, 2.0))
        poses.append((-0.5,  1.0, 2.0))
        poses.append((-0.5,  2.0, 2.0))
        poses.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])

        # Polynomial fit.
        waypoints = np.array(poses)
        deg = 6
        t = np.arange(waypoints.shape[0])
        fx = np.poly1d(np.polyfit(t, waypoints[:,0], deg))
        fy = np.poly1d(np.polyfit(t, waypoints[:,1], deg))
        fz = np.poly1d(np.polyfit(t, waypoints[:,2], deg))
        t_scaled = np.linspace(t[0], t[-1], int(duration*ctrl_freq))

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
        ref_x = fx(t_scaled)
        ref_y = fy(t_scaled)
        ref_z = fz(t_scaled)

        # velocity
        ref_vx = dfx(t_scaled)
        ref_vy = dfy(t_scaled)
        ref_vz = dfz(t_scaled)

        # accleration
        ref_ax = ddfx(t_scaled)
        ref_ay = ddfy(t_scaled)
        ref_az = ddfz(t_scaled)

        # jerk
        ref_jx = d3fx(t_scaled)
        ref_jy = d3fy(t_scaled)
        ref_jz = d3fz(t_scaled)

        # euler values
        euler_values = []
        # body rates
        body_rates = []

        for i in range(0, int(duration*ctrl_freq)):
            # desired yaw angle
            # vx = ref_vx[i]
            # vy = ref_vy[i]

            yaw_des = initial_obs[8]
            phi_dot = 0

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

        p_ref = np.column_stack((ref_x, ref_y, ref_z))
        v_ref = np.column_stack((ref_vx, ref_vy, ref_vz))
        a_ref = np.column_stack((ref_ax, ref_ay, ref_az))

        euler_ref = np.array(euler_values).reshape(-1,3)
        body_rates_ref = np.array(body_rates).reshape(-1,3)

        return np.column_stack([p_ref, v_ref, a_ref, euler_ref, body_rates_ref]).reshape(-1, 15)