import numpy as np
import casadi as ca


class MPCController:
    def __init__(self, N=10):
        self.N = N
        self.dt = 0.1

        self.qr_ratio = ca.MX.sym('qr_ratio', 1)
        self.Q = np.diag([2, 1]) / self.qr_ratio[0]
        self.R = np.diag([1, 1])

        self.x = ca.MX.sym('x', 4, N+1)
        self.u = ca.MX.sym('u', 2, N)
        self.x0_ = ca.MX.sym('x0', 4)
        self.target_state = ca.MX.sym('target states', N*2)

        cost = 0
        g = []
        g.append(self.x[:,0] - self.x0_[:4])
        for k in range(N):
            x_next = self.vehicle_dynamics(self.x[:, k], self.u[:, k], L=0.3)
            g.append(self.x[:, k+1] - x_next)
            state_error = self.x[:2, k]
            state_error[0] -= self.target_state[k]
            state_error[1] -= self.target_state[N+k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            cost += ca.mtimes([self.u[:, k].T, self.R, self.u[:, k]])

        for k in range(N):
            g.append((self.x[3,k]**2)*ca.tan(self.u[1,k]))
        for k in range(N):
            g.append(self.x[3,k+1])

        opt_variables = ca.vertcat(ca.reshape(self.x, -1, 1), ca.reshape(self.u, -1, 1))
        opt_constraints = ca.vertcat(*g)

        # 'p' - equality constraints

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
        self.lbg = np.zeros((4 * (self.N + 1) + 2*self.N, 1))
        self.ubg = np.zeros((4 * (self.N + 1) + 2*self.N, 1))

    def vehicle_dynamics(self, xk, uk, L=0.34, Sa=0.34, Ta=20./4.65, fr=1.):
        x, y, psi, v = xk[0], xk[1], xk[2], xk[3]
        a, delta = uk[0], uk[1]
        return ca.vertcat(
            x + v * ca.cos(psi) * self.dt,
            y + v * ca.sin(psi) * self.dt,
            psi + (v / L) * ca.tan(Sa*delta) * self.dt,
            v + (Ta * a - fr) * self.dt
        )

    def mpc(self, x0, traj, mu=1.6, g=9.81, L=0.34, lookahead_factor=1.0):
        params = np.concatenate((x0, traj.T.flatten(), np.array([lookahead_factor])))
        x_init = np.tile(x0, (self.N+1, 1)).T
        u_init = np.zeros((2, self.N))
        initial_guess = ca.vertcat(ca.reshape(x_init[:4], -1, 1), ca.reshape(u_init, -1, 1))

        self.ubg[-2*self.N:-self.N] = mu*g*L
        self.lbg[-2*self.N:-self.N] = -mu*g*L
        self.ubg[-self.N:] = 5.

        solution = self.solver(x0=initial_guess, lbg=self.lbg, ubg=self.ubg, p=params)

        optimal_solution = solution['x']
        optimal_u = ca.reshape(optimal_solution[4*(self.N+1):], 2, self.N).full()
        u = optimal_u[:, 0]
        return float(u[0]), float(u[1])
    

x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
nlp = {'x':vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
S = nlpsol('S', 'ipopt', nlp)
print(S)



r = S(x0=[2.5,3.0,0.75],\
      lbg=0, ubg=0)
x_opt = r['x']
print('x_opt: ', x_opt)

