"""
Copyright (C) 2020 Benjamin Bokser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Reference Material:

Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control
Donghyun Kim et al.

https://www.youtube.com/watch?v=RrnkPrcpyEA
https://github.com/MMehrez/ ...Sim_1_MPC_Robot_PS_sing_shooting.m
"""

import numpy as np
import scipy
import csv

import casadi as cs

import control


class Mpc:

    def __init__(self, dt=1e-3, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = dt  # sampling time (s)
        self.N = 3  # prediction horizon
        self.mass = float(12.12427)  # kg

        self.f_max = 5
        self.f_min = -self.f_max

        with open('spryped_urdf_rev06/spryped_data_body.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            next(data)  # skip headers
            values = list(zip(*(row for row in data)))  # transpose rows to columns
            values = np.array(values)  # convert list of nested lists to array

        ixx = values[1].astype(np.float)
        ixy = values[2].astype(np.float)
        ixz = values[3].astype(np.float)
        iyy = values[4].astype(np.float)
        iyz = values[5].astype(np.float)
        izz = values[6].astype(np.float)

        # m = np.zeros((6, 6))
        # m[0:3, 0:3] = np.eye(3) * self.mass
        i = np.zeros((3, 3))
        i[0, 0] = ixx
        i[0, 1] = ixy
        i[0, 2] = ixz
        i[1, 0] = ixy
        i[1, 1] = iyy
        i[1, 2] = iyz
        i[2, 0] = ixz
        i[2, 1] = iyz
        i[2, 2] = izz
        self.inertia = i  # inertia tensor in local frame

    def mpcontrol(self, rz_phi, r1, r2, x):
        i_global = np.dot(np.dot(rz_phi, self.inertia), rz_phi.T)
        i_inv = np.linalg.inv(i_global)
        i11 = i_inv[0, 0]
        i12 = i_inv[0, 1]
        i13 = i_inv[0, 2]
        i21 = i_inv[1, 0]
        i22 = i_inv[1, 1]
        i23 = i_inv[1, 2]
        i31 = i_inv[2, 0]
        i32 = i_inv[2, 1]
        i33 = i_inv[2, 2]

        rz11 = rz_phi[0, 0]
        rz12 = rz_phi[0, 1]
        rz13 = rz_phi[0, 2]
        rz21 = rz_phi[1, 0]
        rz22 = rz_phi[1, 1]
        rz23 = rz_phi[1, 2]
        rz31 = rz_phi[2, 0]
        rz32 = rz_phi[2, 1]
        rz33 = rz_phi[2, 2]

        r1x = r1[0]
        r1y = r1[1]
        r1z = r1[2]
        r2x = r2[0]
        r2y = r2[1]
        r2z = r2[2]
        # r = foot position

        theta_x = cs.SX.sym('theta_x')
        theta_y = cs.SX.sym('theta_y')
        theta_z = cs.SX.sym('theta_z')
        p_x = cs.SX.sym('p_x')
        p_y = cs.SX.sym('p_y')
        p_z = cs.SX.sym('p_z')
        omega_x = cs.SX.sym('omega_x')
        omega_y = cs.SX.sym('omega_y')
        omega_z = cs.SX.sym('omega_z')
        pdot_x = cs.SX.sym('pdot_x')
        pdot_y = cs.SX.sym('pdot_y')
        pdot_z = cs.SX.sym('pdot_z')
        states = [theta_x, theta_y, theta_z,
                  p_x, p_y, p_z,
                  omega_x, omega_y, omega_z,
                  pdot_x, pdot_y, pdot_z]  # state vector x
        n_states = len(states)  # number of states

        f1_x = cs.SX.sym('f1_x')  # controls
        f1_y = cs.SX.sym('f1_y')  # controls
        f1_z = cs.SX.sym('f1_z')  # controls
        f2_x = cs.SX.sym('f2_x')  # controls
        f2_y = cs.SX.sym('f2_y')  # controls
        f2_z = cs.SX.sym('f2_z')  # controls
        controls = [f1_x, f1_y, f1_z, f2_x, f2_y, f2_z]
        n_controls = len(controls)  # number of controls

        gravity = -9.807
        dt = self.dt
        mass = self.mass

        # x_next = np.dot(A, states) + np.dot(B, controls) + g  # the discrete dynamics of the system
        x_next = [(dt * omega_x * rz11 + dt * omega_y * rz12 + dt * omega_z * rz13 + theta_x + theta_y + theta_z),
                  (dt * omega_x * rz21 + dt * omega_y * rz22 + dt * omega_z * rz23 + theta_x + theta_y + theta_z),
                  (dt * omega_x * rz31 + dt * omega_y * rz32 + dt * omega_z * rz33 + theta_x + theta_y + theta_z),
                  (dt * pdot_x + dt * pdot_y + dt * pdot_z + p_x + p_y + p_z),
                  (dt * pdot_x + dt * pdot_y + dt * pdot_z + p_x + p_y + p_z),
                  (dt * pdot_x + dt * pdot_y + dt * pdot_z + p_x + p_y + p_z),
                  (dt * f1_x * (i12 * r1z - i13 * r1y) + dt * f1_y * (-i11 * r1z + i13 * r1x)
                   + dt * f1_z * (i11 * r1y - i12 * r1x)
                   + dt * f2_x * (i12 * r2z - i13 * r2y) + dt * f2_y * (-i11 * r2z + i13 * r2x)
                   + dt * f2_z * (i11 * r2y - i12 * r2x)
                   + omega_x + omega_y + omega_z),
                  (dt * f1_x * (i22 * r1z - i23 * r1y) + dt * f1_y * (-i21 * r1z + i23 * r1x)
                   + dt * f1_z * (i21 * r1y - i22 * r1x)
                   + dt * f2_x * (i22 * r2z - i23 * r2y) + dt * f2_y * (-i21 * r2z + i23 * r2x)
                   + dt * f2_z * (i21 * r2y - i22 * r2x)
                   + omega_x + omega_y + omega_z),
                  (dt * f1_x * (i32 * r1z - i33 * r1y) + dt * f1_y * (-i31 * r1z + i33 * r1x)
                   + dt * f1_z * (i31 * r1y - i32 * r1x)
                   + dt * f2_x * (i32 * r2z - i33 * r2y) + dt * f2_y * (-i31 * r2z + i33 * r2x)
                   + dt * f2_z * (i31 * r2y - i32 * r2x)
                   + omega_x + omega_y + omega_z),
                  (dt * f1_x / mass + dt * f1_y / mass + dt * f1_z / mass + dt * f2_x / mass
                   + dt * f2_y / mass + dt * f2_z / mass + pdot_x + pdot_y + pdot_z),
                  (dt * f1_x / mass + dt * f1_y / mass + dt * f1_z / mass + dt * f2_x / mass
                   + dt * f2_y / mass + dt * f2_z / mass + pdot_x + pdot_y + pdot_z),
                  (dt * f1_x / mass + dt * f1_y / mass + dt * f1_z / mass + dt * f2_x / mass
                   + dt * f2_y / mass + dt * f2_z / mass + gravity + pdot_x + pdot_y + pdot_z)]

        fn = cs.Function('fn', [theta_x, theta_y, theta_z,
                             p_x, p_y, p_z,
                             omega_x, omega_y, omega_z,
                             pdot_x, pdot_y, pdot_z,
                             f1_x, f1_y, f1_z,
                             f2_x, f2_y, f2_z],
                      x_next)  # nonlinear mapping of function f(x,u)

        u = cs.SX.sym('u', n_controls, self.N)  # decision variables, control action matrix
        st_ref = cs.SX.sym('st_ref', n_states + n_states)  # initial and reference states

        x = cs.SX.sym('x', n_states, (self.N + 1))  # represents the states over the opt problem.

        # compute solution symbolically
        x[:, 0] = st_ref[0:12]  # initial state
        # In Python, slicing is left inclusive and right exclusive,
        # whereas in MATLAB slicing is inclusive at both.
        # Matlab uses one based indexing while python uses zero based indexing.
        for k in range(0, self.N - 1):  # N-1 because of python zero based indexing
            st = x[:, k]  # extract the previous state from x
            con = u[:, k]  # extract controls from control matrix
            # st_next = fn(st, con)  # pass states and controls through function
            for j in range(0, 11):
                x[j, k + 1] = fn(st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8], st[9], st[10],
                                 st[11], con[0], con[1], con[2], con[3], con[4], con[5])[j]
                # because CasADI is dumb
            # st_next = [st + element * self.dt for element in f_value]
            # ^n/a here. Already x(k+1). Mehrez used xdot as rhs

        # function for optimal traj knowing optimal sol
        ff = cs.Function('ff', [u, st_ref], [x])

        obj = 0  # objective function
        constr = []  # constraints vector

        Q = np.zeros((12, 12))  # state weighing matrix
        Q[0, 0] = 1
        Q[1, 1] = 1
        Q[2, 2] = 1
        Q[3, 3] = 1
        Q[4, 4] = 1
        Q[5, 5] = 1
        Q[6, 6] = 1
        Q[7, 7] = 1
        Q[8, 8] = 1
        Q[9, 9] = 1
        Q[10, 10] = 1
        Q[11, 11] = 1

        R = np.zeros((6, 6))  # control weighing matrix
        R[0, 0] = 1
        R[1, 1] = 1
        R[2, 2] = 1
        R[3, 3] = 1
        R[4, 4] = 1
        R[5, 5] = 1

        # compute objective
        for k in range(0, self.N-1):  # 0 and N-1 because of python zero based indexing
            st = x[:, k + 1]
            con = u[:, k]  # control action
            # calculate objective
            obj = obj + cs.mtimes(cs.mtimes((st - st_ref[11:23]).T, Q), st - st_ref[11:23]) \
                + cs.mtimes(cs.mtimes(con.T, R), con)

        # compute constraints
        for k in range(0, self.N):  # would be N+1 in matlab
            for j in range(0, 11):
                constr = cs.vertcat(constr, x[j, k])  # f1x

        # make decision variables one column vector
        opt_variables = cs.vertcat(cs.reshape(x, n_states*(self.N + 1), 1), cs.reshape(u, n_controls*self.N, 1))

        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'max_iter': 100}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        args = {lbg: -2,  # inequality constraints: lower bound
                ubg: 2,  # inequality constraints: upper bound
                lbx: self.f_min,  # input constraints: lower bound
                ubx: self.f_max,  # input constraints: upper bound
                }

        # -------------Starting Simulation Loop Now------------------------------------- #
        t0 = 0
        x0 = array([0, 0, 0, 0]).T  # initial condition of the bot, gets updated every iteration (input)
        xs = array([0, 0, 0, 0]).T  # reference posture (desired)

        xx[:, 0] = x0  # contains history of states
        t[0] = t0

        u0 = np.zeros(N, 2)  # two control inputs

        sim_tim = 4  # max sim time
        # start MPC
        mpciter = 0
        xx1 = []
        u_cl = []

        while np.linalg.norm(x0 - xs) > 1e-2 and mpciter < sim_tim / T:
            args[p] = array([x0, xs]).T  # set values of parameters vector
            args[x0] = np.reshape(u0.T, (2 * N, 1))  # init value of optimization variables

            sol = solver('x0', args[x0], 'lbx', args[lbx], 'ubx', args[ubx],
                         'lbg', args[lbg], 'ubg', args[ubg], 'p', args[p])

            u = np.reshape(np.full(sol.x).T, (2, self.N)).T
            ff_value = ff(u.T, args[p])  # compute optimal solution trajectory
            xx1[:, 0:3, mpciter + 1] = np.full(ff_value).T  # store the "predictions" here
            u_cl = [u_cl, u[0, :]]  # control actions. might have to append

            t[mpciter + 1] = t0
            [t0, x0, u0] = shift(t0, x0, u, f)

            xx[:, mpciter + 2] = x0
            mpciter = mpciter + 1

        ss_error = np.linalg.norm(x0 - xs)  # defaults to Euclidean norm

        return u_cl

    def shift(self, t0, x0, u, f):
        st = x0
        con = u[0, :].T  # propagate control action
        st = f(st, con)  # st+dt*f(st,con)
        x0 = scipy.sparse.csr_matrix.todense(st)  # convert sparse matrix to dense

        t0 = t0 + self.dt
        u0 = [u[1:size(u, 0), :], u[size(u, 0), :]]
        return t0, x0, u0
