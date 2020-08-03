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
"""

import numpy as np
import scipy
import csv

from casadi import *

import control


class Mpc:

    def __init__(self, dt=1e-3, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = dt  # sampling time (s)
        self.N = 3  # prediction horizon
        self.mass = float(12.12427)  # kg
        self.gravity = np.array([[0, 0, -9.807]])

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
        # r1 = foot position
        theta_x = SX.sym('theta_x')
        theta_y = SX.sym('theta_y')
        theta_z = SX.sym('theta_z')
        theta = np.array([theta_x, theta_y, theta_z])
        p_x = SX.sym('p_x')
        p_y = SX.sym('p_y')
        p_z = SX.sym('p_z')
        p = np.array([p_x, p_y, p_z])
        omega_x = SX.sym('omega_x')
        omega_y = SX.sym('omega_y')
        omega_z = SX.sym('omega_z')
        omega = np.array([omega_x, omega_y, omega_z])
        pdot_x = SX.sym('pdot_x')
        pdot_y = SX.sym('pdot_y')
        pdot_z = SX.sym('pdot_z')
        pdot = np.array([pdot_x, pdot_y, pdot_z])
        states = np.array([theta.T, p.T, omega.T, pdot.T]).T  # state vector x
        n_states = len(states)  # number of states

        f1_x = SX.sym('f1_x')  # controls
        f1_y = SX.sym('f1_y')  # controls
        f1_z = SX.sym('f1_z')  # controls
        f1 = np.array([f1_x, f1_y, f1_z])
        f2_x = SX.sym('f2_x')  # controls
        f2_y = SX.sym('f2_y')  # controls
        f2_z = SX.sym('f2_z')  # controls
        f2 = np.array([f2_x, f2_y, f2_z])
        controls = np.array([f1.T, f2.T]).T
        n_controls = len(controls)  # number of controls

        g = np.zeros((3, 4))
        g[0:4, 3:] = self.gravity.T
        g = g.T
        print(g)
        A = np.array([[np.ones((3, 3)), np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3)), np.ones((3, 3))],
                      [np.zeros((3, 3)), np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.ones((3, 3))]])
        A[0, 2] *= rz_phi * self.dt  # define
        A[1, 3] *= self.dt

        B = np.array([[np.zeros((3, 3)), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.zeros((3, 3))],
                      [np.ones((3, 3)), np.ones((3, 3))]])
        B[2, 0] = i_inv * r1 * self.dt
        B[2, 1] = i_inv * r2 * self.dt
        B[3, 0] *= self.dt / self.mass
        B[3, 1] *= self.dt / self.mass

        x_next = np.dot(A, states) + np.dot(B, controls) + g  # the discrete dynamics of the system

        fn = Function('fn', [states, controls], x_next)  # nonlinear mapping of function f(x,u)
        u = SX.sym('u', n_controls, self.N)  # decision variables, control action matrix
        st_ref = SX.sym('st_ref', n_states + n_states)  # initial and reference states

        x = SX.sym('x', n_states, (self.N + 1))  # represents the states over the opt problem.

        # compute solution symbolically
        x[:, 0] = st_ref[0:3]  # initial state
        'In Python, slicing is left inclusive and right exclusive, '
        'whereas in MATLAB slicing is inclusive at both.'
        'Matlab uses one based indexing while python uses zero based indexing.'
        for k in range(0, self.N - 1):  # N-1 because of python zero based indexing
            st = x[:, k]  # extract the previous state from x
            con = u[:, k]  # extract controls from control matrix
            # st_next = fn(st, con)  # pass states and controls through function
            f_value = fn(st, con)
            st_next = st + (self.dt * f_value)
            x[:, k + 1] = st_next

        # function for optimal traj knowing optimal sol
        ff = Function('ff', [u, st_ref], x)

        obj = 0  # objective function
        constr = []  # constraints vector

        Q = zeros(4, 4)  # state weighing matrix
        Q[0, 0] = 1
        Q[1, 1] = 1
        Q[2, 2] = 1
        Q[3, 3] = 1

        R = zeros(2, 2)  # control weighing matrix
        R[0, 0] = 1
        R[1, 1] = 1

        # compute objective
        for k in range(0, self.N - 1):  # 0 and N-1 because of python zero based indexing
            st = x[:, k + 1]
            con = u[:, k]  # control action
            # calculate objective
            obj = obj + dot(dot((st - st_ref[3:6]).T, Q), st - st_ref[3:6]) \
                  + dot(dot(con.T, R), con)

        # compute constraints
        for k in range(0, self.N):  # would be N+1 in matlab
            constr = np.vstack(constr, x(0, k))
            constr = np.vstack(constr, x(1, k))
            constr = np.vstack(constr, x(2, k))
            constr = np.vstack(constr, x(3, k))

        # make decision variables one column vector
        opt_variables = reshape(u, 2 * self.N, 1)
        qp = {'f', obj, 'x', opt_variables, 'constr', constr, 'st_ref', st_ref}

        solver = qpsol('S', 'qpoases', qp)

        args = {lbg: -2,  # inequality constraints: lower bound
                ubg: 2,  # inequality constraints: upper bound
                lbx: self.f_min,  # input constraints: lower bound
                ubx: self.f_max,  # input constraints: upper bound
                }

        # -------------Starting Simulation Loop Now------------------------------------- #
        t0 = 0
        x0 = array([0, 0, 0, 0]).T  # initial condition, gets updated every iteration (input)
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
