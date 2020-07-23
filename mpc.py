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

from casadi import *
import numpy as np
import scipy

import control


class Mpc:

    def __init__(self, leg, m, dt=1e-3, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = dt  # sampling time (s)
        self.N = 3  # prediction horizon
        self.leg = leg
        self.m = m

        self.f_max = 5
        self.f_min = -self.f_max

    def mpcontrol(self, grav, Iinv, r1, r2, xs):
        # r1 = foot position
        theta = SX.sym('theta')
        p = SX.sym('p')
        omega = SX.sym('omega')
        pdot = SX.sym('pdot')
        states = np.array([theta, p, omega, pdot]).T  # state vector x
        n_states = len(states)  # number of states

        f1 = SX.sym('f1')  # controls
        f2 = SX.sym('f2')  # controls
        controls = np.array([f1, f2]).T
        n_controls = len(controls)  # number of controls

        g = ([SX.zeros(1, 3), SX.zeros(1, 3), SX.zeros(1, 3), grav.T]).T

        A = SX.eye(4)
        A[0, 2] = dot(Rz(phi), self.dt)  # define
        A[1, 3] = self.dt

        B = ([SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)])
        B[2, 0] = Iinv * r1 * self.dt
        B[2, 1] = Iinv * r2 * self.dt
        B[3, 0] = ones(3, 3) * self.dt / m
        B[3, 1] = ones(3, 3) * self.dt / m

        x_next = dot(A, states) + dot(B, controls) + g  # the discrete dynamics of the system

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
            st_next = fn(st, con)  # pass states and controls through function
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
        x0 = array([0, 0, 0, 0]).T  # initial condition, gets updated every iteration
        xs = array([0, 0, 0, 0]).T  # reference posture (INPUT?)

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
