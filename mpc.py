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

import control


class Mpc:

    def __init__(self, leg, dt=1e-3, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = dt  # sampling time (s)
        self.N = 3  # prediction horizon

        self.f_max = 5
        self.f_min = -self.f_max

    def control(self, leg, x_dd_des=None):

        theta = SX.sym('theta')
        p = SX.sym('p')
        omega = SX.sym('omega')
        pdot = SX.sym('pdot')
        states = array([theta, p, omega, pdot]).T  # state vector x
        n_states = len(states)  # number of states

        f1 = SX.sym('f1')  # controls
        f2 = SX.sym('f2')  # controls
        controls = array([f1, f2]).T
        n_controls = len(controls)  # number of controls

        g = ([SX.zeros(1, 3), SX.zeros(1, 3), SX.zeros(1, 3), grav.T]).T

        A = SX.eye(4)
        A[0, 2] = dot(Rz(phi), dt)  # define
        A[1, 3] = dt

        B = ([SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)],
             [SX.zeros(3, 3), SX.zeros(3, 3)])
        B[2, 0] = gIinv1*dt
        B[2, 1] = gIinv2*dt
        B[3, 0] = ones(3, 3)*self.dt/m
        B[3, 1] = ones(3, 3)*self.dt/m

        x_next = dot(A, states) + dot(B, controls) + g  # the discrete dynamics of the system

        fn = Function('fn', [states, controls], x_next)  # nonlinear mapping of function f(x,u)
        u = SX.sym('u', n_controls, self.N)  # decision variables, control action matrix
        param = SX.sym('param', n_states + n_states)  # parameters, including initial and reference

        x = SX.sym('x', n_states, (self.N+1))

        # compute solution symbolically
        x[:, 0] = P[0:3]  # check to make sure slicing is done correctly here!
        'In Python, slicing is left inclusive and right exclusive, '
        'whereas in MATLAB slicing is inclusive at both.'
        'Matlab uses one based indexing while python uses zero based indexing.'
        for k in range(0, self.N-1):  # N-1 because of python zero based indexing
            st = x[:, k]
            con = u[:, k]
            st_next = fn(st, con)
            x[:, k+1] = st_next
        # function for optimal traj knowing optimal sol
        ff = Function('ff', [u, param], x)

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
        for k in range(0, self.N-1):  # 0 and N-1 because of python zero based indexing
            st = x[:, k]
            con = u[:, k]  # control action
            # calculate objective
            obj = obj + dot(dot((st-param[3:6]).T, Q), st-param[3:6]) \
                + dot(dot(con.T, R), con)
        # compute constraints
        for k in range(0, self.N):  # would be N+1 in matlab
            constr = np.vstack(constr, x(0, k))
            constr = np.vstack(constr, x(1, k))
            constr = np.vstack(constr, x(2, k))
            constr = np.vstack(constr, x(3, k))

        # make decision variables one column vector
        opt_variables = reshape(u, 2*self.N, 1)
        qp = {'f', obj, 'x', opt_variables, 'constr', constr, 'param', param}

        solver = qpsol('S', 'qpoases', qp)
        args = {lbg: -2,  # inequality constraints: lower bound
                ubg: 2,  # inequality constraints: upper bound
                lbx: f_min,  # input constraints: lower bound
                ubx: f_max,  # input constraints: upper bound
                }

        t0 = 0
        x0 = array([0, 0, 0, 0]).T  # initial condition
        xs = array([0, 0, 0, 0]).T  # reference posture

        xx[:, 0] = x0  # contains history of states
        t[0] = t0

        # start MPC
        mpciter = 0
        xx1 = []
        u_cl = []

        while np.linalg.norm(x0 - xs) > 1e-2 and mpciter < sim_tim / T:
            args[p] = array([x0, xs]).T  # set values of parameters vector
            args[x0] = np.reshape(u0.T, (2*N, 1))  # init value of optimization variables

            sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,
                         'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p)

            u = np.reshape(np.full(sol.x).T, (2, self.N)).T
            ff_value = ff(u.T, args.p)  # compute optimal solution trajectory
            xx1[:, 0:3, mpciter+1] = np.full(ff_value).T

            u_cl = [u_cl, u[0, :]]
            t[mpciter+1] = t0  # assign new start time for next iteration
            [t0, x0, u0] = shift(t0, x0, u, f)

            xx[:, mpciter+2] = x0
            mpciter = mpciter + 1

        ss_error = np.linalg.norm(x0-xs)  # defaults to Euclidean norm

    def shift(self, t0, x0, u, f):
        st = x0
        con = u[0, :].T
        st = f(st, con)
        x0 = np.full(st)

        t0 = t0 + self.dt
        u0 = [u[1:size(u, 0), :], u[size(u, 0), :]]
        return t0, x0, u0
