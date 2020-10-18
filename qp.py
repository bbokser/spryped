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
https://github.com/MMehrez/ ...Sim_3_MPC_Robot_PS_obs_avoid_mul_sh.m
"""

import numpy as np
import numpy.matlib

import csv
import itertools

import casadi as cs

import wbc


class Qp:

    def __init__(self, controller, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = 0.025  # sampling time (s)
        self.controller = controller
        self.n_del_fr = None
        self.n_del_f = None

    def qpcontrol(self, ):

        del_fr_x = cs.SX.sym('del_fr_x')
        del_fr_y = cs.SX.sym('del_fr_y')
        del_fr_z = cs.SX.sym('del_fr_z')

        del_fr = [del_fr_x, del_fr_y, del_fr_z]  # reaction force relaxation vector
        self.n_del_fr = len(del_fr)

        del_f_x = cs.SX.sym('del_f_x')
        del_f_y = cs.SX.sym('del_f_y')
        del_f_z = cs.SX.sym('del_f_z')

        del_f = [del_f_x, del_f_y, del_f_z]  # floating base acceleration relaxation vector
        self.n_del_f = len(del_f)  # number of controls

        gravity = self.controller.grav
        dt = self.dt

        x_next = []

        self.fn = cs.Function('fn', [q_dd, gravity, A, J, S, f_r], x_next)  # nonlinear mapping of function f(x,u)

        obj = 0  # objective function
        constr = []  # constraints vector

        Q1 = np.zeros((6, 6))  # state weighing matrix
        Q[0, 0] = 1
        Q[1, 1] = 1
        Q[2, 2] = 1
        Q[3, 3] = 1
        Q[4, 4] = 1
        Q[5, 5] = 1

        Q2 = np.zeros((6, 6))  # control weighing matrix
        Q2[0, 0] = 1
        Q2[1, 1] = 1
        Q2[2, 2] = 1
        Q2[3, 3] = 1
        Q2[4, 4] = 1
        Q2[5, 5] = 1

        st = x[:, 0]  # initial state
        constr = cs.vertcat(constr, st - st_ref[0:self.n_states])  # initial condition constraints
        # compute objective and constraints

        obj = obj + cs.mtimes(cs.mtimes(del_fr.T, Q1), del_fr) + cs.mtimes(cs.mtimes(del_f.T, Q2), del_f)

        f_value = self.fn(st[0], st[1], st[2], st[3], st[4], st[5],
                          st[6], st[7], st[8], st[9], st[10], st[11],
                          con[0], con[1], con[2], con[3], con[4], con[5])
        st_n_e = np.array(f_value)
        constr = cs.vertcat(constr, st_next - st_n_e)  # compute constraints

        # TODO: add additional constraints

        opt_variables = cs.vertcat(del_fr, del_f)
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'verbose': 0, 'printLevel': "low"}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        # check this since we changed horizon length
        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-cs.inf, c_length))  # inequality constraints
        lbg[0:self.N] = itertools.repeat(0, self.N)  # dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        lbx = list(itertools.repeat(-500, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(500, o_length))  # input inequality constraints
        ubx[(self.n_states * (self.N + 1) + 2)::3] = [0 for i in range(20)]  # upper bound on all f1z and f2z

        if c_l == 0:  # if left leg is not in contact... don't calculate output forces for that leg.
            ubx[(self.n_states * (self.N + 1))::6] = [0 for i in range(10)]  # upper bound on all f1x
            ubx[(self.n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # upper bound on all f1y
            lbx[(self.n_states * (self.N + 1))::6] = [0 for i in range(10)]  # lower bound on all f1x
            lbx[(self.n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # lower bound on all f1y
            lbx[(self.n_states * (self.N + 1) + 2)::6] = [0 for i in range(10)]  # lower bound on all f1z
        if c_r == 0:  # if right leg is not in contact... don't calculate output forces for that leg.
            ubx[(self.n_states * (self.N + 1) + 3)::6] = [0 for i in range(10)]  # upper bound on all f2x
            ubx[(self.n_states * (self.N + 1) + 4)::6] = [0 for i in range(10)]  # upper bound on all f2y
            lbx[(self.n_states * (self.N + 1) + 3)::6] = [0 for i in range(10)]  # lower bound on all f2x
            lbx[(self.n_states * (self.N + 1) + 4)::6] = [0 for i in range(10)]  # lower bound on all f2y
            lbx[(self.n_states * (self.N + 1) + 5)::6] = [0 for i in range(10)]  # lower bound on all f2z

        # -------------Starting Simulation Loop Now------------------------------------- #
        # DM is very similar to SX, but with the difference that the nonzero elements are numerical values and
        # not symbolic expressions.
        # DM is mainly used for storing matrices in CasADi and as inputs and outputs of functions.
        # It is not intended to be used for computationally intensive calculations.
        # initial condition of the bot, gets updated every iteration (input)

        u0 = np.zeros((self.N, self.n_controls))  # six control inputs
        X0 = np.matlib.repmat(x_in, 1, self.N + 1).T  # initialization of the state's decision variables

        xx1 = []
        u_cl = np.zeros([1, 6])

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (self.n_states * (self.N + 1), 1)),
                        np.reshape(u0.T, (self.n_controls * self.N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][self.n_states * (self.N + 1):])
        u = np.reshape(solu.T, (self.n_controls, self.N)).T  # get controls from the solution

        solx = np.array(sol['x'][0:self.n_states * (self.N + 1)])
        # store the "predictions" here
        xx1 = np.append(xx1, np.reshape(solx.T, (self.n_states, self.N + 1)).T[0:self.n_states])

        u_cl = np.vstack([u_cl, u])  # control actions.

        # Get the solution trajectory
        X0 = np.reshape(solx.T, (self.n_states, self.N + 1)).T
        # Shift trajectory to initialize the next step
        X0 = np.append(X0[1:, :], X0[-1, :])

        u_cl = np.delete(u_cl, 0, axis=0)  # delete first row because it's just zeros for construction
        u_cl = u_cl[0, :]  # ignore rows other than new first row
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)

        # print("Time elapsed: ", t1 - t0)

        return u_cl

