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
        self.controller = controller

    def qpcontrol(self, fr_mpc):

        n_del_fr = 3  # reaction force relaxation vector (del_f_x, y, z)
        n_del_f = 4  # floating base acceleration relaxation vector (q1, q2, q3, q4)

        g = self.controller.grav  # joint space gravity-induced torque

        Q1 = np.zeros((6, 6))  # state weighing matrix
        Q1[0, 0] = 1
        Q1[1, 1] = 1
        Q1[2, 2] = 1
        Q1[3, 3] = 1
        Q1[4, 4] = 1
        Q1[5, 5] = 1

        Q2 = np.zeros((6, 6))  # control weighing matrix
        Q2[0, 0] = 1
        Q2[1, 1] = 1
        Q2[2, 2] = 1
        Q2[3, 3] = 1
        Q2[4, 4] = 1
        Q2[5, 5] = 1

        # compute objective
        del_fr = cs.SX.sym('del_fr', n_del_fr)  # decision variables, control action matrix
        del_f = cs.SX.sym('del_f', n_del_f)  # represents the states over the opt problem.

        obj = cs.mtimes(cs.mtimes(del_fr.T, Q1), del_fr) + cs.mtimes(cs.mtimes(del_f.T, Q2), del_f)

        # compute constraints
        A = np.dot(self.controller.J.T, self.controller.Mx).reshape(-1, )
        q_dd_des = np.dot(self.controller.J.T, self.controller.x_dd_des).reshape(-1, )
        fr = cs.SX.sym('fr', 3)  # ground reaction force
        q_dd = cs.SX.sym('q_dd', 4)  # resultant joint acceleration
        constr = []  # constraints vector
        constr = cs.vertcat(constr, cs.mtimes(A, q_dd) - g - cs.mtimes(self.controller.J.T, fr))  # Aq + g = J.T*fr
        constr = cs.vertcat(constr, q_dd - q_dd_des - del_f)  # q_dd = q_dd_cmd + del_f
        constr = cs.vertcat(constr, fr - fr_mpc - del_fr)  # fr = fr_mpc + del_fr
        constr = cs.vertcat(constr, fr)  # fr >= 0

        opt_variables = cs.vertcat(del_fr, del_f)
        # param = cs.SX.sym('param', )  # contains initial and reference states
        qp = {'x': opt_variables, 'f': obj, 'g': constr}
        opts = {'print_time': 0, 'error_on_fail': 0, 'verbose': 0, 'printLevel': "low"}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        # check this since we changed horizon length
        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(0, c_length))  # equality constraints
        ubg = list(itertools.repeat(0, c_length))  # equality constraints
        ubg[3] = 1e10  # fr >= 0

        # opt variable constraints
        lbx = list(itertools.repeat(-500, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(500, o_length))  # input inequality constraints

        # setup is finished, now solve-------------------------------------------------------------------------------- #
        # parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.zeros(n_del_fr), np.zeros(n_del_f))  # can't do vertcat, inconsistent sizing?

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        sol_del_f = np.array(sol['x'][n_del_fr:])

        sol_del_fr = np.array(sol['x'][0:n_del_fr])

        return sol_del_fr, sol_del_f

