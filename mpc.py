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

import time
import numpy as np
import numpy.matlib

import csv
import itertools

import casadi as cs


class Mpc:

    def __init__(self, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = 0.025  # sampling time (s)
        self.N = 10  # prediction horizon
        # horizon length = self.dt*self.N = .25 seconds
        self.mass = float(12.12427)  # kg
        self.mu = 0.7  # coefficient of friction
        self.b = 40 * np.pi / 180  # maximum kinematic leg angle
        self.fn = None
        self.n_states = None
        self.n_controls = None

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

        self.rh_r = np.array([.14397, .13519, .03581])  # vector from CoM to hip
        self.rh_l = np.array([-.14397, .13519, .03581])  # vector from CoM to hip

    def mpcontrol(self, rz_phi, r1, r2, x_in, x_ref, c_l, c_r):  # p_l, p_r, c_l, c_r):

        i_global = np.dot(np.dot(rz_phi, self.inertia), rz_phi.T)
        i_inv = np.linalg.inv(i_global)
        '''
        # vector from CoM to hip in global frame (should just use body frame?)
        rh_l_g = np.dot(rz_phi, self.rh_l)
        rh_r_g = np.dot(rz_phi, self.rh_r)

        # desired footstep position in global coords
        pf_l = x_in[1] + r1
        pf_r = x_in[1] + r2
        print(pf_r)
        '''
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

        # r = foot position
        r1x = r1[0]
        r1y = r1[1]
        r1z = r1[2]
        r2x = r2[0]
        r2y = r2[1]
        r2z = r2[2]

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
        self.n_states = len(states)  # number of states

        f1_x = cs.SX.sym('f1_x')  # controls
        f1_y = cs.SX.sym('f1_y')  # controls
        f1_z = cs.SX.sym('f1_z')  # controls
        f2_x = cs.SX.sym('f2_x')  # controls
        f2_y = cs.SX.sym('f2_y')  # controls
        f2_z = cs.SX.sym('f2_z')  # controls
        controls = [f1_x, f1_y, f1_z, f2_x, f2_y, f2_z]
        self.n_controls = len(controls)  # number of controls

        gravity = -9.807
        dt = self.dt
        mass = self.mass
        # gravity = cs.SX.sym("gravity")
        # dt = cs.SX.sym("dt")
        # mass = cs.SX.sym("mass")
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

        self.fn = cs.Function('fn', [theta_x, theta_y, theta_z,
                                     p_x, p_y, p_z,
                                     omega_x, omega_y, omega_z,
                                     pdot_x, pdot_y, pdot_z,
                                     f1_x, f1_y, f1_z,
                                     f2_x, f2_y, f2_z],
                              x_next)  # nonlinear mapping of function f(x,u)

        u = cs.SX.sym('u', self.n_controls, self.N)  # decision variables, control action matrix
        st_ref = cs.SX.sym('st_ref', self.n_states + self.n_states)  # initial and reference states
        x = cs.SX.sym('x', self.n_states, (self.N + 1))  # represents the states over the opt problem.
        # st_n_e = np.zeros((self.n_states, (self.N + 1)))  # represents left hand side of the dynamics eq

        obj = 0  # cs.SX(0)  # objective function
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

        st = x[:, 0]  # initial state
        constr = cs.vertcat(constr, st - st_ref[0:self.n_states])  # initial condition constraints
        # compute objective and constraints
        for k in range(0, self.N):  # 0 because of python zero based indexing
            st = x[:, k]  # state
            con = u[:, k]  # control action
            # calculate objective
            obj = obj + cs.mtimes(cs.mtimes((st - st_ref[self.n_states:(self.n_states * 2)]).T, Q),
                                  st - st_ref[self.n_states:(self.n_states * 2)]) + cs.mtimes(cs.mtimes(con.T, R), con)
            st_next = x[:, k + 1]
            f_value = self.fn(st[0], st[1], st[2], st[3], st[4], st[5],
                              st[6], st[7], st[8], st[9], st[10], st[11],
                              con[0], con[1], con[2], con[3], con[4], con[5])
            st_n_e = np.array(f_value)
            constr = cs.vertcat(constr, st_next - st_n_e)  # compute constraints

        # add additional constraints
        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[0, k] - self.mu * u[2, k])  # f1x - mu*f1z

        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[1, k] - self.mu * u[2, k])  # f1y - mu*f1z

        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[3, k] - self.mu * u[5, k])  # f2x - mu*f2z

        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[4, k] - self.mu * u[5, k])  # f2y - mu*f2z

        opt_variables = cs.vertcat(cs.reshape(x, self.n_states * (self.N + 1), 1),
                                   cs.reshape(u, self.n_controls * self.N, 1))
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

        # print("Time elapsed for MPC: ", t1 - t0)

        return u_cl

