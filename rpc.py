# coding=utf-8
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

G. Bledt, “Regularized predictive control framework for robust dynamic legged locomotion,” thesis, 2020.

https://www.youtube.com/watch?v=RrnkPrcpyEA
https://github.com/MMehrez/ ...Sim_3_MPC_Robot_PS_obs_avoid_mul_sh.m
"""

# import time
import numpy as np
import numpy.matlib
import csv
import itertools

import casadi as cs
import transforms3d


class Rpc:

    def __init__(self, phi_switch, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = 0.025  # sampling time (s)
        self.N = 10  # prediction horizon
        # horizon length = self.dt*self.N = .25 seconds
        self.mass = float(12.12427)  # kg
        self.mu = 0.5  # coefficient of friction
        self.b = 40 * np.pi / 180  # maximum kinematic leg angle
        self.fn = None
        self.phi_switch = phi_switch

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

        self.rh_r = np.array([.14397, .13519, .03581])  # vector from CoM to hip in the body frame
        self.rh_l = np.array([-.14397, .13519, .03581])  # vector from CoM to hip in the body frame
        self.pdot_d = np.array([0, 0, 0])  # desired movement speed

    def rpcontrol(self, rz_phi, r1, r2, x_in, x_ref, s_phi_1, s_phi_2):
        # inertia matrix inverse
        i_global = np.dot(np.dot(rz_phi, self.inertia), rz_phi.T)  # TODO: Check
        i_inv = np.linalg.inv(i_global)
        '''
        # vector from CoM to hip in global frame (should just use body frame?)
        rh_l_g = np.dot(rz_phi, self.rh_l)
        rh_r_g = np.dot(rz_phi, self.rh_r)

        # actual footstep position vector from CoM to end effector in global coords
        r1 = r1 + rh_l_g + x_in[1]
        r2 = r2 + rh_r_g + x_in[1]

        # desired footstep position in global coords
        pf_l = x_in[1] + r1
        pf_r = x_in[1] + r2
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

        # controls
        f1_x = cs.SX.sym('f1_x')
        f1_y = cs.SX.sym('f1_y')
        f1_z = cs.SX.sym('f1_z')
        f2_x = cs.SX.sym('f2_x')
        f2_y = cs.SX.sym('f2_y')
        f2_z = cs.SX.sym('f2_z')
        r1x = cs.SX.sym('r1x')
        r1y = cs.SX.sym('r1y')
        r1z = cs.SX.sym('r1z')
        r2x = cs.SX.sym('r2x')
        r2y = cs.SX.sym('r2y')
        r2z = cs.SX.sym('r2z')
        controls = [r1x, r1y, r1z,
                    f1_x, f1_y, f1_z,
                    r2x, r2y, r2z,
                    f2_x, f2_y, f2_z]
        n_controls = len(controls)  # number of controls

        g = -9.807
        dt = self.dt
        m = self.mass
        dt2 = 0.5 * (dt ** 2)

        x_next = [dt * omega_x + dt2 * m * (f1_x * s_phi_1 + f2_x * s_phi_2) + theta_x,
                  dt * omega_y + dt2 * m * (f1_y * s_phi_1 + f2_y * s_phi_2) + theta_y,
                  dt * omega_z + dt2 * g + dt2 * m * (f1_z * s_phi_1 + f2_z * s_phi_2) + theta_z,
                  dt * pdot_x + dt2 * i11 * (s_phi_1 * (
                          f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt2 * i12 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt2 * i13 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + p_x,
                  dt * pdot_y + dt2 * i21 * (s_phi_1 * (
                          f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt2 * i22 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt2 * i23 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + p_y,
                  dt * pdot_z + dt2 * i31 * (s_phi_1 * (
                          f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt2 * i32 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt2 * i33 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + p_z,
                  dt * m * (f1_x * s_phi_1 + f2_x * s_phi_2) + omega_x,
                  dt * m * (f1_y * s_phi_1 + f2_y * s_phi_2) + omega_y,
                  dt * g + dt * m * (f1_z * s_phi_1 + f2_z * s_phi_2) + omega_z,
                  dt * i11 * (s_phi_1 * (f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt * i12 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt * i13 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + pdot_x,
                  dt * i21 * (s_phi_1 * (f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt * i22 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt * i23 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + pdot_y,
                  dt * i31 * (s_phi_1 * (f1_x * (-r1y * rz31 + r1z * rz21) + f1_y * (r1x * rz31 - r1z * rz11) + f1_z * (
                          -r1x * rz21 + r1y * rz11)) + s_phi_2 * (f2_x * (-r2y * rz31 + r2z * rz21) + f2_y * (
                          r2x * rz31 - r2z * rz11) + f2_z * (-r2x * rz21 + r2y * rz11))) + dt * i32 * (s_phi_1 * (
                          f1_x * (-r1y * rz32 + r1z * rz22) + f1_y * (r1x * rz32 - r1z * rz12) + f1_z * (
                          -r1x * rz22 + r1y * rz12)) + s_phi_2 * (f2_x * (-r2y * rz32 + r2z * rz22) + f2_y * (
                          r2x * rz32 - r2z * rz12) + f2_z * (-r2x * rz22 + r2y * rz12))) + dt * i33 * (s_phi_1 * (
                          f1_x * (-r1y * rz33 + r1z * rz23) + f1_y * (r1x * rz33 - r1z * rz13) + f1_z * (
                          -r1x * rz23 + r1y * rz13)) + s_phi_2 * (f2_x * (-r2y * rz33 + r2z * rz23) + f2_y * (
                          r2x * rz33 - r2z * rz13) + f2_z * (-r2x * rz23 + r2y * rz13))) + pdot_z]

        self.fn = cs.Function('fn', [theta_x, theta_y, theta_z,
                                     p_x, p_y, p_z,
                                     pdot_x, pdot_y, pdot_z,
                                     omega_x, omega_y, omega_z,
                                     r1x, r1y, r1z,
                                     f1_x, f1_y, f1_z,
                                     r2x, r2y, r2z,
                                     f2_x, f2_y, f2_z],
                              x_next)  # nonlinear mapping of function f(x,u)

        u = cs.SX.sym('u', n_controls, self.N)  # decision variables, control action matrix
        st_ref = cs.SX.sym('st_ref', n_states + n_states)  # initial and reference states
        x = cs.SX.sym('x', n_states, (self.N + 1))  # represents the states over the opt problem.

        obj = 0  # objective function
        constr = []  # constraints vector
        k = 10
        Q = np.zeros((12, 12))  # state weighing matrix
        Q[0, 0] = k
        Q[1, 1] = k
        Q[2, 2] = k
        Q[3, 3] = k
        Q[4, 4] = k
        Q[5, 5] = k
        Q[6, 6] = k
        Q[7, 7] = k
        Q[8, 8] = k
        Q[9, 9] = k
        Q[10, 10] = k
        Q[11, 11] = k

        R = np.zeros((12, 12))  # control weighing matrix
        R[0, 0] = k / 2
        R[1, 1] = k / 2
        R[2, 2] = k / 2
        R[3, 3] = k / 2
        R[4, 4] = k / 2
        R[5, 5] = k / 2
        R[6, 6] = k / 2
        R[7, 7] = k / 2
        R[8, 8] = k / 2
        R[9, 9] = k / 2
        R[10, 10] = k / 2
        R[11, 11] = k / 2

        W = np.concatenate((np.concatenate((Q, np.zeros((12, 12))), axis=1),
                            np.concatenate((np.zeros((12, 12)), R), axis=1)), axis=0)
        projection = np.eye(3)
        projection[2, 2] = 0  # projection matrix for zero ground height assumption

        constr = cs.vertcat(constr, x[:, 0] - st_ref[0:n_states])  # initial condition constraints
        eta_chi = cs.SX.sym('eta_chi', n_states + n_controls, 1)  # H_x, heuristics
        # compute objective and constraints
        for k in range(0, self.N):
            st = x[:, k]  # state
            con = u[:, k]  # control action

            # analytic locomotion heuristics
            # hip centered stepping
            r_theta = np.array(transforms3d.euler.euler2mat(st[0], st[1], st[2], axes='sxyz'))
            eta_hip_l = cs.mtimes(projection, cs.mtimes(r_theta, self.rh_l))
            eta_hip_r = cs.mtimes(projection, cs.mtimes(r_theta, self.rh_r))
            # capture point
            pdot = np.array(([st[6], st[7], st[8]]))
            eta_cap = cs.sqrt(st[5] / 9.807) * (pdot - self.pdot_d)
            # vertical impulse scaling
            eta_imp = m * g * self.phi_switch / (s_phi_1 + s_phi_2)
            # centripetal acceleration
            theta_dot = np.array((st[9], st[10], st[11]))
            eta_cen = m * cs.cross(theta_dot, pdot)  # there is an approximation for this, p. 89

            # calculate objective
            eta_r1 = eta_hip_l + eta_cap
            eta_r2 = eta_hip_r + eta_cap
            eta_f = eta_imp + eta_cen
            # eta_chi = [24, 1]
            eta_chi[0:12] = 0
            eta_chi[12:15] = eta_r1
            eta_chi[15:18] = eta_f
            eta_chi[18:21] = eta_r2
            eta_chi[21:24] = eta_f
            print(eta_chi)
            chi = cs.vertcat(st, con)
            chi_err = eta_chi - chi
            obj = obj + cs.mtimes(cs.mtimes(chi_err.T, W), chi_err)
            """
            obj = obj + cs.mtimes(cs.mtimes((st - st_ref[n_states:(n_states * 2)]).T, Q),
                                  st - st_ref[n_states:(n_states * 2)]) \
                + cs.mtimes(cs.mtimes(con.T, R), con)
                """
            # calculate dynamics constraint
            st_next = x[:, k + 1]
            f_value = self.fn(st[0], st[1], st[2], st[3], st[4], st[5],
                              st[6], st[7], st[8], st[9], st[10], st[11],
                              con[0], con[1], con[2], con[3], con[4], con[5],
                              con[6], con[7], con[8], con[9], con[10], con[11])
            st_n_e = np.array(f_value)
            constr = cs.vertcat(constr, st_next - st_n_e)  # compute constraints

        # add additional constraints
        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[0, k] - self.mu * u[2, k])  # f1x - mu*f1z
            constr = cs.vertcat(constr, -u[0, k] - self.mu * u[2, k])  # -f1x - mu*f1z

            constr = cs.vertcat(constr, u[1, k] - self.mu * u[2, k])  # f1y - mu*f1z
            constr = cs.vertcat(constr, -u[1, k] - self.mu * u[2, k])  # -f1y - mu*f1z

            constr = cs.vertcat(constr, u[3, k] - self.mu * u[5, k])  # f2x - mu*f2z
            constr = cs.vertcat(constr, -u[3, k] - self.mu * u[5, k])  # -f2x - mu*f2z

            constr = cs.vertcat(constr, u[4, k] - self.mu * u[5, k])  # f2y - mu*f2z
            constr = cs.vertcat(constr, -u[4, k] - self.mu * u[5, k])  # -f2y - mu*f2z

        opt_variables = cs.vertcat(cs.reshape(x, n_states * (self.N + 1), 1),
                                   cs.reshape(u, n_controls * self.N, 1))
        nlp_prob = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'ipopt.print_level': 0, 'ipopt.acceptable_tol': 1e-6,
                'ipopt.acceptable_obj_change_tol': 1e-6}
        solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:(self.N + 1)] = itertools.repeat(0, self.N + 1)  # IC + dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        st_len = n_states * (self.N + 1)

        lbx[(st_len + 2)::3] = [0 for i in range(20)]  # lower bound on all f1z and f2z

        if c_l == 0:  # if left leg is not in contact... don't calculate output forces for that leg.
            ubx[(n_states * (self.N + 1))::6] = [0 for i in range(10)]  # upper bound on all f1x
            ubx[(n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # upper bound on all f1y
            lbx[(n_states * (self.N + 1))::6] = [0 for i in range(10)]  # lower bound on all f1x
            lbx[(n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # lower bound on all f1y
            ubx[(n_states * (self.N + 1) + 2)::6] = [0 for i in range(10)]  # upper bound on all f1z

        if c_r == 0:  # if right leg is not in contact... don't calculate output forces for that leg.
            ubx[(n_states * (self.N + 1) + 3)::6] = [0 for i in range(10)]  # upper bound on all f2x
            ubx[(n_states * (self.N + 1) + 4)::6] = [0 for i in range(10)]  # upper bound on all f2y
            lbx[(n_states * (self.N + 1) + 3)::6] = [0 for i in range(10)]  # lower bound on all f2x
            lbx[(n_states * (self.N + 1) + 4)::6] = [0 for i in range(10)]  # lower bound on all f2y
            ubx[(n_states * (self.N + 1) + 5)::6] = [0 for i in range(10)]  # upper bound on all f2z

        # setup is finished, now solve-------------------------------------------------------------------------------- #

        u0 = np.zeros((self.N, n_controls))  # six control inputs
        X0 = np.matlib.repmat(x_in, 1, self.N + 1).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (n_states * (self.N + 1), 1)),
                        np.reshape(u0.T, (n_controls * self.N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][n_states * (self.N + 1):])
        u = np.reshape(solu.T, (n_controls, self.N)).T  # get controls from the solution

        u_cl = u[0, :]  # ignore rows other than new first row
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)

        # print("Time elapsed for MPC: ", t1 - t0)

        return u_cl
