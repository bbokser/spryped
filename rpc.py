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


# import transforms3d


def e2mat(x, y, z):
    # euler to rotation matrix conversion compatible with CasADI symbolics
    m = cs.SX(np.zeros((3, 3)))
    ch = cs.cos(z)  # TODO: Check if this should be z and not x
    sh = cs.sin(z)
    ca = cs.cos(y)
    sa = cs.sin(y)
    cb = cs.cos(x)
    sb = cs.sin(x)

    m[0, 0] = ch * ca
    m[0, 1] = sh * sb - ch * sa * cb
    m[0, 2] = ch * sa * sb + sh * cb
    m[1, 0] = sa
    m[1, 1] = ca * cb
    m[1, 2] = -ca * sb
    m[2, 0] = -sh * ca
    m[2, 1] = sh * sa * cb + ch * sb
    m[2, 2] = -sh * sa * sb + ch * cb
    return m


class Rpc:

    def __init__(self, mpc_dt, height, phi_switch, n, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = mpc_dt  # sampling time (s)
        self.N = n  # prediction horizon
        # horizon length = self.dt*self.N = .25 seconds
        self.mass = float(12.12427)  # kg
        self.mu = 0.5  # coefficient of friction
        self.b = 40 * np.pi / 180  # maximum kinematic leg angle
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
        self.height = height  # height of CoM from ground, must be positive  # TODO: Check with model
        self.beta = np.pi * 45 / 180  # max kinematic leg angle allowed
        self.pdot_d = np.array([0, 0, 0])  # desired movement speed

    def rpcontrol(self, b_orient, rz_phi, x_in, x_ref, s_phi_1, s_phi_2, pf_l, pf_r):

        # vector from CoM to hip in global frame (should just use body frame?)
        rh_l_g = np.dot(b_orient, self.rh_l)  # TODO: should this still be rz_phi?
        rh_r_g = np.dot(b_orient, self.rh_r)

        # actual initial footstep position vector from CoM to end effector in global coords
        pf_1_i = pf_l - rh_l_g + x_in[3:6]  # + np.array([0, 0, 0.8325])  # TODO: Check math
        pf_2_i = pf_r - rh_r_g + x_in[3:6]  # + np.array([0, 0, 0.8325])

        # inertia matrix inverse
        i_global = np.dot(np.dot(rz_phi, self.inertia), rz_phi.T)  # TODO: Check
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

        x_states = cs.SX(np.array([theta_x, theta_y, theta_z,
                                   p_x, p_y, p_z,
                                   omega_x, omega_y, omega_z,
                                   pdot_x, pdot_y, pdot_z]).T)  # state vector x

        f1 = cs.SX(np.array([f1_x, f1_y, f1_z]).T)  # controls
        f2 = cs.SX(np.array([f2_x, f2_y, f2_z]).T)  # controls

        A = cs.SX(np.array([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))

        B = cs.SX(np.array([[m * dt2, 0, 0, 0, 0, 0],
                            [0, m * dt2, 0, 0, 0, 0],
                            [0, 0, m * dt2, 0, 0, 0],
                            [0, 0, 0, i11 * dt2, i12 * dt2, i13 * dt2],
                            [0, 0, 0, i21 * dt2, i22 * dt2, i23 * dt2],
                            [0, 0, 0, i31 * dt2, i32 * dt2, i33 * dt2],
                            [m * dt, 0, 0, 0, 0, 0],
                            [0, m * dt, 0, 0, 0, 0],
                            [0, 0, m * dt, 0, 0, 0],
                            [0, 0, 0, i11 * dt, i12 * dt, i13 * dt],
                            [0, 0, 0, i21 * dt, i22 * dt, i23 * dt],
                            [0, 0, 0, i31 * dt, i32 * dt, i33 * dt]]))

        d = cs.SX(np.array([0, 0, dt2 * g, 0, 0, 0, 0, 0, dt * g, 0, 0, 0]).T)

        r1 = np.array([[0, -r1z, r1y],
                       [r1z, 0, -r1x],
                       [-r1y, r1x, 0]])

        r2 = np.array([[0, -r2z, r2y],
                       [r2z, 0, -r2x],
                       [-r2y, r2x, 0]])

        rz_t = np.transpose(rz_phi)
        rz_r1 = np.dot(rz_t, r1)
        rz_r2 = np.dot(rz_t, r2)
        h_1 = cs.SX(np.vstack((np.identity(3), rz_r1)))
        h_2 = cs.SX(np.vstack((np.identity(3), rz_r2)))

        fn = {}
        for k in range(0, self.N):
            # forces and torques acting on the CoM
            h = s_phi_1[k] * cs.mtimes(h_1, f1) + s_phi_2[k] * cs.mtimes(h_2, f2)
            # the discrete dynamics equation
            x_dyn = cs.mtimes(A, x_states) + cs.mtimes(B, h) + d
            fn["{0}".format(k)] = cs.Function('fn', [theta_x, theta_y, theta_z, p_x, p_y, p_z,
                                                     pdot_x, pdot_y, pdot_z, omega_x, omega_y, omega_z,
                                                     r1x, r1y, r1z, f1_x, f1_y, f1_z,
                                                     r2x, r2y, r2z, f2_x, f2_y, f2_z],
                                              [x_dyn])  # nonlinear mapping of function f(x,u)

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
        ground = np.array([0, 0, 1]).T  # ground plane, assumed level for now

        constr = cs.vertcat(constr, x[:, 0] - st_ref[0:n_states])  # initial condition constraints
        eta_chi = cs.SX.sym('eta_chi', n_states + n_controls, 1)  # H_x, heuristics
        # compute objective and constraints
        for k in range(0, self.N):
            st = x[:, k]  # state
            con = u[:, k]  # control action

            # analytic locomotion heuristics
            # hip centered stepping
            r_theta = e2mat(x=st[0], y=st[1], z=st[2])
            eta_hip_l = cs.mtimes(projection, cs.mtimes(r_theta, self.rh_l))
            eta_hip_r = cs.mtimes(projection, cs.mtimes(r_theta, self.rh_r))  # TODO: Check if using right frame
            # capture point
            pdot = np.array(([st[6], st[7], st[8]]))
            eta_cap = cs.sqrt(st[5] / 9.807) * (pdot - self.pdot_d)  # sqrt can cause NaN
            # vertical impulse scaling
            # if you get divide by zero here, schedulers are wrong--it'd be commanding both feet to swing at same time
            eta_imp = m * g * self.phi_switch / (s_phi_1[k] + s_phi_2[k])
            # centripetal acceleration
            theta_dot = np.array((st[9], st[10], st[11]))
            eta_cen = m * cs.cross(theta_dot, pdot)  # there is an approximation for this, p. 89

            eta_r1 = eta_cap + eta_hip_l
            eta_r2 = eta_cap + eta_hip_r
            eta_f = eta_imp + eta_cen
            eta_chi[0:12] = 0
            eta_chi[12:15] = eta_r1
            eta_chi[15:18] = eta_f
            eta_chi[18:21] = eta_r2
            eta_chi[21:24] = eta_f
            # calculate objective
            chi = cs.vertcat(st, con)
            chi_err = eta_chi - chi
            obj = obj + cs.mtimes(cs.mtimes(chi_err.T, W), chi_err)

            # calculate dynamics constraint
            st_next = x[:, k + 1]
            st_n_e = fn[str(k)](st[0], st[1], st[2], st[3], st[4], st[5],
                                st[6], st[7], st[8], st[9], st[10], st[11],
                                con[0], con[1], con[2], con[3], con[4], con[5],
                                con[6], con[7], con[8], con[9], con[10], con[11])
            constr = cs.vertcat(constr, st_next - st_n_e)  # compute constraints

        # footstep positions transferred from body (r_i) to world (p_i) coordinates
        pf_1 = cs.horzcat(pf_1_i, u[0:3, 0:self.N] + x[0:3, 0:self.N])  # TODO: Replace first value or append?
        # pf_1 = u[0:3, 0:self.N] + x[0:3, 0:self.N]
        # pf_1[:, 0] = pf_1_i
        pf_2 = cs.horzcat(pf_2_i, u[6:9, 0:self.N] + x[6:9, 0:self.N])  # 3x11
        # pf_2 = u[6:9, 0:self.N] + x[6:9, 0:self.N]
        # pf_2[:, 0] = pf_2_i

        zg = 0  # estimated ground height, for terrain perception. Currently zero bc there is no terrain perception
        # add additional constraints

        for k in range(0, self.N):
            # Foot placed on ground (=0)
            constr = cs.vertcat(constr, s_phi_1[k] * (zg - pf_1[2, k]))
            constr = cs.vertcat(constr, s_phi_2[k] * (zg - pf_2[2, k]))
            # Foot stationary during stance (=0)
            constr = cs.vertcat(constr, s_phi_1[k + 1] * s_phi_1[k] * (pf_1[2, k + 1] - pf_1[2, k]))
            constr = cs.vertcat(constr, s_phi_2[k + 1] * s_phi_2[k] * (pf_2[2, k + 1] - pf_2[2, k]))

        len_eq = np.shape(constr)[0]  # Length of the equality constraints TODO: Check if this is correct

        for k in range(0, self.N):
            # Kinematic leg limits (<=0)
            # TODO: p.71 should it really be - rh, or + rh?
            # constr = cs.vertcat(constr, s_phi_1[k] * (cs.norm_2(u[0:3, k] - self.rh_l) - x[5, k] * np.tan(self.beta)))
            # constr = cs.vertcat(constr, s_phi_2[k] * (cs.norm_2(u[6:9, k] - self.rh_r) - x[5, k] * np.tan(self.beta)))
            # Positive ground force normal (<=0)
            constr = cs.vertcat(constr, -s_phi_1[k] * u[3:6, k] * ground)
            constr = cs.vertcat(constr, -s_phi_2[k] * u[9:12, k] * ground)
            # Lateral Force Friction Pyramids (<=0)
            constr = cs.vertcat(constr, u[3, k] - self.mu * u[5, k])  # f1x - mu*f1z
            constr = cs.vertcat(constr, -u[3, k] - self.mu * u[5, k])  # -f1x - mu*f1z

            constr = cs.vertcat(constr, u[4, k] - self.mu * u[5, k])  # f1y - mu*f1z
            constr = cs.vertcat(constr, -u[4, k] - self.mu * u[5, k])  # -f1y - mu*f1z

            constr = cs.vertcat(constr, u[9, k] - self.mu * u[11, k])  # f2x - mu*f2z
            constr = cs.vertcat(constr, -u[9, k] - self.mu * u[11, k])  # -f2x - mu*f2z

            constr = cs.vertcat(constr, u[10, k] - self.mu * u[11, k])  # f2y - mu*f2z
            constr = cs.vertcat(constr, -u[10, k] - self.mu * u[11, k])  # -f2y - mu*f2z

        st_len = n_states * (self.N + 1)

        ct_len = n_controls * self.N
        opt_variables = cs.vertcat(cs.reshape(x, st_len, 1),
                                   cs.reshape(u, ct_len, 1))

        nlp_prob = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'ipopt.print_level': 0, 'ipopt.acceptable_tol': 1e-3,
                'ipopt.acceptable_obj_change_tol': 1e-3, "verbose": False}
        # "ipopt.hessian_approximation": "limited-memory"}
        solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        # constraint upper and lower bounds
        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        # IC + dynamics + foot equality constraint
        lbg[0:len_eq] = itertools.repeat(0, len_eq)  # set lower limit of equality constraints to zero
        ubg = list(itertools.repeat(0, c_length))  # default upper limit of inequality constraints is zero

        # upper and lower bounds for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        lbx[0:st_len:n_states] = itertools.repeat(-np.pi / 4, self.N + 1)  # set lower limit of pitch
        lbx[1:st_len:n_states] = itertools.repeat(-np.pi / 4, self.N + 1)  # set lower limit of roll
        lbx[5:st_len:n_states] = itertools.repeat(0.5, self.N + 1)  # set lower limit of p_z
        lbx[st_len + 0::n_controls] = itertools.repeat(-0.5, self.N)  # set lower limit of r1_x
        lbx[st_len + 1::n_controls] = itertools.repeat(-0.25, self.N)  # set lower limit of r1_y  # TODO: Check CAD
        lbx[st_len + 2::n_controls] = itertools.repeat(-0.9, self.N)  # set lower limit of r1_z
        lbx[st_len + 3::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f1_x
        lbx[st_len + 4::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f1_y
        lbx[st_len + 5::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f1_z
        lbx[st_len + 6::n_controls] = itertools.repeat(-0.5, self.N)  # set lower limit of r2_x
        lbx[st_len + 7::n_controls] = itertools.repeat(-0.5, self.N)  # set lower limit of r2_y
        lbx[st_len + 8::n_controls] = itertools.repeat(-0.9, self.N)  # set lower limit of r2_z
        lbx[st_len + 9::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f2_x
        lbx[st_len + 10::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f2_y
        lbx[st_len + 11::n_controls] = itertools.repeat(0, self.N)  # set lower limit of f2_z

        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints
        ubx[0:st_len:n_states] = itertools.repeat(np.pi / 4, self.N + 1)  # set upper limit of pitch
        ubx[1:st_len:n_states] = itertools.repeat(np.pi / 4, self.N + 1)  # set upper limit of roll
        ubx[st_len + 0::n_controls] = itertools.repeat(0.5, self.N)  # set upper limit of r1_x
        ubx[st_len + 1::n_controls] = itertools.repeat(0.5, self.N)  # set upper limit of r1_y
        ubx[st_len + 2::n_controls] = itertools.repeat(-0.4, self.N)  # set upper limit of r1_z
        ubx[st_len + 3::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f1_x
        ubx[st_len + 4::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f1_y
        ubx[st_len + 5::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f1_z
        ubx[st_len + 6::n_controls] = itertools.repeat(0.5, self.N)  # set upper limit of r2_x
        ubx[st_len + 7::n_controls] = itertools.repeat(0.25, self.N)  # set upper limit of r2_y
        ubx[st_len + 8::n_controls] = itertools.repeat(-0.4, self.N)  # set upper limit of r2_z
        ubx[st_len + 9::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f2_x
        ubx[st_len + 10::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f2_y
        ubx[st_len + 11::n_controls] = itertools.repeat(200, self.N)  # set upper limit of f2_z

        # setup is finished, now solve-------------------------------------------------------------------------------- #

        u0 = np.zeros((self.N, n_controls))  # control inputs
        X0 = np.matlib.repmat(x_in, 1, self.N + 1).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (st_len, 1)),
                        np.reshape(u0.T, (ct_len, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][st_len:])
        u = np.reshape(solu.T, (self.N, n_controls)).T  # get controls from the solution
        # print("u = ", u)
        u_cl = u[:, 0]  # ignore columns other than new first column
        print("u_cl = ", u_cl)
        r_l = u_cl[0:3] + rh_l_g  # convert back to hip coord sys
        fmpc_l = u_cl[3:6]
        r_r = u_cl[6:9] + rh_r_g  # convert back to hip coord sys
        fmpc_r = u_cl[9:12]
        # print("Time elapsed for MPC: ", t1 - t0)
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)
        return r_l, fmpc_l, r_r, fmpc_r
