"""
Copyright (C) 2013 Travis DeWolf
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
import sys

import numpy as np
import transforms3d

import control


class Control(control.Control):
    """
    A controller that implements whole body control.
    Controls the (x,y,z) position of the end-effector.
    """

    def __init__(self, dt=1e-3, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Control, self).__init__(**kwargs)
        self.dt = dt
        self.DOF = 3  # task space dimensionality
        self.vel_comp = False  # velocity compensation in GC space
        self.null_control = True
        self.leveler = False

        self.kp = np.zeros((3, 3))
        self.kp[0, 0] = 2000
        self.kp[1, 1] = 2000
        self.kp[2, 2] = 2000

        self.kv = np.zeros((3, 3))
        self.kv[0, 0] = 100
        self.kv[1, 1] = 100
        self.kv[2, 2] = 100

        self.ko = np.zeros((3, 3))
        self.ko[0, 0] = 1000
        self.ko[1, 1] = 1000
        self.ko[2, 2] = 1000

        self.kl = np.zeros((4, 4))
        self.kl[3, 3] = 10

        self.kn = np.zeros((4, 4))
        self.kn[0, 0] = 10
        self.kn[1, 1] = 10  # 100
        self.kn[2, 2] = 10
        self.kn[3, 3] = 10

        self.kd = np.zeros((4, 4))
        np.fill_diagonal(self.kd, 1)

        self.kf = np.zeros((3, 3))
        self.kf[0, 0] = 0.01
        self.kf[1, 1] = 0.01
        self.kf[2, 2] = 0.01

        self.Mq = None
        self.Mx = None
        self.x_dd_des = None
        self.J = None
        self.x = None
        self.grav = None
        self.velocity = None
        self.q_e = None
        self.ctrlr_dof = np.array([True, True, True, False, False, False])

    def wb_control(self, leg, target, b_orient, force, x_dd_des=None):
        """
        Generates a control signal to apply a specified force vector.

        leg Leg: the leg model being controlled
        des list: the desired system position
        """
        self.target = target
        self.b_orient = np.array(b_orient)

        # which dim to control of [x, y, z, alpha, beta, gamma]
        ctrlr_dof = self.ctrlr_dof

        # calculate the Jacobian
        JEE = leg.gen_jacEE()[ctrlr_dof]  # print(np.linalg.matrix_rank(JEE))
        # rank of matrix is 3, can only control 3 DOF with one OSC

        # generate the mass matrix in end-effector space
        self.Mq = leg.gen_Mq()
        Mx = leg.gen_Mx(Mq=self.Mq, JEE=JEE)

        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        # multiply with rotation matrix for base to world
        self.x = np.dot(b_orient, leg.position()[:, -1])  # select last position value to get EE xyz
        # self.x = leg.position()[:, -1]

        # calculate operational space velocity vector
        self.velocity = np.dot(b_orient, (np.transpose(np.dot(JEE, leg.dq)).flatten())[0:3])

        # calculate linear acceleration term based on PD control
        x_dd_des[:3] = np.dot(self.kp, (self.target[0:3] - self.x)) + np.dot(self.kv, -self.velocity)

        # Orientation-Control------------------------------------------------------------------------------#
        # calculate end effector orientation unit quaternion
        self.q_e = leg.orientation(b_orient=b_orient)

        # calculate the target orientation unit quaternion
        q_d = transforms3d.euler.euler2quat(self.target[3], self.target[4], self.target[5], axes='sxyz')
        q_d = q_d / np.linalg.norm(q_d)  # convert to unit vector quaternion

        # calculate the rotation between current and target orientations
        q_r = transforms3d.quaternions.qmult(
            q_d, transforms3d.quaternions.qconjugate(self.q_e))
        # q_r = transforms3d.quaternions.qmult(
        #     q_r, transforms3d.euler.euler2quat(-np.pi/2, 0, 0, axes='sxyz'))
        # self.q_r = q_r
        # convert rotation quaternion to Euler angle forces
        x_dd_des[3:] = np.dot(self.ko, q_r[1:] * np.sign(q_r[0]))
        # x_dd_des[3:] = np.dot(self.ko, transforms3d.euler.quat2euler(q_r, axes='rxyz'))
        # ------------------------------------------------------------------------------------------------#

        x_dd_des = x_dd_des[ctrlr_dof]  # get rid of dim not being controlled

        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # calculate force
        Fx = np.dot(Mx, x_dd_des)
        Aq_dd = (np.dot(JEE.T, Fx).reshape(-1, ))

        self.grav = leg.gen_grav(b_orient=b_orient)

        if force is None:
            force_control = 0
        else:
            Fr = np.dot(b_orient, force)
            force_control = (np.dot(JEE.T, Fr).reshape(-1, ))

        k_force = 10  # 5.9
        self.u = Aq_dd - self.grav - force_control*k_force
        self.x_dd_des = x_dd_des
        self.Mx = Mx
        self.J = JEE
        # self.u = (np.dot(JEE.T, Fx).reshape(-1, )) - self.grav + (np.dot(JEE.T, Fr).reshape(-1, ))

        # add in velocity compensation in GC space for stability
        if self.vel_comp is True:
            self.u += -np.dot(self.Mq, np.dot(self.kd, leg.dq)).flatten()

        # if null_control is selected, add a control signal in the
        # null space to try to move the leg to selected position
        if self.null_control is True:
            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            leg_des_angle = np.array([-np.pi / 2, np.pi * 32 / 180, -np.pi * 44.18 / 180, np.pi * 12.18 / 180.])
            prop_val = ((leg_des_angle - leg.q) + np.pi) % (np.pi * 2) - np.pi
            q_des = (np.dot(self.kn, prop_val))
            #        + np.dot(self.knd, -leg.dq.reshape(-1, )))
            Fq_null = np.dot(self.Mq, q_des)
            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(self.Mq)))
            null_filter = np.eye(len(leg.L)) - np.dot(JEE.T, Jdyn_inv)
            null_signal = np.dot(null_filter, Fq_null).reshape(-1, )
            self.u += null_signal

        if self.leveler is True:
            # keeps ee pitch level
            base_y = transforms3d.euler.mat2euler(b_orient, axes='ryxz')[0]  # get y axis rotation of base
            q1 = leg.q[1]
            q2 = leg.q[2]
            # keep ee level
            ee_target = -(q1 + q2)
            angles = np.array([0, np.pi * 32 / 180, 0, ee_target])
            u_level = np.dot(self.kl, -base_y + angles - leg.q)
            self.u += u_level

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, leg)

        return self.u
