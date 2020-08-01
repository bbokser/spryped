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

    def __init__(self, leg, dt=1e-3, null_control=False, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Control, self).__init__(**kwargs)
        self.dt = dt
        self.DOF = 3  # task space dimensionality
        self.null_control = null_control
        self.leveler = True

        self.kp = np.zeros((3, 3))
        self.kp[0, 0] = 2000
        self.kp[1, 1] = 2000
        self.kp[2, 2] = 2000

        self.kv = np.zeros((3, 3))
        self.kv[0, 0] = 100
        self.kv[1, 1] = 100
        self.kv[2, 2] = 100
        '''
        self.kd = np.zeros((4, 4))
        self.kd[0, 0] = 50
        self.kd[1, 1] = 50
        self.kd[2, 2] = 50
        self.kd[3, 3] = 100
        '''
        self.ko = np.zeros((3, 3))
        self.ko[0, 0] = 100
        self.ko[1, 1] = 100
        self.ko[2, 2] = 100

        self.kn = np.zeros((4, 4))
        self.kn[0, 0] = 0
        self.kn[1, 1] = 0
        self.kn[2, 2] = 0
        self.kn[3, 3] = 10

        self.Mq = None
        self.x = None
        self.grav = None
        self.velocity = None
        self.q_e = None

    def control(self, leg, target, b_orient, x_dd_des=None):
        """
        Generates a control signal to move the
        joints to the specified target.

        leg Leg: the leg model being controlled
        des list: the desired system position
        x_dd_des np.array: desired acceleration
        """
        self.target = target
        self.b_orient = np.array(b_orient)

        # which dim to control of [x, y, z, alpha, beta, gamma]
        ctrlr_dof = np.array([True, True, True, False, False, False])

        # calculate the Jacobian
        JEE = leg.gen_jacEE()[ctrlr_dof]

        # generate the mass matrix in end-effector space
        self.Mq = leg.gen_Mq()
        Mx = leg.gen_Mx(Mq=self.Mq, JEE=JEE)

        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        self.x = np.dot(b_orient, leg.position()[:, -1])  # multiply with rotation matrix for base to world

        # calculate operational space velocity vector
        self.velocity = np.dot(b_orient, (np.transpose(np.dot(JEE, leg.dq)).flatten())[0:3])

        # calculate linear acceleration term based on PD control
        x_dd_des[:3] = np.dot(self.kp, (self.target[0:3] - self.x)) + np.dot(self.kv, -self.velocity)

        # calculate end effector orientation unit quaternion
        self.q_e = leg.orientation(b_orient=b_orient)

        # calculate the target orientation unit quaternion
        q_d = transforms3d.euler.euler2quat(self.target[3], self.target[4], self.target[5], axes='rxyz')
        q_d = q_d / np.linalg.norm(q_d)  # convert to unit vector quaternion

        # calculate the rotation between current and target orientations
        q_r = transforms3d.quaternions.qmult(
            q_d, transforms3d.quaternions.qconjugate(self.q_e))

        # convert rotation quaternion to Euler angle forces
        x_dd_des[3:] = np.dot(self.ko, q_r[1:] * np.sign(q_r[0]))

        x_dd_des = x_dd_des[ctrlr_dof]  # get rid of dim not being controlled

        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # calculate force
        Fx = np.dot(Mx, x_dd_des)

        self.grav = leg.gen_grav(b_orient=b_orient)

        self.u = (np.dot(JEE.T, Fx).reshape(-1, )) - self.grav

        # add in velocity compensation in GC space for stability
        # self.u = np.dot(JEE.T, Fx).reshape(-1, ) \
        #     - np.dot(Mq, np.dot(self.kd, leg.dq)).flatten() - self.grav

        # if null_control is selected, add a control signal in the
        # null space to try to move the leg to selected position
        if self.null_control:
            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            prop_val = ((leg.ee_angle() - leg.q) + np.pi) % (np.pi * 2) - np.pi
            q_des = (np.dot(self.kn, prop_val))
            #        + np.dot(self.knd, -leg.dq.reshape(-1, )))

            Fq_null = np.dot(Mq, q_des)

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))

            null_filter = np.eye(len(leg.L)) - np.dot(JEE.T, Jdyn_inv)

            null_signal = np.dot(null_filter, Fq_null).reshape(-1, )

            self.u += null_signal

        if self.leveler:
            # keeps toes level (for now)
            base_y = transforms3d.euler.mat2euler(b_orient, axes='ryxz')[0]  # get y axis rotation of base
            u_level = np.dot(self.kn, -base_y + leg.ee_angle() - leg.q)
            self.u += u_level

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, leg)

        return self.u
