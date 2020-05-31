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

import transformations
import control


class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of the end-effector.
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

        # self.kv = np.zeros((3, 3))
        # self.kv[0, 0] = 10
        # self.kv[1, 1] = 10
        # self.kv[2, 2] = 10

        self.kd = np.zeros((4, 4))
        self.kd[0, 0] = 50
        self.kd[1, 1] = 50
        self.kd[2, 2] = 50
        self.kd[3, 3] = 100

        self.ko = np.zeros((3, 3))
        self.ko[0, 0] = 100
        self.ko[1, 1] = 100
        self.ko[2, 2] = 100

        self.kn = np.zeros((4, 4))
        self.kn[0, 0] = 0
        self.kn[1, 1] = 0
        self.kn[2, 2] = 0
        self.kn[3, 3] = 10

        self.knd = np.zeros((4, 4))
        self.knd[0, 0] = 0
        self.knd[1, 1] = 0
        self.knd[2, 2] = 0
        self.knd[3, 3] = 1

    def control(self, leg, x_dd_des=None):
        """
        Generates a control signal to move the
        joints to the specified target.

        leg Leg: the leg model being controlled
        des list: the desired system position
        x_dd_des np.array: desired acceleration
        """
        # which dim to control of [x, y, z, alpha, beta, gamma]
        ctrlr_dof = np.array([True, True, True, False, False, False])

        # calculate the Jacobian
        JEE = leg.gen_jacEE()[ctrlr_dof]

        # generate the mass matrix in end-effector space
        Mq = leg.gen_Mq()
        Mx = leg.gen_Mx(Mq=Mq, JEE=JEE)

        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        self.x = leg.position()[:, -1]

        # Calculate operational space velocity vector
        # self.velocity = (np.transpose(np.dot(JEE, leg.dq)).flatten())[0:3]

        # x_dd_des[:3] = np.dot(self.kp, (self.target[0:3] - self.x)) + np.dot(self.kv, -self.velocity)
        x_dd_des[:3] = np.dot(self.kp, (self.target[0:3] - self.x))

        # calculate end effector orientation unit quaternion
        q_e = leg.orientation()

        # calculate the target orientation unit quaternion
        q_d = transformations.unit_vector(
            transformations.quaternion_from_euler(
                self.target[3], self.target[4], self.target[5],
                axes='rxyz'))  # converting angles from 'rotating xyz'

        # calculate the rotation between current and target orientations
        q_r = transformations.quaternion_multiply(
            q_d, transformations.quaternion_conjugate(q_e))

        # convert rotation quaternion to Euler angle forces
        x_dd_des[3:] = np.dot(self.ko, q_r[1:] * np.sign(q_r[0]))

        x_dd_des = x_dd_des[ctrlr_dof]  # get rid of dim not being controlled

        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # calculate force
        Fx = np.dot(Mx, x_dd_des)

        # self.u = (np.dot(JEE.T, Fx).reshape(-1, )) - leg.gen_grav()

        # add in velocity compensation in GC space for stability
        self.u = (np.dot(JEE.T, Fx).reshape(-1, ) -
                  np.dot(Mq, np.dot(self.kd, leg.dq)).flatten()) - leg.gen_grav()

        # if null_control is selected, add a control signal in the
        # null space to try to move the leg to selected position
        if self.null_control:
            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            prop_val = ((leg.ee_angle() - leg.q) + np.pi) % (np.pi * 2) - np.pi
            q_des = (np.dot(self.kn, prop_val) +
                     np.dot(self.knd, -leg.dq.reshape(-1, )))

            Fq_null = np.dot(Mq, q_des)

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))

            null_filter = np.eye(len(leg.L)) - np.dot(JEE.T, Jdyn_inv)

            null_signal = np.dot(null_filter, Fq_null).reshape(-1, )

            self.u += null_signal

        if self.leveler:
            # keeps toes level (for now)
            u_level = np.dot(self.kn, leg.ee_angle()-leg.q)
            self.u += u_level

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, leg)

        return self.u

    def gen_target(self, leg):
        target_alpha = -np.pi / 2
        target_beta = 0  # can't control, ee Jacobian is zeros in that row
        target_gamma = 0
        self.target = np.array([0, 0, -0.8325, target_alpha, target_beta, target_gamma])

        return self.target.tolist()
