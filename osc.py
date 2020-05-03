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
from simple_pid import PID
import transformations
import control

class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of the end-effector.
    """

    def __init__(self, dt=1e-3, null_control=False, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Control, self).__init__(**kwargs)
        self.dt = dt
        self.DOF = 3  # task space dimensionality
        self.null_control = null_control
        self.pid = PID(self.kp, self.ki, self.kd)
        self.pid.sample_time = self.dt  # 1e-3

    def control(self, robot, x_dd_des=None):
        """
        Generates a control signal to move the
        joints to the specified target.

        robot Robot: the robot model being controlled
        des list: the desired system position
        x_dd_des np.array: desired acceleration
        """
        # which dim to control of [x, y, z, alpha, beta, gamma]
        ctrlr_dof = np.array([True, True, True, False, True, False])

        # calculate the Jacobian
        JEE = robot.gen_jacEE()[ctrlr_dof]

        # generate the mass matrix in end-effector space
        Mq = robot.gen_Mq()
        Mx = robot.gen_Mx(Mq=Mq, JEE=JEE)

        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        # self.pid.setpoint = self.target
        self.x = robot.position()[:, -1]

        # Calculate operational space linear velocity vector
        # self.velocity = np.transpose(np.dot(JEE, robot.dq)).flatten()

        # x_dd_des[:3] = self.kp * (self.target[0:3] - self.x)
        self.pid.setpoint = self.target[0:3]
        x_dd_des[:3] = self.pid(self.x)

        # calculate end effector orientation unit quaternion
        q_e = robot.orientation()

        # calculate the target orientation unit quaternion
        q_d = transformations.unit_vector(
            transformations.quaternion_from_euler(
                self.target[3], self.target[4], self.target[5],
                axes='rxyz'))  # converting angles from 'rotating xyz'

        # calculate the rotation between current and target orientations
        q_r = transformations.quaternion_multiply(
            q_d, transformations.quaternion_conjugate(q_e))

        # convert rotation quaternion to Euler angle forces
        x_dd_des[3:] = self.ko * q_r[1:] * np.sign(q_r[0])

        x_dd_des = x_dd_des[ctrlr_dof]  # get rid of dim not being controlled
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # calculate force
        Fx = np.dot(Mx, x_dd_des)

        # generate gravity term in task space
        tau_grav = robot.gen_grav()
        # add in velocity compensation in GC space for stability
        self.u = (np.dot(JEE.T, Fx).reshape(-1, )) - tau_grav
        # self.u = (np.dot(JEE.T, Fx).reshape(-1, ) -
        #          np.dot(Mq, self.kd * robot.dq).flatten()) - tau_grav

        # inverse kinematics basic proportional control
        # self.u = self.kp*(robot.inv_kinematics(self.target)-robot.q)

        # ik simple_pid
        # self.pid.setpoint = robot.inv_kinematics(self.target)
        # self.u = self.pid(robot.q)

        # if null_control is selected, add a control signal in the
        # null space to try to move the robot to selected position
        if self.null_control:

            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            prop_val = ((robot.angles - robot.q) + np.pi) % (np.pi*2) - np.pi
            # print("prop = ", np.shape(prop_val))
            q_des = (self.kp * prop_val +
                     self.kd * -robot.dq.reshape(-1, ))
            # print("q_des = ", np.shape(q_des))

            u_null = np.dot(Mq, q_des)

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))

            null_filter = np.eye(len(robot.L)) - np.dot(JEE.T, Jdyn_inv)

            null_signal = np.dot(null_filter, u_null).reshape(-1,)
            # print("null = ", np.shape(null_signal))
            self.u += null_signal

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, robot)

        return self.u

    def gen_target(self, robot):
        target_alpha = 0
        target_beta = -2*np.pi  # keep foot flat for now
        target_gamma = 0
        self.target = np.array([0.077, 0.131, -0.708, target_alpha, target_beta, target_gamma])

        return self.target.tolist()
