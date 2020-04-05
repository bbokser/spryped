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

import control

class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of the end-effector.
    """

    def __init__(self, dt=1e-3, null_control=True, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Control, self).__init__(**kwargs)
        self.dt = dt
        self.DOF = 3  # task space dimensionality
        self.null_control = null_control
        self.pid = PID(10, 3, 0.5)
        self.pid.sample_time = self.dt # 1e-3


    def control(self, robot, x_dd_des=None):
        """
        Generates a control signal to move the
        joints to the specified target.

        robot Robot: the robot model being controlled
        des list: the desired system position
        x_dd_des np.array: desired task-space acceleration,
                        system goes to self.target if None
        """
        # calculate the Jacobian
        JEE = robot.gen_jacEE()

        # calculate desired end-effector acceleration
        if x_dd_des is None:
            # self.pid.setpoint = self.target
            self.x = robot.position()[:, -1]
            self.velocity = np.transpose(np.dot(JEE, robot.dq)).flatten()
            # compute new output from the PID according to the system's current value
            # x_dd_des = self.pid(self.x)
            x_dd_des = self.kp * (self.target - self.x)
            # x_dd_des = np.add(self.kp * (self.target - self.x), -self.kd * self.velocity)

        # generate the mass matrix in end-effector space
        Mq = robot.gen_Mq()
        Mx = robot.gen_Mx()

        # calculate force
        Fx = np.dot(Mx, x_dd_des)

        # generate gravity term in task space
        tau_grav = robot.gen_grav()

        # add in velocity compensation in GC space for stability
        # self.u = (np.dot(JEE.T, Fx).reshape(-1, )) # + tau_grav
        self.u = (np.dot(JEE.T, Fx).reshape(-1, ) -
                  np.dot(Mq, self.kd * robot.dq).flatten()) + tau_grav

        # simple inverse kinematics PID control
        # self.u = self.kp*(robot.inv_kinematics(self.target)-robot.q)
        # self.pid.setpoint = robot.inv_kinematics(self.target)
        # self.u = self.pid(robot.q)

        # if null_control is selected and the task space has
        # fewer DOFs than the robot, add a control signal in the
        # null space to try to move the robot to its resting state
        # if self.null_control and self.DOF < len(robot.L):

        # calculate our secondary control signal
        # calculated desired joint angle acceleration
        # prop_val = ((robot.rest_angles - robot.q) + np.pi) % (np.pi*2) - np.pi
        # q_des = (self.kp * prop_val + \
        #          self.kd * -robot.dq).reshape(-1,)

        # Mq = robot.gen_Mq()
        # u_null = np.dot(Mq, q_des)

        # calculate the null space filter
        # Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
        # null_filter = np.eye(len(robot.L)) - np.dot(JEE.T, Jdyn_inv)

        # null_signal = np.dot(null_filter, u_null).reshape(-1,)

        # self.u += null_signal

        # if self.write_to_file is True:
        # feed recorders their signals
        #     self.u_recorder.record(0.0, self.u)
        #     self.xy_recorder.record(0.0, self.x)
        #     self.dist_recorder.record(0.0, self.target - self.x)

        # add in any additional signals
        for addition in self.additions:
            self.u += addition.generate(self.u, robot)

        return self.u

    def gen_target(self, robot):
        # Generate a random target
        # gain = np.sum(robot.L) * .75
        # bias = -np.sum(robot.L) * 0

        # self.target = np.random.random(size=(3,)) * gain + bias

        self.target = np.array([0.077, 0.131, -0.708])

        return self.target.tolist()
