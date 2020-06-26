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
"""

import time
import sys
import curses

import numpy as np
import transforms3d
import pybullet as p
import pybullet_data

GRAVITY = -9.807

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
planeID = p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

bot = p.loadURDF("spryped_urdf_rev06/urdf/spryped_urdf_rev06.urdf", [0, 0, 1.2],
                 robotStartOrientation, useFixedBase=1, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

p.setGravity(0, 0, GRAVITY)

# print(p.getJointInfo(bot, 3))
np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, leg_left, leg_right,
                 controller_left, controller_right,
                 mpc_left, mpc_right, contact_left, contact_right, dt=1e-3):
        self.dt = dt
        self.tau_l = None
        self.tau_r = None
        self.leg_left = leg_left
        self.leg_right = leg_right
        self.controller_left = controller_left
        self.controller_right = controller_right
        self.mpc_left = mpc_left
        self.mpc_right = mpc_right
        self.init_alpha = -np.pi / 2
        self.init_beta = 0  # can't control, ee Jacobian is zeros in that row
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -0.6, self.init_alpha, self.init_beta, self.init_gamma])
        self.target_l = self.target_init
        self.target_r = self.target_init
        self.s_r = 1
        self.s_l = 1
        self.delay_term = 0

    def run(self):
        # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")
        p.setTimeStep(self.dt)
        useRealTime = 0
        p.setRealTimeSimulation(useRealTime)

        jointArray = range(p.getNumJoints(bot))

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(bot)):
            p.setJointMotorControl2(bot, i, p.VELOCITY_CONTROL, force=0.5)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(bot, i, 1)  # enable joint torque sensing

        steps = 0

        while 1:
            time.sleep(self.dt)
            # update target after specified period of time passes
            steps = steps + 1

            base_or_p = np.array(p.getBasePositionAndOrientation(bot)[1])
            # pybullet gives quaternions in xyzw format
            # transforms3d takes quaternions in wxyz format, so you need to shift values
            base_orientation = np.zeros(4)
            base_orientation[0] = base_or_p[3]  # w
            base_orientation[1] = base_or_p[0]  # x
            base_orientation[2] = base_or_p[1]  # y
            base_orientation[3] = base_or_p[2]  # z
            base_orientation = transforms3d.quaternions.quat2mat(base_orientation)

            # self.target_r = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
            self.tau_r = self.controller_right.control(
                leg=self.leg_right, target=self.target_r, base_orientation=base_orientation)
            u_r = self.leg_right.apply_torque(u=self.tau_r, dt=self.dt)
            # self.target_l = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
            self.tau_l = self.controller_left.control(
                leg=self.leg_left, target=self.target_l, base_orientation=base_orientation)
            u_l = self.leg_left.apply_torque(u=self.tau_l, dt=self.dt)

            '''
            if self.statemachine() == 1:
                self.target_r = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
                self.tau_r = self.controller_right.control(leg=self.leg_right, target=self.target_r)
                u_r = self.leg_right.apply_torque(u=self.tau_r, dt=self.dt)

                # u_l = self.mpc_left.mpcontrol(leg=self.leg_left)  # and positions, velocities

            else:
                self.target_l = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
                self.tau_l = self.controller_left.control(leg=self.leg_left, target=self.target_l)
                u_l = self.leg_left.apply_torque(u=self.tau_l, dt=self.dt)

                # u_r = self.mpc_right.mpcontrol(leg=self.leg_right)  # and positions, velocities
            '''

            # tau_d_left = self.contact_left.contact(leg=self.leg_left, g=self.leg_left.grav)
            
            torque = np.zeros(8)
            torque[0:4] = u_l
            torque[0] *= -1  # readjust to match motor polarity
            torque[4:8] = -u_r
            torque[7] *= -1  # readjust to match motor polarity
            # print(torque)
            p.setJointMotorControlArray(bot, jointArray, p.TORQUE_CONTROL, forces=torque)

            omega = p.getBaseVelocity(bot)[1]  # base angular velocity in global coordinates

            # fw kinematics
            # print(np.transpose(np.append(np.dot(base_orientation, self.leg_left.position()[:, -1]),
            #                              np.dot(base_orientation, self.leg_right.position()[:, -1]))))
            # joint velocity
            # print("vel = ", self.leg_left.velocity())
            # encoder feedback
            # print(np.transpose(np.append(self.leg_left.q, self.leg_right.q)))

            # sys.stdout.write("\033[F")  # back to previous line
            # sys.stdout.write("\033[K")  # clear line

            if useRealTime == 0:
                p.stepSimulation()

    def statemachine(self):
        # finite state machine
        left_torque_check = self.reaction_torques()[0:4]
        right_torque_check = self.reaction_torques()[4:8]
        if left_torque_check <= right_torque_check:
            self.s_r = 1  # right swing
        elif something:
            self.s_r = 0  # left swing
        if somethingelse:
            self.s_l = 1
        elif somethingother:
            self.s_l = 0
        return self.s_r, self.s_l

    def reaction_torques(self):
        # returns joint reaction torques
        reaction_force = [j[2] for j in p.getJointStates(bot, range(8))]  # j[2]=jointReactionForces
        #  [Fx, Fy, Fz, Mx, My, Mz]
        reaction_force = np.array(reaction_force)
        torques = reaction_force[:, 4]  # selected all joints My
        torques[0] = reaction_force[0, 5]  # selected joint 1 Mz
        torques[4] = reaction_force[4, 5]  # selected joint 5 Mz
        return torques

    def get_states(self):
        self.q = [j[0] for j in p.getJointStates(bot, range(8))]
        self.dq = [j[1] for j in p.getJointStates(bot, range(8))]
        state = append(state_q, state_dq)
        return state
