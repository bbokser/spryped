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

import pybullet as p
import pybullet_data

# from osc import Control
# from robot import Robot

GRAVITY = -9.807

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
planeID = p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

bot = p.loadURDF("spryped_urdf_rev06/urdf/spryped_urdf_rev06.urdf", [0, 0, 0.8],
                 robotStartOrientation, useFixedBase=0, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

p.setGravity(0, 0, GRAVITY)

# print(p.getJointInfo(bot, 3))
np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, dt=1e-3):
        self.dt = dt
        self.tau_l = None
        self.tau_r = None

    def run(self, leg_left, leg_right, controller_left, controller_right):

        self.leg_left = leg_left
        self.leg_right = leg_right
        self.controller_left = controller_left
        self.controller_right = controller_right

        p.setTimeStep(self.dt)

        useRealTime = 0
        p.setRealTimeSimulation(useRealTime)

        jointArray = range(p.getNumJoints(bot))

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(bot)):
            p.setJointMotorControl2(bot, i, p.VELOCITY_CONTROL, force=0.5)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(bot, i, 1)  # enable joint torque sensing

        # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        steps = 0

        while 1:
            time.sleep(self.dt)
            # update target after specified period of time passes
            steps = steps + 1

            if steps == 1:
                self.target_l = self.controller_left.gen_target(self.leg_left)
                self.target_r = self.controller_right.gen_target(self.leg_right)
            else:
                self.target_l = self.controller_left.target
                self.target_r = self.controller_right.target

            self.tau_l = self.controller_left.control(self.leg_left)
            self.tau_r = self.controller_right.control(self.leg_right)
            torque = np.zeros(8)

            u_l = self.leg_left.apply_torque(u=self.tau_l, dt=self.dt)
            u_r = self.leg_right.apply_torque(u=self.tau_r, dt=self.dt)
            torque[0:4] = u_l
            torque[0] *= -1  # readjust to match motor polarity
            torque[4:8] = -u_r
            torque[7] *= -1  # readjust to match motor polarity
            # print(torque)

            baseorientation = p.getBasePositionAndOrientation(bot)[1]
            # print(p.getEulerFromQuaternion(baseorientation))

            omega = p.getBaseVelocity(bot)[1]  # base angular velocity in global coordinates
            print(omega)

            # fw kinematics
            # print(np.transpose(np.append((leg_left.position()[:, -1]), (leg_right.position()[:, -1]))))
            # joint velocity
            # print("vel = ", leg_left.velocity())
            # encoder feedback
            # print(np.transpose(np.append(leg_left.q, leg_right.q)))

            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line

            p.setJointMotorControlArray(bot, jointArray, p.TORQUE_CONTROL, forces=torque)

            if useRealTime == 0:
                p.stepSimulation()
        '''
        def reaction(self):
            reaction_force = [j[2] for j in p.getJointStates(bot, range(7))]  # j[2]=jointReactionForces
            #  [Fx, Fy, Fz, Mx, My, Mz]
            # print(states[1][4]) #'%s' % float('%.1g' % pront[1])) # selected joint 2, My
            return reaction_force[:][4]  # selected joint 1 and 5 Mz, all other joints My
        '''

        def get_states(self):
            self.q = [j[0] for j in p.getJointStates(bot, range(7))]
            self.dq = [j[1] for j in p.getJointStates(bot, range(7))]
            state = append(state_q, state_dq)
            return state
