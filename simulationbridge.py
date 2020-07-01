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
# physicsClient = p.connect(p.GUI)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
# planeID = p.loadURDF("plane.urdf")
p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

bot = p.loadURDF("spryped_urdf_rev06/urdf/spryped_urdf_rev06.urdf", [0, 0, 1.2],
                 robotStartOrientation, useFixedBase=1,
                 flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

p.setGravity(0, 0, GRAVITY)

jointArray = range(p.getNumJoints(bot))

useRealTime = 0

def reaction_torques():
    # returns joint reaction torques
    reaction_force = [j[2] for j in p.getJointStates(bot, range(8))]  # j[2]=jointReactionForces
    #  [Fx, Fy, Fz, Mx, My, Mz]
    reaction_force = np.array(reaction_force)
    torques = reaction_force[:, 4]  # selected all joints My
    torques[0] = reaction_force[0, 5]  # selected joint 1 Mz
    torques[4] = reaction_force[4, 5]  # selected joint 5 Mz
    return torques


class Sim:

    def __init__(self, dt=1e-3):
        self.dt = dt

        # print(p.getJointInfo(bot, 3))
        # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")
        p.setTimeStep(self.dt)

        p.setRealTimeSimulation(useRealTime)

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(bot)):
            p.setJointMotorControl2(bot, i, p.VELOCITY_CONTROL, force=0.5)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(bot, i, 1)  # enable joint torque sensing

    def sim_run(self, u_l, u_r):
        
        time.sleep(self.dt)
        # update target after specified period of time passes

        base_or_p = np.array(p.getBasePositionAndOrientation(bot)[1])
        # pybullet gives quaternions in xyzw format
        # transforms3d takes quaternions in wxyz format, so you need to shift values
        base_orientation = np.zeros(4)
        base_orientation[0] = base_or_p[3]  # w
        base_orientation[1] = base_or_p[0]  # x
        base_orientation[2] = base_or_p[1]  # y
        base_orientation[3] = base_or_p[2]  # z
        base_orientation = transforms3d.quaternions.quat2mat(base_orientation)

        torque = np.zeros(8)
        torque[0:4] = u_l
        torque[0] *= -1  # readjust to match motor polarity
        torque[4:8] = -u_r
        torque[7] *= -1  # readjust to match motor polarity
        # print(torque)
        # print(self.reaction_torques()[0:4])
        p.setJointMotorControlArray(bot, jointArray, p.TORQUE_CONTROL, forces=torque)

        # omega = p.getBaseVelocity(bot)[1]  # base angular velocity in global coordinates

        # Pull values in from simulator, select relevant ones, reshape to 2D array
        q = np.reshape([j[0] for j in p.getJointStates(1, range(0, 8))], (-1, 1))

        if useRealTime == 0:
            p.stepSimulation()

        return q, base_orientation
    '''
    def get_states(self):
        self.q = [j[0] for j in p.getJointStates(bot, range(8))]
        self.dq = [j[1] for j in p.getJointStates(bot, range(8))]
        state = append(state_q, state_dq)
        return state
    '''