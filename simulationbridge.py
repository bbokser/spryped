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

import numpy as np
import transforms3d
import pybullet as p
import pybullet_data


GRAVITY = -9.807
# physicsClient = p.connect(p.GUI)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
plane = p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

bot = p.loadURDF("spryped_urdf_rev06/urdf/spryped_urdf_rev06.urdf", [0, 0, 0.8],
                 robotStartOrientation, useFixedBase=0,
                 flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

p.setGravity(0, 0, GRAVITY)

# p.changeDynamics(bot, 3, lateralFriction=0.5)
# p.changeDynamics(bot, 7, lateralFriction=0.5)

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
        self.omega_xyz = None
        self.omega = None
        self.v = None
        self.record_rt = False  # record video in real time
        # print(p.getJointInfo(bot, 3))

        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        p.setTimeStep(self.dt)

        # p.setRealTimeSimulation(useRealTime)
        p.setRealTimeSimulation(0)

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(bot)):
            p.setJointMotorControl2(bot, i, p.VELOCITY_CONTROL, force=0.5)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(bot, i, 1)  # enable joint torque sensing

    def sim_run(self, u_l, u_r):

        base_or_p = np.array(p.getBasePositionAndOrientation(bot)[1])
        # pybullet gives quaternions in xyzw format
        # transforms3d takes quaternions in wxyz format, so you need to shift values
        b_orient = np.zeros(4)
        b_orient[0] = base_or_p[3]  # w
        b_orient[1] = base_or_p[0]  # x
        b_orient[2] = base_or_p[1]  # y
        b_orient[3] = base_or_p[2]  # z
        b_orient = transforms3d.quaternions.quat2mat(b_orient)

        torque = np.zeros(8)
        torque[0:4] = u_l
        torque[0] *= -1  # readjust to match motor polarity
        torque[4:8] = -u_r
        torque[7] *= -1  # readjust to match motor polarity

        # print(self.reaction_torques()[0:4])
        p.setJointMotorControlArray(bot, jointArray, p.TORQUE_CONTROL, forces=torque)
        velocities = p.getBaseVelocity(bot)
        self.v = velocities[0]  # base linear velocity in global Cartesian coordinates
        self.omega_xyz = velocities[1]  # base angular velocity in Euler XYZ
        # print(omega_xyz[2], omega_xyz[1], omega_xyz[0])
        # base angular velocity in quaternions
        # self.omega = transforms3d.euler.euler2quat(omega_xyz[0], omega_xyz[1], omega_xyz[2], axes='rxyz')
        # found to be intrinsic Euler angles (r)

        # Pull values in from simulator, select relevant ones, reshape to 2D array
        q = np.reshape([j[0] for j in p.getJointStates(1, range(0, 8))], (-1, 1))

        # Detect contact of feet with ground plane
        c1 = bool(len([c[8] for c in p.getContactPoints(bot, plane, 3)]))
        c2 = bool(len([c[8] for c in p.getContactPoints(bot, plane, 7)]))

        # dq = [j[1] for j in p.getJointStates(bot, range(8))]

        if useRealTime == 0:
            p.stepSimulation()

        return q, b_orient, c1, c2
