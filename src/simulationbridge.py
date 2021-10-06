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

import os
import numpy as np
import transforms3d
import pybullet as p
import pybullet_data

import actuator


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

    def __init__(self, dt=1e-3, fixed=False, record=False):
        self.dt = dt
        self.omega_xyz = None
        self.omega = None
        self.v = None
        self.record_rt = record  # record video in real time

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        model_path = "res/spryped_urdf_rev06/urdf/spryped_urdf_rev06.urdf"
        self.bot = p.loadURDF(os.path.join(path_parent, model_path), [0, 0, 0.8],  # 0.31
                              robotStartOrientation, useFixedBase=fixed,
                              flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)
        self.numJoints = p.getNumJoints(self.bot)
        p.setGravity(0, 0, -9.807)

        # p.changeDynamics(self.bot, 3, lateralFriction=0.5)
        self.jointArray = range(p.getNumJoints(self.bot))

        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        p.setTimeStep(self.dt)
        self.useRealTime = 0
        p.setRealTimeSimulation(self.useRealTime)

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(self.bot)):
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0.5)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(self.bot, i, 1)  # enable joint torque sensing

    def sim_run(self, u_l, u_r):

        base_or_p = np.array(p.getBasePositionAndOrientation(self.bot)[1])
        # pybullet gives quaternions in xyzw format
        # transforms3d takes quaternions in wxyz format, so you need to shift values
        b_orient = np.zeros(4)
        b_orient[0] = base_or_p[3]  # w
        b_orient[1] = base_or_p[0]  # x
        b_orient[2] = base_or_p[1]  # y
        b_orient[3] = base_or_p[2]  # z
        b_orient = transforms3d.quaternions.quat2mat(b_orient)

        # Pull values in from simulator, select relevant ones, reshape to 2D array
        q = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
        q_dot = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))

        command = np.zeros(8)
        command[0:4] = u_l
        command[0] *= -1  # readjust to match motor polarity
        command[4:8] = -u_r
        command[7] *= -1  # readjust to match motor polarity

        torque = np.zeros(8)
        torque[0] = actuator.actuate(i=command[0], q_dot=q_dot[0])
        torque[1] = actuator.actuate(i=command[1], q_dot=q_dot[1])
        torque[2] = actuator.actuate(i=command[2], q_dot=q_dot[2])
        torque[3] = actuator.actuate(i=command[3], q_dot=q_dot[3])
        torque[4] = actuator.actuate(i=command[4], q_dot=q_dot[4])
        torque[5] = actuator.actuate(i=command[5], q_dot=q_dot[5])
        torque[6] = actuator.actuate(i=command[6], q_dot=q_dot[6])
        torque[7] = actuator.actuate(i=command[7], q_dot=q_dot[7])

        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)
        velocities = p.getBaseVelocity(self.bot)
        self.v = velocities[0]  # base linear velocity in global Cartesian coordinates
        self.omega_xyz = velocities[1]  # base angular velocity in Euler XYZ
        # base angular velocity in quaternions
        # self.omega = transforms3d.euler.euler2quat(omega_xyz[0], omega_xyz[1], omega_xyz[2], axes='rxyz')
        # found to be intrinsic Euler angles (r)

        # Detect contact of feet with ground plane
        c1 = bool(len([c[8] for c in p.getContactPoints(self.bot, self.plane, 3)]))
        c2 = bool(len([c[8] for c in p.getContactPoints(self.bot, self.plane, 7)]))

        if self.useRealTime == 0:
            p.stepSimulation()

        return q, b_orient, c1, c2
