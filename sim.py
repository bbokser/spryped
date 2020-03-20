'''
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
'''

import time

import numpy as np

import pybullet as p
import pybullet_data

#from osc import Control
#from robot import Robot

GRAVITY = -9.81
  
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
planeID = p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
bot = p.loadURDF("spryped_urdf_rev05/urdf/spryped_urdf_rev05.urdf", [0,0,0.8],
                         robotStartOrientation, useFixedBase=1)
p.setGravity(0,0, GRAVITY)
        
class Runner:

    def __init__(self, dt=1e-3):
        self.dt = dt
        self.tau = None

    def run(self, robot, control_shell):

        self.robot = robot

        self.shell = control_shell

        p.setTimeStep(self.dt)

        #p.setRealTimeSimulation(useRealTime)

        useRealTime = 0

        torque = 4
        u = [torque, torque, torque, torque, torque, torque, torque, torque]

        jointArray = range(p.getNumJoints(bot))

        # Disable the default velocity/position motor:
        for i in range(p.getNumJoints(bot)):
            p.setJointMotorControl2(bot, i, p.VELOCITY_CONTROL, force=1)
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(bot,i,1) # enable joint torque sensing

        steps = 0

        while 1:
            time.sleep(self.dt)
            # update target after specified period of time passes
            steps = steps + 1

            if steps == 1:
                self.target = self.shell.controller.gen_target(self.robot)
            else:
                self.target = self.shell.controller.target

            self.tau = self.shell.control(self.robot)
            self.robot.apply_torque(u=self.tau, dt=self.dt)
            
            p.setJointMotorControlArray(bot, jointArray, p.TORQUE_CONTROL, forces=u)
            #states=[j[2] for j in p.getJointStates(bot,range(7))] # j[2] selects jointReactionForces
            #print(states[1][4]) #'%s' % float('%.1g' % pront[1])) # selected joint 2, My
            if (useRealTime == 0):
                p.stepSimulation()

        def reaction(self):
            reaction_force=[j[2] for j in p.getJointStates(bot,range(7))] # j[2]=jointReactionForces
            #  [Fx, Fy, Fz, Mx, My, Mz]
            return reaction_force[:][4] # selected joint 1 and 5 Mz, all other joints My

        def get_states(self):
            self.q = [j[0] for j in p.getJointStates(bot,range(7))]
            self.dq = [j[1] for j in p.getJointStates(bot,range(7))]
            state = append(state_q, state_dq)
            return state








