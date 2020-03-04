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

from osc import Control
from robot import Robot

GRAVITY = -9.81
dt = 1e-3

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
planeID = p.loadURDF("plane.urdf")
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("spryped_urdf_rev05/urdf/spryped_urdf_rev05.urdf", [0,0,0.8],
                   robotStartOrientation, useFixedBase=1)

p.setGravity(0,0, GRAVITY)
p.setTimeStep(dt)

useRealTime = 0

#p.setRealTimeSimulation(useRealTime)
torque = 4;
u = [torque, torque, torque, torque, torque, torque, torque, torque]

jointArray = range(p.getNumJoints(robot))

# Disable the default velocity/position motor:
for i in range(p.getNumJoints(robot)):
  p.setJointMotorControl2(robot, i, p.VELOCITY_CONTROL, force=1)
  # force=1 allows us to easily mimic joint friction rather than disabling
  p.enableJointForceTorqueSensor(robot,i,1) # enable joint torque sensing

robot = Robot()
c = Control()
print(c.control())
'''
while(1):
    time.sleep(dt)
    p.setJointMotorControlArray(robot, jointArray, p.TORQUE_CONTROL, forces=u)
    
    states=[j[2] for j in p.getJointStates(robot,range(7))] # j[2] selects jointReactionForces
    #  [Fx, Fy, Fz, Mx, My, Mz]
    print(states[1][4]) #'%s' % float('%.1g' % pront[1])) # selected joint 2, My
    
    if (useRealTime == 0):
        p.stepSimulation()
        
time.sleep(10000)

'''






