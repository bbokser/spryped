import pybullet as p
import numpy as np

class spryped:

   def __init__(self, urdfRootPath=''):
      self.urdfRootPath = urdfRootPath
      self.reset()

   def buildJointNameToIdDict(self):
      nJoints = p.getNumJoints(self.spryped)
      self.jointNameToId = {}
      for i in range(nJoints):
         jointInfo = p.getJointInfo(self.spryped, i)
         self.jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
      self.resetPose()
      for i in range(100):
         p.stepSimulation()

   def buildMotorIdList(self):
      self.motorIdList.append(self.jointNameToId['motor_left_femur'])
      self.motorIdList.append(self.jointNameToId['motor_left_tibiotarsus'])
      self.motorIdList.append(self.jointNameToId['motor_left_tarsometatarsus'])
      self.motorIdList.append(self.jointNameToId['motor_left_toe'])
      self.motorIdList.append(self.jointNameToId['motor_right_femur'])
      self.motorIdList.append(self.jointNameToId['motor_right_tibiotarsus'])
      self.motorIdList.append(self.jointNameToId['motor_right_tarsometatarsus'])
      self.motorIdList.append(self.jointNameToId['motor_right_toe'])

   def reset(self):
      self.spryped = p.loadURDF("%s/spryped rev03/urdf/spryped rev03.urdf" % self.urdfRootPath, [0,0,0.8], [0,0,0])
      self.kp = 1
      self.kd = 0.1
      self.maxForce = 35
      self.nMotors = 8
      self.motorIdList = []
      self.motorDir = [-1, -1, -1, -1, 1, 1, 1, 1]
      self.buildJointNameToIdDict()
      self.buildMotorIdList()

   def setMotorAngleById(self, motorId, desiredAngle):
      p.setJointMotorControl2(bodyIndex=self.spryped,
                            jointIndex=motorId,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=desiredAngle,
                            positionGain=self.kp,
                            velocityGain=self.kd,
                            force=self.maxForce)

   def setMotorAngleByName(self, motorName, desiredAngle):
      self.setMotorAngleById(self.jointNameToId[motorName], desiredAngle)

   def resetPose(self):
      kneeFrictionForce = 0
      kneeangle = 0

      self.setMotorAngleByName('motor_left_femur', self.motorDir[0] * halfpi)
      self.setMotorAngleByName('motor_right_femur', self.motorDir[1] * halfpi)
      p.setJointMotorControl2(bodyIndex=self.quadruped,
                            jointIndex=self.jointNameToId['left femur'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=kneeFrictionForce)
      p.setJointMotorControl2(bodyIndex=self.quadruped,
                            jointIndex=self.jointNameToId['right femur'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=kneeFrictionForce)
   def getBasePosition(self):
      position, orientation = p.getBasePositionAndOrientation(self.quadruped)
      return position

   def getBaseOrientation(self):
      position, orientation = p.getBasePositionAndOrientation(self.quadruped)
      return orientation

   def applyAction(self, motorCommands):
      motorCommandsWithDir = np.multiply(motorCommands, self.motorDir)
      for i in range(self.nMotors):
         self.setMotorAngleById(self.motorIdList[i], motorCommandsWithDir[i])

   def getMotorAngles(self):
      motorAngles = []
      for i in range(self.nMotors):
         jointState = p.getJointState(self.spryped, self.motorIdList[i])
         motorAngles.append(jointState[0])
      motorAngles = np.multiply(motorAngles, self.motorDir)
      return motorAngles

   def getMotorVelocities(self):
      motorVelocities = []
      for i in range(self.nMotors):
         jointState = p.getJointState(self.spryped, self.motorIdList[i])
         motorVelocities.append(jointState[1])
      motorVelocities = np.multiply(motorVelocities, self.motorDir)
      return motorVelocities

   def getMotorTorques(self):
      motorTorques = []
      for i in range(self.nMotors):
         jointState = p.getJointState(self.spryped, self.motorIdList[i])
         motorTorques.append(jointState[3])
      motorTorques = np.multiply(motorTorques, self.motorDir)
      return motorTorques

