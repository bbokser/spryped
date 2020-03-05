'''
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
'''
import numpy as np
import math
import csv

import sim

from RobotBase import RobotBase

values = []
with open('spryped_urdf_rev05/urdf/spryped_urdf_rev05.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',') 
    next(data) # skip headers
    values = list(zip(*(row for row in data))) # transpose rows to columns
    values = np.array(values) # convert list of nested lists to array

comx = values[1].astype(np.float)
comy = values[2].astype(np.float)
comz = values[3].astype(np.float)

mass = values[7].astype(np.float)
ixx = values[8].astype(np.float)
ixy = values[9].astype(np.float)
ixz = values[10].astype(np.float)
iyy = values[11].astype(np.float)
iyz = values[12].astype(np.float)
izz = values[13].astype(np.float)

ox = values[38].astype(np.float)
oy = values[39].astype(np.float)
oz = values[40].astype(np.float)

coml = []
for j in range(9):
    comlen = math.sqrt((comx[j]**2)+(comz[j]**2)) # joint origin to COM ignoring y axis
    #comlen = math.sqrt((comx[j]**2)+(comy[j]**2)+(comz[j]**2)) # joint origin to COM length
    coml.append(comlen)
#print(coml[1])

# estimating init link angles
#p = 4
#dist = math.sqrt((ox[p]**2)+(oz[p]**2))
#angle = np.degrees(math.atan(oz[p]/dist))
#print("dist = ", dist)
#print("angle p = ", angle)
               
# link masses
if mass[1]!=mass[5]:
    print("WARNING: femur L/R masses unequal, check CAD")
if mass[2]!=mass[6]:
    print("WARNING: tibiotarsus L/R masses unequal, check CAD")
if mass[3]!=mass[7]:
    print("WARNING: tarsometatarsus L/R masses unequal, check CAD")
if mass[4]!=mass[8]:
    print("WARNING: toe L/R masses unequal, check CAD")

# link lengths must be manually updated
L0 = 0.1 # body
L1 = 0.1 # femur left
L2 = 0.199 # tibiotarsus left
L3 = 0.5 # tarsometatarsus left
L4 = 0.061 # toe left
L = np.array([L0, L1, L2, L3, L4, L1, L2, L3, L4])

class Robot(RobotBase):
    # first value in q and dq refer to BODY position
    def __init__(self, init_q=[0, np.pi/4, -np.pi*40.3/180, np.pi*84.629/180,-np.pi*44.329/180,
                 np.pi/4, -np.pi*40.3/180, np.pi*84.629/180, -np.pi*44.329/180.],
                 init_dq=[0., 0., 0., 0., 0., 0., 0., 0., 0.], singularity_thresh=.00025, **kwargs):

        self.DOF = 9
        RobotBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        # mass matrices
        self.MM = []
        for i in range(9):
            M = np.zeros((6, 6))
            M[0:3, 0:3] = np.eye(3)*float(mass[i])
            M[3, 3] = ixx[i]
            M[3, 4] = ixy[i]
            M[3, 5] = ixz[i]
            M[4, 3] = ixy[i]
            M[4, 4] = iyy[i]
            M[4, 5] = iyz[i]
            M[5, 3] = ixz[i]
            M[5, 4] = iyz[i]
            M[5, 5] = izz[i]
            #self.MM.insert(i,M)
            self.MM.append(M)

        #self.state = np.zeros(7)
        # initialize sim
        # self.sim = pyRobot.pySim(dt=1e-5)
        # send reset command to sim
        # self.sim.reset(self.state)
        # reset state
        self.reset()
        self.update_state()

    def apply_torque(self, u, dt=None):
        # Takes in torque and timestep and updates robot accordingly

        if dt is None:
            dt = self.dt

        u = -1*np.array(u, dtype='float')

        for ii in range(int(np.ceil(dt/1e-5))):
            self.sim.step(self.state, u)
        self.update_state()
        
    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        q1 = q[1]
    
        JCOM1 = np.zeros((6, 4))
        JCOM1[1, 0] = -L[1]*np.sin(q1) - 2*coml[1]*np.sin(2*q1)
        JCOM1[2, 0] = L[1]*np.cos(q1) + 2*coml[1]*np.cos(2*q1)
        JCOM1[3, :] = 1

        return JCOM1
    
    def gen_jacCOM2(self, q=None):
        """Generates the Jacobian from the COM of the second
        link to the origin frame"""
        q = self.q if q is None else q
        q1 = q[1]
        q2 = q[2]
        
        JCOM2 = np.zeros((6, 4))
        JCOM2[0, 1] = L[2]*np.cos(q2)
        JCOM2[1, 0] = -(L[1]+L[2]*np.cos(q2)+coml[2])*np.sin(q1)
        JCOM2[1, 1] = -L[2]*np.sin(q2)*np.cos(q1)
        JCOM2[2, 0] = (L[1]+L[2]*np.cos(q2)+coml[2])*np.cos(q1)
        JCOM2[2, 1] = -L[2]*np.sin(q1)*np.sin(q2)
        JCOM2[5, :] = 1

        return JCOM2
    
    def gen_jacCOM3(self, q=None):
        """Generates the Jacobian from the COM of the third
        link to the origin frame"""
        q = self.q if q is None else q
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        JCOM3 = np.zeros((6, 4))
        JCOM3[0, 0] = (L[1]+coml[3]*np.cos(q3))*np.sin(q1)*np.sin(q2+q3)
        JCOM3[0, 1] = -L[1]*np.cos(q1)*np.cos(q2+q3)\
            + L[2]*np.cos(q2-q3) - coml[3]*np.sin(q3)*np.sin(q2+q3)\
            - coml[3]*np.cos(q1)*np.cos(q3)*np.cos(q2+q3)
        JCOM3[0, 2] = -L[1]*np.cos(q1)*np.cos(q2 + q3) - L[2]*np.cos(q2 - q3)\
            + L[3]*np.cos(q3) + coml[3]*np.sin(q3)*np.sin(q2 + q3)*np.cos(q1)\
            - coml[3]*np.sin(q3)*np.sin(q2 + q3)\
            - coml[3]*np.cos(q1)*np.cos(q3)*np.cos(q2 + q3)\
            + coml[3]*np.cos(q3)*np.cos(q2 + q3)
        JCOM3[1, 0] = -(L[1]+coml[3]*np.cos(q3))*np.sin(q1)*np.cos(q2+q3)
        JCOM3[1, 1] = -L[1]*np.sin(q2+q3)*np.cos(q1) - L[2]*np.sin(q2-q3)\
            + coml[3]*np.sin(q3)*np.cos(q2+q3)\
            - coml[3]*np.sin(q2+q3)*np.cos(q1)*np.cos(q3)
        JCOM3[1, 2] = -L[1]*np.sin(q2 + q3)*np.cos(q1) + L[2]*np.sin(q2 - q3)\
            - L[3]*np.sin(q3) - coml[3]*np.sin(q3)*np.cos(q1)*np.cos(q2 + q3)\
            + coml[3]*np.sin(q3)*np.cos(q2 + q3)\
            - coml[3]*np.sin(q2 + q3)*np.cos(q1)*np.cos(q3)\
            + coml[3]*np.sin(q2 + q3)*np.cos(q3)
        JCOM3[2, 0] = (L[1]+coml[3]*np.cos(q3))*np.cos(q1)
        JCOM3[2, 2] = -coml[3]*np.sin(q1)*np.sin(q3)
        JCOM3[5, :] = 1

        return JCOM3
    
    def gen_jacCOM4(self, q=None):
        """Generates the Jacobian from the COM of the fourth
        link to the origin frame"""
        q = self.q if q is None else q
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        JCOM4 = np.zeros((6, 4))
        JCOM4[0, 0] = (L[1] + coml[4]*np.cos(q4))*np.sin(q1)*np.sin(q2 + q3 + q4)
        JCOM4[0, 1] = -L[1]*np.cos(q1)*np.cos(q2 + q3 + q4)\
            + L[2]*np.cos(-q2 + q3 + q4) - coml[4]*np.sin(q4)*np.sin(q2 + q3 + q4)\
            - coml[4]*np.cos(q1)*np.cos(q4)*np.cos(q2 + q3 + q4)
        JCOM4[0, 2] = -L[1]*np.cos(q1)*np.cos(q2 + q3 + q4)\
            - L[2]*np.cos(-q2 + q3 + q4) + L[3]*np.cos(q3 - q4)\
            - coml[4]*np.sin(q4)*np.sin(q2 + q3 + q4)\
            - coml[4]*np.cos(q1)*np.cos(q4)*np.cos(q2 + q3 + q4)
        JCOM4[0, 3] = -L[1]*np.cos(q1)*np.cos(q2 + q3 + q4)\
            - L[2]*np.cos(-q2 + q3 + q4) - L[3]*np.cos(q3 - q4) + L[4]*np.cos(q4)\
            + coml[4]*np.sin(q4)*np.sin(q2 + q3 + q4)*np.cos(q1)\
            - coml[4]*np.sin(q4)*np.sin(q2 + q3 + q4)\
            - coml[4]*np.cos(q1)*np.cos(q4)*np.cos(q2 + q3 + q4)\
            + coml[4]*np.cos(q4)*np.cos(q2 + q3 + q4)
        JCOM4[1, 0] = -(L[1] + coml[4]*np.cos(q4))*np.sin(q1)*np.cos(q2 + q3 + q4)
        JCOM4[1, 1] = -L[1]*np.sin(q2 + q3 + q4)*np.cos(q1)\
            + L[2]*np.sin(-q2 + q3 + q4) + coml[4]*np.sin(q4)*np.cos(q2 + q3 + q4)\
            - coml[4]*np.sin(q2 + q3 + q4)*np.cos(q1)*np.cos(q4)
        JCOM4[1, 2] = -L[1]*np.sin(q2 + q3 + q4)*np.cos(q1)\
            - L[2]*np.sin(-q2 + q3 + q4) - L[3]*np.sin(q3 - q4)\
            + coml[4]*np.sin(q4)*np.cos(q2 + q3 + q4)\
            - coml[4]*np.sin(q2 + q3 + q4)*np.cos(q1)*np.cos(q4)
        JCOM4[1, 3] = -L[1]*np.sin(q2 + q3 + q4)*np.cos(q1)\
            - L[2]*np.sin(-q2 + q3 + q4) + L[3]*np.sin(q3 - q4) - L[4]*np.sin(q4)\
            - coml[4]*np.sin(q4)*np.cos(q1)*np.cos(q2 + q3 + q4)\
            + coml[4]*np.sin(q4)*np.cos(q2 + q3 + q4)\
            - coml[4]*np.sin(q2 + q3 + q4)*np.cos(q1)*np.cos(q4)\
            + coml[4]*np.sin(q2 + q3 + q4)*np.cos(q4)
        JCOM4[2, 0] = (L[1] + coml[4]*np.cos(q4))*np.cos(q1)
        JCOM4[2, 3] = -coml[4]*np.sin(q1)*np.sin(q4)
        JCOM4[5, :] = 1
        
        return JCOM4

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from the end effector to the origin frame"""
        q = self.q if q is None else q
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        JEE = np.zeros((3, 4)) # Only x, y, z forces controlled, others dropped
        JEE[0, 0] = L[4]*np.sin(q1)*np.sin(q2 + q3 + q4)*np.cos(q4)
        JEE[0, 1] = -L[4]*(np.sin(q4)*np.sin(q2 + q3 + q4)\
            + np.cos(q1)*np.cos(q4)*np.cos(q2 + q3 + q4))
        JEE[0, 2] = JEE[0, 1]
        JEE[0, 3] = L[4]*(np.cos(q2 + q3 + 2*q4)\
            - np.cos(-q1 + q2 + q3 + 2*q4)/2 - np.cos(q1 + q2 + q3 + 2*q4)/2)
        JEE[1, 0] = -L[4]*np.sin(q1)*np.cos(q4)*np.cos(q2 + q3 + q4)
        JEE[1, 1] = L[4]*(np.sin(q4)*np.cos(q2 + q3 + q4)\
            - np.sin(q2 + q3 + q4)*np.cos(q1)*np.cos(q4))
        JEE[1, 2] = JEE[1, 1]
        JEE[1, 3] = L[4]*(np.sin(q2 + q3 + 2*q4)\
            - np.sin(-q1 + q2 + q3 + 2*q4)/2 - np.sin(q1 + q2 + q3 + 2*q4)/2)
        JEE[2, 0] = L[4]*np.cos(q1)*np.cos(q4)
        JEE[2, 3] = -L[4]*np.sin(q1)*np.sin(q4)

        return JEE
    
    def gen_Mq(self, q=None):
        # Mass matrix
        M1 = self.MM[1]
        M2 = self.MM[2]
        M3 = self.MM[3]
        
        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        JCOM3 = self.gen_jacCOM3(q=q)
        JCOM4 = self.gen_jacCOM3(q=q)
        
        Mq = (np.dot(JCOM1.T, np.dot(M1, JCOM1)) +
              np.dot(JCOM2.T, np.dot(M2, JCOM2)) +
              np.dot(JCOM3.T, np.dot(M3, JCOM3)) +
              np.dot(JCOM4.T, np.dot(M4, JCOM4)))

        return Mq

    def reset(self, q=[], dq=[]):
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if q:
            assert len(q) == self.DOF
        if dq:
            assert len(dq) == self.DOF

        state = np.zeros(self.DOF*2)
        # slice w/ step size of 2 to interweave q and dq into state
        state[::2] = self.init_q if not q else np.copy(q)
        state[1::2] =  self.init_dq if not dq else np.copy(dq)
        
        self.update_state() # is this necessary? Seems redundant

    def update_state(self):
        # Update the local variables
        state = sim.get_state()
        self.t = self.state[0]
        self.q = self.state[1:4]
        self.dq = self.state[4:]
        
#robert = Robot()
#print(robert.MM[7])
#print(robert.gen_Mx())
