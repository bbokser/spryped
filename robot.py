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

import numpy as np
import math
import csv

import pybullet as p

from RobotBase import RobotBase


class Robot(RobotBase):
    # first value in q and dq refer to BODY position
    def __init__(self, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            # init_dq = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
            init_dq = [0., 0., 0., 0.]  # just left leg
        if init_q is None:
            # init_q = [0, -np.pi / 4, np.pi * 40.3 / 180, -np.pi * 84.629 / 180, np.pi * 44.329 / 180,
            #           -np.pi / 4, -np.pi * 40.3 / 180, np.pi * 84.629 / 180, -np.pi * 44.329 / 180.]
            init_q = [-2 * np.pi / 4, np.pi * 32 / 180, -np.pi * 44.17556088 / 180, np.pi * 12.17556088 / 180.]
        self.DOF = 4
        RobotBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        values = []
        with open('spryped_urdf_rev05/urdf/spryped_urdf_rev05.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            next(data)  # skip headers
            values = list(zip(*(row for row in data)))  # transpose rows to columns
            values = np.array(values)  # convert list of nested lists to array

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

        self.coml = []
        for j in range(9):
            comlen = math.sqrt((comx[j] ** 2) + (comz[j] ** 2))  # joint origin to COM ignoring y axis
            #  comlen = math.sqrt((comx[j]**2)+(comy[j]**2)+(comz[j]**2)) # joint origin to COM length
            self.coml.append(comlen)
        # print(self.coml[1])

        # estimating init link angles
        # p = 4
        # dist = math.sqrt((ox[p]**2)+(oz[p]**2))
        # angle = np.degrees(math.atan(oz[p]/dist))
        # print("dist = ", dist)
        # print("angle p = ", angle)

        # link masses
        if mass[1] != mass[5]:
            print("WARNING: femur L/R masses unequal, check CAD")
        if mass[2] != mass[6]:
            print("WARNING: tibiotarsus L/R masses unequal, check CAD")
        if mass[3] != mass[7]:
            print("WARNING: tarsometatarsus L/R masses unequal, check CAD")
        if mass[4] != mass[8]:
            print("WARNING: toe L/R masses unequal, check CAD")

        # link lengths (mm) must be manually updated
        L0 = 288  # body
        L1 = 114  # femur left
        L2 = 199  # tibiotarsus left
        L3 = 500  # tarsometatarsus left
        L4 = 61  # toe left
        self.L = np.array([L0, L1, L2, L3, L4, L1, L2, L3, L4])

        # mass matrices
        self.MM = []
        for i in range(9):
            M = np.zeros((6, 6))
            M[0:3, 0:3] = np.eye(3) * float(mass[i])
            M[3, 3] = ixx[i]
            M[3, 4] = ixy[i]
            M[3, 5] = ixz[i]
            M[4, 3] = -ixy[i]
            M[4, 4] = iyy[i]
            M[4, 5] = iyz[i]
            M[5, 3] = -ixz[i]
            M[5, 4] = -iyz[i]
            M[5, 5] = izz[i]
            # self.MM.insert(i,M)
            self.MM.append(M)

        # self.state = np.zeros(7)
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

        u = -1 * np.array(u, dtype='float')

        # for ii in range(int(np.ceil(dt / 1e-5))):
        #    self.sim.step(self.state, u)
        self.update_state()
        return u

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        # q1 = q[1]
        q1 = q[0]

        L1 = self.L[1]
        coml1 = self.coml[1]

        JCOM1 = np.zeros((6, 4))
        JCOM1[1, 0] = -L1 * np.sin(q1) - 2 * coml1 * np.sin(2 * q1)
        JCOM1[2, 0] = L1 * np.cos(q1) + 2 * coml1 * np.cos(2 * q1)
        JCOM1[3, :] = 1

        return JCOM1

    def gen_jacCOM2(self, q=None):
        """Generates the Jacobian from the COM of the second
        link to the origin frame"""
        q = self.q if q is None else q
        '''
        q1 = q[1]
        q2 = q[2]
        '''
        q1 = q[0]
        q2 = q[1]

        L1 = self.L[1]
        L2 = self.L[2]
        coml2 = self.coml[2]

        JCOM2 = np.zeros((6, 4))
        JCOM2[0, 1] = L2 * np.cos(q2)
        JCOM2[1, 0] = -(L1 + L2 * np.cos(q2) + coml2) * np.sin(q1)
        JCOM2[1, 1] = -L2 * np.sin(q2) * np.cos(q1)
        JCOM2[2, 0] = (L1 + L2 * np.cos(q2) + coml2) * np.cos(q1)
        JCOM2[2, 1] = -L2 * np.sin(q1) * np.sin(q2)
        JCOM2[5, :] = 1

        return JCOM2

    def gen_jacCOM3(self, q=None):
        """Generates the Jacobian from the COM of the third
        link to the origin frame"""
        q = self.q if q is None else q
        '''
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        '''
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        coml3 = self.coml[3]

        JCOM3 = np.zeros((6, 4))
        JCOM3[0, 0] = (L1 + coml3 * np.cos(q3)) * np.sin(q1) * np.sin(q2 + q3)
        JCOM3[0, 1] = -L1 * np.cos(q1) * np.cos(q2 + q3) \
            + L2 * np.cos(q2 - q3) - coml3 * np.sin(q3) * np.sin(q2 + q3) \
            - coml3 * np.cos(q1) * np.cos(q3) * np.cos(q2 + q3)
        JCOM3[0, 2] = -L1 * np.cos(q1) * np.cos(q2 + q3) - L2 * np.cos(q2 - q3) \
            + L3 * np.cos(q3) + coml3 * np.sin(q3) * np.sin(q2 + q3) * np.cos(q1) \
            - coml3 * np.sin(q3) * np.sin(q2 + q3) \
            - coml3 * np.cos(q1) * np.cos(q3) * np.cos(q2 + q3) \
            + coml3 * np.cos(q3) * np.cos(q2 + q3)
        JCOM3[1, 0] = -(L1 + coml3 * np.cos(q3)) * np.sin(q1) * np.cos(q2 + q3)
        JCOM3[1, 1] = -L1 * np.sin(q2 + q3) * np.cos(q1) - L2 * np.sin(q2 - q3) \
            + coml3 * np.sin(q3) * np.cos(q2 + q3) \
            - coml3 * np.sin(q2 + q3) * np.cos(q1) * np.cos(q3)
        JCOM3[1, 2] = -L1 * np.sin(q2 + q3) * np.cos(q1) + L2 * np.sin(q2 - q3) \
            - L3 * np.sin(q3) - coml3 * np.sin(q3) * np.cos(q1) * np.cos(q2 + q3) \
            + coml3 * np.sin(q3) * np.cos(q2 + q3) \
            - coml3 * np.sin(q2 + q3) * np.cos(q1) * np.cos(q3) \
            + coml3 * np.sin(q2 + q3) * np.cos(q3)
        JCOM3[2, 0] = (L1 + coml3 * np.cos(q3)) * np.cos(q1)
        JCOM3[2, 2] = -coml3 * np.sin(q1) * np.sin(q3)
        JCOM3[5, :] = 1

        return JCOM3

    def gen_jacCOM4(self, q=None):
        """Generates the Jacobian from the COM of the fourth
        link to the origin frame"""
        q = self.q if q is None else q
        '''
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        '''
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        L4 = self.L[4]
        coml4 = self.coml[4]

        JCOM4 = np.zeros((6, 4))
        JCOM4[0, 0] = (L1 + coml4 * np.cos(q4)) * np.sin(q1) * np.sin(q2 + q3 + q4)
        JCOM4[0, 1] = -L1 * np.cos(q1) * np.cos(q2 + q3 + q4) \
            + L2 * np.cos(-q2 + q3 + q4) - coml4 * np.sin(q4) * np.sin(q2 + q3 + q4) \
            - coml4 * np.cos(q1) * np.cos(q4) * np.cos(q2 + q3 + q4)
        JCOM4[0, 2] = -L1 * np.cos(q1) * np.cos(q2 + q3 + q4) \
            - L2 * np.cos(-q2 + q3 + q4) + L3 * np.cos(q3 - q4) \
            - coml4 * np.sin(q4) * np.sin(q2 + q3 + q4) \
            - coml4 * np.cos(q1) * np.cos(q4) * np.cos(q2 + q3 + q4)
        JCOM4[0, 3] = -L1 * np.cos(q1) * np.cos(q2 + q3 + q4) \
            - L2 * np.cos(-q2 + q3 + q4) - L3 * np.cos(q3 - q4) + L4 * np.cos(q4) \
            + coml4 * np.sin(q4) * np.sin(q2 + q3 + q4) * np.cos(q1) \
            - coml4 * np.sin(q4) * np.sin(q2 + q3 + q4) \
            - coml4 * np.cos(q1) * np.cos(q4) * np.cos(q2 + q3 + q4) \
            + coml4 * np.cos(q4) * np.cos(q2 + q3 + q4)
        JCOM4[1, 0] = -(L1 + coml4 * np.cos(q4)) * np.sin(q1) * np.cos(q2 + q3 + q4)
        JCOM4[1, 1] = -L1 * np.sin(q2 + q3 + q4) * np.cos(q1) \
            + L2 * np.sin(-q2 + q3 + q4) + coml4 * np.sin(q4) * np.cos(q2 + q3 + q4) \
            - coml4 * np.sin(q2 + q3 + q4) * np.cos(q1) * np.cos(q4)
        JCOM4[1, 2] = -L1 * np.sin(q2 + q3 + q4) * np.cos(q1) \
            - L2 * np.sin(-q2 + q3 + q4) - L3 * np.sin(q3 - q4) \
            + coml4 * np.sin(q4) * np.cos(q2 + q3 + q4) \
            - coml4 * np.sin(q2 + q3 + q4) * np.cos(q1) * np.cos(q4)
        JCOM4[1, 3] = -L1 * np.sin(q2 + q3 + q4) * np.cos(q1) \
            - L2 * np.sin(-q2 + q3 + q4) + L3 * np.sin(q3 - q4) - L4 * np.sin(q4) \
            - coml4 * np.sin(q4) * np.cos(q1) * np.cos(q2 + q3 + q4) \
            + coml4 * np.sin(q4) * np.cos(q2 + q3 + q4) \
            - coml4 * np.sin(q2 + q3 + q4) * np.cos(q1) * np.cos(q4) \
            + coml4 * np.sin(q2 + q3 + q4) * np.cos(q4)
        JCOM4[2, 0] = (L1 + coml4 * np.cos(q4)) * np.cos(q1)
        JCOM4[2, 3] = -coml4 * np.sin(q1) * np.sin(q4)
        JCOM4[5, :] = 1

        return JCOM4

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from the end effector to the origin frame"""
        q = self.q if q is None else q
        '''
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        '''
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        L4 = self.L[4]

        JEE = np.zeros((3, 4))  # Only x, y, z forces controlled, others dropped
        JEE[0, 0] = L4 * np.sin(q1) * np.sin(q2 + q3 + q4) * np.cos(q4)
        JEE[0, 1] = -L4 * (np.sin(q4) * np.sin(q2 + q3 + q4)
                           + np.cos(q1) * np.cos(q4) * np.cos(q2 + q3 + q4))
        JEE[0, 2] = JEE[0, 1]
        JEE[0, 3] = L4 * (np.cos(q2 + q3 + 2 * q4)
                          - np.cos(-q1 + q2 + q3 + 2 * q4) / 2 - np.cos(q1 + q2 + q3 + 2 * q4) / 2)
        JEE[1, 0] = -L4 * np.sin(q1) * np.cos(q4) * np.cos(q2 + q3 + q4)
        JEE[1, 1] = L4 * (np.sin(q4) * np.cos(q2 + q3 + q4)
                          - np.sin(q2 + q3 + q4) * np.cos(q1) * np.cos(q4))
        JEE[1, 2] = JEE[1, 1]
        JEE[1, 3] = L4 * (np.sin(q2 + q3 + 2 * q4)
                          - np.sin(-q1 + q2 + q3 + 2 * q4) / 2 - np.sin(q1 + q2 + q3 + 2 * q4) / 2)
        JEE[2, 0] = L4 * np.cos(q1) * np.cos(q4)
        JEE[2, 3] = -L4 * np.sin(q1) * np.sin(q4)

        return JEE

    def gen_Mq(self, q=None):
        # Mass matrix
        M1 = self.MM[1]
        M2 = self.MM[2]
        M3 = self.MM[3]
        M4 = self.MM[4]

        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        JCOM3 = self.gen_jacCOM3(q=q)
        JCOM4 = self.gen_jacCOM3(q=q)

        Mq = (np.dot(JCOM1.T, np.dot(M1, JCOM1)) +
              np.dot(JCOM2.T, np.dot(M2, JCOM2)) +
              np.dot(JCOM3.T, np.dot(M3, JCOM3)) +
              np.dot(JCOM4.T, np.dot(M4, JCOM4)))

        return Mq

    def position(self, q=None):
        """forward kinematics
        Compute x,y,z position of end effector relative to base

        q np.array: a set of angles to return positions for
        """
        if q is None:
            q1 = self.q[0]
            q2 = self.q[1]
            q3 = self.q[2]
            q4 = self.q[3]
        else:
            q1 = q[0]
            q2 = q[1]
            q3 = q[2]
            q4 = q[3]

        # L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        L4 = self.L[4]

        x = -np.cumsum([  # 0,  # body link
            0,  # femur
            L2 * np.sin(q2),
            L3 * np.sin(q2 + q3),
            L4 * np.sin(q2 + q3 + q4)])

        y = np.cumsum([  # L0 * np.cos(q0),
            L1 * np.cos(q1),
            L2 * np.cos(q2) * np.cos(q1),
            L3 * np.cos(q2 + q3) * np.cos(q1),
            L4 * np.cos(q2 + q3 + q4) * np.cos(q1)])

        z = np.cumsum([  # L0 * np.sin(q0),
            L1 * np.sin(q1),
            L2 * np.cos(q2) * np.sin(q1),
            L3 * np.cos(q2 + q3) * np.sin(q1),
            L4 * np.cos(q2 + q3 + q4) * np.sin(q1)])

        return np.array([x, y, z], dtype=float)

    def velocity(self, dq=None):
        # Calculate operational space linear velocity vector
        if dq is None:
            dq = self.dq
        JEE = self.gen_jacEE()
        return np.dot(JEE, self.dq)

    def reset(self, q=None, dq=None):
        if q is None:
            q = []
        if dq is None:
            dq = []
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if q:
            assert len(q) == self.DOF
        if dq:
            assert len(dq) == self.DOF

        # state = np.zeros(self.DOF * 2)
        # slice w/ step size of 2 to interweave q and dq into state
        # state[::2] = self.init_q if not q else np.copy(q)
        # state[1::2] = self.init_dq if not dq else np.copy(dq)

        self.update_state()  # is this necessary? Seems redundant

    def update_state(self):
        # Update the local variables
        # Pull values in from PyBullet, select relevant ones, reshape to 2D array
        self.q = np.reshape([j[0] for j in p.getJointStates(1, range(0, 4))], (-1, 1))  # only using left leg values rn
        # self.q = np.dot(np.squeeze(self.q), np.transpose([1., -1., -1., 1.]))  # adjust polarity of motors to match math
        self.q[1] *= -1
        self.q[2] *= -1
        self.q[3] *= -1
        # Calibrate encoders
        self.q = np.add(self.q.flatten(), np.array([-2 * np.pi / 4, np.pi * 32 / 180,
                                                   -np.pi * 44.17556088 / 180, np.pi * 12.17556088 / 180.]))
        self.dq = np.reshape([j[1] for j in p.getJointStates(1, range(0, 4))], (-1, 1))

# robert = Robot()
# print(robert.MM[7])
# print(robert.gen_Mx())
