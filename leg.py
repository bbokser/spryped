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

import csv

import transforms3d

from legbase import LegBase


class Leg(LegBase):

    def __init__(self, leg, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0., 0., 0.]  # just left leg

        if init_q is None:
            init_q = [-np.pi / 2, np.pi * 32 / 180, -np.pi * 44.17556088 / 180, np.pi * 12.17556088 / 180.]

        self.DOF = 4

        LegBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        # values = []

        if leg == 1:
            spryped_data = str('spryped_urdf_rev06/spryped_data_left.csv')
        else:
            spryped_data = str('spryped_urdf_rev06/spryped_data_right.csv')

        with open(spryped_data, 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            next(data)  # skip headers
            values = list(zip(*(row for row in data)))  # transpose rows to columns
            values = np.array(values)  # convert list of nested lists to array

        ixx = values[1].astype(np.float)
        ixy = values[2].astype(np.float)
        ixz = values[3].astype(np.float)
        iyy = values[4].astype(np.float)
        iyz = values[5].astype(np.float)
        izz = values[6].astype(np.float)

        self.coml = values[7].astype(np.float)

        with open('spryped_urdf_rev06/urdf/spryped_urdf_rev06.csv', 'r') as csvfile:
            data_direct = csv.reader(csvfile, delimiter=',')
            next(data_direct)  # skip headers
            values_direct = list(zip(*(row for row in data_direct)))  # transpose rows to columns
            values_direct = np.array(values_direct)  # convert list of nested lists to array

        self.mass = values_direct[7].astype(np.float)
        self.mass = np.delete(self.mass, 0)  # remove body value

        # estimating init link angles
        # p = 4
        # dist = math.sqrt((ox[p]**2)+(oz[p]**2))
        # angle = np.degrees(math.atan(oz[p]/dist))
        # print("dist = ", dist)
        # print("angle p = ", angle)

        # link masses
        if self.mass[0] != self.mass[4]:
            print("WARNING: femur L/R masses unequal, check CAD")
        if self.mass[1] != self.mass[5]:
            print("WARNING: tibiotarsus L/R masses unequal, check CAD")
        if self.mass[2] != self.mass[6]:
            print("WARNING: tarsometatarsus L/R masses unequal, check CAD")
        if self.mass[3] != self.mass[7]:
            print("WARNING: toe L/R masses unequal, check CAD")

        # link lengths (mm) must be manually updated

        L0 = .114  # femur
        L1 = .199  # tibiotarsus
        L2 = .500  # tarsometatarsus
        L3 = .061  # toe
        self.L = np.array([L0, L1, L2, L3])

        # mass matrices and gravity
        self.MM = []
        self.Fg = []
        self.gravity = np.array([[0, 0, -9.807]]).T
        self.extra = np.array([[0, 0, 0]]).T

        for i in range(0, 4):
            M = np.zeros((6, 6))
            M[0:3, 0:3] = np.eye(3) * float(self.mass[i])
            M[3, 3] = ixx[i]
            M[3, 4] = ixy[i]
            M[3, 5] = ixz[i]
            M[4, 3] = ixy[i]
            M[4, 4] = iyy[i]
            M[4, 5] = iyz[i]
            M[5, 3] = ixz[i]
            M[5, 4] = iyz[i]
            M[5, 5] = izz[i]
            self.MM.append(M)

        self.angles = init_q
        self.q_previous = init_q
        self.dq_previous = init_dq
        self.d2q_previous = init_dq
        self.kv = 0.05
        self.leg = leg
        self.reset()
        self.q_calibration = np.array(init_q)

    def gen_jacCOM0(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        q0 = q[0]

        l0 = self.coml[0]

        JCOM0 = np.zeros((6, 4))
        JCOM0[1, 0] = -l0*np.sin(q0)
        JCOM0[2, 0] = l0*np.cos(q0)
        JCOM0[3, 0] = 1

        return JCOM0

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        q0 = q[0]
        q1 = q[1]

        L0 = self.L[0]

        l1 = self.coml[1]

        JCOM1 = np.zeros((6, 4))
        JCOM1[0, 1] = -l1*np.cos(q1)
        JCOM1[1, 0] = -(L0 + l1*np.cos(q1))*np.sin(q0)
        JCOM1[1, 1] = -l1*np.sin(q1)*np.cos(q0)
        JCOM1[2, 0] = (L0 + l1*np.cos(q1))*np.cos(q0)
        JCOM1[2, 1] = -l1*np.sin(q0)*np.sin(q1)
        JCOM1[3, 0] = 1
        JCOM1[5, 1] = 1

        return JCOM1

    def gen_jacCOM2(self, q=None):
        """Generates the Jacobian from the COM of the third
        link to the origin frame"""
        q = self.q if q is None else q

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]

        L0 = self.L[0]
        L1 = self.L[1]

        l2 = self.coml[2]

        JCOM2 = np.zeros((6, 4))
        JCOM2[0, 1] = -L1*np.cos(q1) - l2*np.cos(q1 + q2)
        JCOM2[0, 2] = -l2*np.cos(q1 + q2)
        JCOM2[1, 0] = -(L0 + L1*np.cos(q1) + l2*np.cos(q1 + q2))*np.sin(q0)
        JCOM2[1, 1] = -(L1*np.sin(q1) + l2*np.sin(q1 + q2))*np.cos(q0)
        JCOM2[1, 2] = -l2*np.sin(q1 + q2)*np.cos(q0)
        JCOM2[2, 0] = (L0 + L1*np.cos(q1) + l2*np.cos(q1 + q2))*np.cos(q0)
        JCOM2[2, 1] = -(L1*np.sin(q1) + l2*np.sin(q1 + q2))*np.sin(q0)
        JCOM2[2, 2] = -l2*np.sin(q0)*np.sin(q1 + q2)
        JCOM2[3, 0] = 1
        JCOM2[5, 1] = 1
        JCOM2[5, 2] = 1

        return JCOM2

    def gen_jacCOM3(self, q=None):
        """Generates the Jacobian from the COM of the fourth
        link to the origin frame"""
        q = self.q if q is None else q

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]

        l3 = self.coml[3]

        JCOM3 = np.zeros((6, 4))
        JCOM3[0, 1] = -L1*np.cos(q1) - L2*np.cos(q1 + q2) - l3*np.cos(q1 + q2 + q3)
        JCOM3[0, 2] = -L2*np.cos(q1 + q2) - l3*np.cos(q1 + q2 + q3)
        JCOM3[0, 3] = -l3*np.cos(q1 + q2 + q3)
        JCOM3[1, 0] = -(L0 + L1*np.cos(q1) + L2*np.cos(q1 + q2) + l3*np.cos(q1 + q2 + q3))*np.sin(q0)
        JCOM3[1, 1] = -(L1*np.sin(q1) + L2*np.sin(q1 + q2) + l3*np.sin(q1 + q2 + q3))*np.cos(q0)
        JCOM3[1, 2] = -(L2*np.sin(q1 + q2) + l3*np.sin(q1 + q2 + q3))*np.cos(q0)
        JCOM3[1, 3] = -l3*np.sin(q1 + q2 + q3)*np.cos(q0)
        JCOM3[2, 0] = (L0 + L1*np.cos(q1) + L2*np.cos(q1 + q2) + l3*np.cos(q1 + q2 + q3))*np.cos(q0)
        JCOM3[2, 1] = -(L1*np.sin(q1) + L2*np.sin(q1 + q2) + l3*np.sin(q1 + q2 + q3))*np.sin(q0)
        JCOM3[2, 2] = -(L2*np.sin(q1 + q2) + l3*np.sin(q1 + q2 + q3))*np.sin(q0)
        JCOM3[2, 3] = -l3*np.sin(q0)*np.sin(q1 + q2 + q3)
        JCOM3[3, 0] = 1
        JCOM3[5, 1] = 1
        JCOM3[5, 2] = 1
        JCOM3[5, 3] = 1

        return JCOM3

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from the end effector to the origin frame"""
        q = self.q if q is None else q

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        JEE = np.zeros((6, 4))  # (3, 4) if only x, y, z forces controlled, others dropped
        JEE[0, 1] = -L1*np.cos(q1) - L2*np.cos(q1 + q2) - L3*np.cos(q1 + q2 + q3)
        JEE[0, 2] = -L2*np.cos(q1 + q2) - L3*np.cos(q1 + q2 + q3)
        JEE[0, 3] = -L3*np.cos(q1 + q2 + q3)
        JEE[1, 0] = -(L0 + L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3*np.cos(q1 + q2 + q3))*np.sin(q0)
        JEE[1, 1] = -(L1*np.sin(q1) + L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3))*np.cos(q0)
        JEE[1, 2] = -(L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3))*np.cos(q0)
        JEE[1, 3] = -L3*np.sin(q1 + q2 + q3)*np.cos(q0)
        JEE[2, 0] = (L0 + L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3*np.cos(q1 + q2 + q3))*np.cos(q0)
        JEE[2, 1] = -(L1*np.sin(q1) + L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3))*np.sin(q0)
        JEE[2, 2] = -(L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3))*np.sin(q0)
        JEE[2, 3] = -L3*np.sin(q0)*np.sin(q1 + q2 + q3)
        JEE[3, 0] = 1
        JEE[5, 1] = 1
        JEE[5, 2] = 1
        JEE[5, 3] = 1

        return JEE

    def gen_Mq(self, q=None):
        # Mass matrix
        M0 = self.MM[0]
        M1 = self.MM[1]
        M2 = self.MM[2]
        M3 = self.MM[3]

        JCOM0 = self.gen_jacCOM0(q=q)
        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        JCOM3 = self.gen_jacCOM3(q=q)

        Mq = (np.dot(JCOM0.T, np.dot(M0, JCOM0)) +
              np.dot(JCOM1.T, np.dot(M1, JCOM1)) +
              np.dot(JCOM2.T, np.dot(M2, JCOM2)) +
              np.dot(JCOM3.T, np.dot(M3, JCOM3)))

        return Mq

    def gen_grav(self, b_orient, q=None):
        # Generate gravity term g(q)
        body_grav = np.dot(b_orient.T, self.gravity)  # adjust gravity vector based on body orientation
        body_grav = np.append(body_grav, np.array([[0, 0, 0]]).T)
        for i in range(0, 4):
            fgi = float(self.mass[i])*body_grav  # apply mass*gravity
            self.Fg.append(fgi)

        J0T = np.transpose(self.gen_jacCOM0(q=q))
        J1T = np.transpose(self.gen_jacCOM1(q=q))
        J2T = np.transpose(self.gen_jacCOM2(q=q))
        J3T = np.transpose(self.gen_jacCOM3(q=q))

        gq = J0T.dot(self.Fg[0]) + J1T.dot(self.Fg[1]) + J2T.dot(self.Fg[2]) + J3T.dot(self.Fg[3])

        return gq.reshape(-1, )

    def inv_kinematics(self, xyz):
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        x = xyz[0]
        # y = xyz[1]
        z = xyz[2]

        d = np.sqrt(x**2 + (abs(z) - L0 - L3)**2)
        q2 = np.pi/180 - np.arccos((-L1**2 - L2**2 + d**2)/(-2*L1*L2))
        alpha = np.arccos((d**2 + L1**2 - L2**2)/(2*d*L1))
        q1 = alpha - np.arcsin((x/d))
        q0 = np.arcsin(z/(L0 + L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3))
        q3 = np.pi/180 - q1 - q2  # keep foot flat for now, simplifies kinematics

        return np.array([q0, q1, q2, q3], dtype=float)

    def position(self, q=None):
        """forward kinematics
        Compute x,y,z position of end effector relative to base.
        This outputs four sets of xyz values, one for each joint including the end effector.

        q np.array: a set of angles to return positions for
        """
        if q is None:
            q0 = self.q[0]
            q1 = self.q[1]
            q2 = self.q[2]
            q3 = self.q[3]
        else:
            q0 = q[0]
            q1 = q[1]
            q2 = q[2]
            q3 = q[3]

        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        x = -np.cumsum([
            0,  # femur
            L1 * np.sin(q1),
            L2 * np.sin(q1 + q2),
            L3 * np.sin(q1 + q2 + q3)])

        y = np.cumsum([
            L0 * np.cos(q0),
            L1 * np.cos(q1) * np.cos(q0),
            L2 * np.cos(q1 + q2) * np.cos(q0),
            L3 * np.cos(q1 + q2 + q3) * np.cos(q0)])

        z = np.cumsum([
            L0 * np.sin(q0),
            L1 * np.cos(q1) * np.sin(q0),
            L2 * np.cos(q1 + q2) * np.sin(q0),
            L3 * np.cos(q1 + q2 + q3) * np.sin(q0)])

        return np.array([x, y, z], dtype=float)

    def velocity(self):  # dq=None
        # Calculate operational space linear velocity vector
        # if dq is None:
        #     dq = self.dq
        JEE = self.gen_jacEE(q=q)
        return np.dot(JEE, self.dq).flatten()

    def orientation(self, b_orient, q=None):
        # Calculate orientation of end effector in quaternions
        if q is None:
            q0 = self.q[0]
            q1 = self.q[1]
            q2 = self.q[2]
            q3 = self.q[3]
        else:
            q0 = q[0]
            q1 = q[1]
            q2 = q[2]
            q3 = q[3]

        REE = np.zeros((3, 3))  # rotation matrix
        REE[0, 0] = np.cos(q1 + q2 + q3)
        REE[0, 1] = -np.sin(q1 + q2 + q3)
        REE[1, 0] = np.sin(q1 + q2 + q3)*np.cos(q0)
        REE[1, 1] = np.cos(q0)*np.cos(q1 + q2 + q3)
        REE[1, 2] = -np.sin(q0)
        REE[2, 0] = np.sin(q0)*np.sin(q1 + q2 + q3)
        REE[2, 1] = np.sin(q0)*np.cos(q1 + q2 + q3)
        REE[2, 2] = np.cos(q0)
        # REE[3, 3] = 1
        REE = np.dot(b_orient, REE)
        q_e = transforms3d.quaternions.mat2quat(REE)
        q_e = q_e / np.linalg.norm(q_e)  # convert to unit vector quaternion

        return q_e

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

    def update_state(self, q_in):
        # Update the local variables
        # Pull values in from simulator and calibrate encoders
        self.q = np.add(q_in.flatten(), self.q_calibration)
        # self.dq = np.reshape([j[1] for j in p.getJointStates(1, range(0, 4))], (-1, 1))
        self.dq = [i * self.kv for i in self.dq_previous] + (self.q - self.q_previous) / self.dt
        # Make sure this only happens once per time step
        self.d2q = [i * self.kv for i in self.d2q_previous] + (self.dq - self.dq_previous) / self.dt

        self.q_previous = self.q
        self.dq_previous = self.dq
        self.d2q_previous = self.d2q
