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


class RobotBase:

    def __init__(self, init_q=None, init_dq=None,
                 dt=1e-3, singularity_thresh=0.00025, options=None):

        self.dt = dt
        self.options = options
        self.singularity_thresh = singularity_thresh

        self.init_q = np.zeros(self.DOF) if init_q is None else init_q
        self.init_dq = np.zeros(self.DOF) if init_dq is None else init_dq

    def apply_torque(self, u, dt):
        # Takes in a torque and timestep and updates the arm simulation accordingly.

        raise NotImplementedError

    def gen_jacEE(self):
        # Generates the Jacobian from end-effector to the origin frame

        raise NotImplementedError

    def gen_Mq(self):
        # Generates the mass matrix for the arm in joint space

        raise NotImplementedError

    def gen_Mx(self, JEE=None, q=None, Mq = None, **kwargs):
        # Generate the mass matrix in operational space
        if q is None:
            q = self.q
        if Mq is None:
            Mq = self.gen_Mq(q=q, **kwargs)

        if JEE is None:
            JEE = self.gen_jacEE(q=q)
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        u, s, v = np.linalg.svd(Mx_inv)
        # cut off any singular values that could cause control problems
        for i in range(len(s)):
            s[i] = 0 if s[i] < self.singularity_thresh else 1. / float(s[i])
        # numpy returns U,S,V.T, so have to transpose both here
        Mx = np.dot(v.T, np.dot(np.diag(s), u.T))

        return Mx

    # def position(self, q=None):
    # Compute x,y position of the end effector

    #   raise NotImplementedError

    def reset(self, q=[], dq=[]):
        # Resets the state of the arm

        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if q:
            assert len(q) == self.DOF
        if dq:
            assert len(dq) == self.DOF

        self.q = np.copy(self.init_q) if not q else np.copy(q)
        self.dq = np.copy(self.init_dq) if not dq else np.copy(dq)
        self.t = 0.0

    def update_state(self):
        # Update the state
        pass

    # @property
    # def x(self):
    #     return self.position()[:, -1]
