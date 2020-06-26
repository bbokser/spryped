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

import control


class Contact:

    def __init__(self, dt=1e-3, **kwargs):
        """
        contact estimation
        :param dt: time step or sampling time
        """

        self.dt = dt
        self.delay_term = 0
        self.tau_d = 0
        la = 1  # lambda, cutoff frequency. PLACEHOLDER VALUE
        self.gamma = np.exp(-la * self.dt)
        self.beta = (1 - self.gamma) / (self.gamma * self.dt)

    def tau_d(self, p_k, tau, g, leg):
        """
        Generalized momentum-based discrete time filtered disturbance torque observer
        Isolates disturbance torque from other sources
        """
        self.tau_d = self.gamma * self.delay_term + self.beta * p_k - (1 - self.gamma) * (self.beta * p + tau - g)
        self.delay_term = self.tau_d - beta * p_k
        return self.tau_d
