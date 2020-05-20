"""
Copyright (C) 2014 Travis DeWolf
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


class Control(object):
    """
    The base class for controllers.
    """

    def __init__(self, kp=75, ki=1, kd=5, ko=100,
                 additions=[]):
        """
        additions list: list of Addition classes to append to
                        the outgoing control signal
        kp float: the position error term gain value
        kd float: the velocity error term gain value
        """

        self.u = np.zeros((4, 1))  # control signal

        self.additions = additions
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ko = ko
        # self.task = task
        self.target = None

    def control(self, leg, x_dd_des):
        """Generates a control signal to apply to the leg"""
        raise NotImplementedError
