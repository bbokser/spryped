"""
Copyright (C) 2014 Travis DeWolf
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np


class Control(object):
    """
    The base class for controllers.
    """

    def __init__(self, additions=[]):
        """
        additions list: list of Addition classes to append to
                        the outgoing control signal
        kp float: the position error term gain value
        kd float: the velocity error term gain value
        """

        self.u = np.zeros((4, 1))  # control signal

        self.additions = additions

        self.target = None
        self.b_orient = None
        self.force = None

    def wb_control(self, leg, target, force, x_dd_des):
        """Generates a position and/or force control signal to apply to the leg"""
        raise NotImplementedError
