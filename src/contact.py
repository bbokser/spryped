"""
Copyright (C) 2020 Benjamin Bokser
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
        self.tau_d = None
        la = 15  # lambda, cutoff frequency. PLACEHOLDER VALUE
        self.gamma = np.exp(-la * self.dt)
        self.beta = (1 - self.gamma) / (self.gamma * self.dt)

    def disturbance_torque(self, Mq, dq, tau_actuated, grav):
        """
        Generalized momentum-based discrete time filtered disturbance torque observer
        Isolates disturbance torque from other sources
        From:
        Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains
        Gerardo Bledt, Patrick M. Wensing, Sam Ingersoll, and Sangbae Kim
        """
        p = np.dot(Mq, dq)
        self.tau_d = self.gamma * self.delay_term \
            + self.beta * p \
            - (1 - self.gamma) * (self.beta * p + tau_actuated + grav)
        self.delay_term = self.tau_d - self.beta * p
        return self.tau_d
