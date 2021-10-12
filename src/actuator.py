"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np


def actuate(i, q_dot, gr=7, gr_out=7, tau_stall=50, rpm_free=190, i_max=13, v_max=48):
    """
    Motor Dynamics
    i = current, Amps
    q_dot = angular velocity of link, after gear ratio (rad/s)
    gr = gear ratio of base actuator (which max torque rating is based on)
    gr_out = gear ratio of output after additional gearing is added
    """
    # omega_max = rpm_free*(2*np.pi/60)*gr  # motor rated speed, rpm to radians and gear ratio
    omega = abs(q_dot * gr_out)  # angular speed (NOT velocity) of motor in rad/s
    tau_stall_m = tau_stall/gr
    kt_m = tau_stall_m / i_max
    # r = (v_max ** 2) / (omega_max * tau_stall_m)
    r = kt_m*v_max/tau_stall_m
    # kt_m1 = tau_stall * r / (v_max*gr)  # torque constant of the motor, Nm/amp. == v_max/omega_max
    tau_m = kt_m * i
    tau_max_m = abs(- omega*(kt_m**2)/r + v_max*kt_m/r)  # max motor torque for given speed
    tau_max_m = np.clip(tau_max_m, -tau_stall_m, tau_stall_m)
    tau_m = np.clip(tau_m, -tau_max_m, tau_max_m)  # ensure motor torque remains within torque-speed curve

    return tau_m * gr_out  # actuator output torque


def saturate(i, q_dot, gr=7, gr_out=7, tau_stall=50, rpm_free=190, i_max=13, v_max=48):
    """
    Uses simple inverse torque-speed relationship
    """
    omega_max = rpm_free * (2 * np.pi / 60) * gr  # rated speed, rpm to radians/s
    omega = abs(q_dot * gr_out)  # angular speed (NOT velocity) of motor in rad/s
    tau_max = (1-omega/omega_max) * tau_stall * gr_out/gr
    return np.clip(i,-tau_max,tau_max)