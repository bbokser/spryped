"""
Copyright (C) 2020 Benjamin Bokser
"""
import numpy as np
from scipy.interpolate import CubicSpline


class Gait:
    def __init__(self, controller, robotleg, t_p, phi_switch, hconst, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.t_p = t_p
        self.phi_switch = phi_switch
        self.dt = dt
        self.init_alpha = -np.pi / 2
        self.init_beta = 0  # can't control, ee Jacobian is zeros in that row
        self.init_gamma = 0
        self.init_angle = np.array([self.init_alpha, self.init_beta, self.init_gamma])
        self.controller = controller
        self.robotleg = robotleg
        self.x_last = None
        self.target = None
        self.hconst = hconst
        self.r_save = np.array([0, 0, -self.hconst])
        self.target = np.hstack(np.append(np.array([0, 0, -self.hconst]), self.init_angle))

    def u(self, state, prev_state, r_in, r_d, delp, b_orient, fr_mpc, skip):

        if state == 'swing':
            if prev_state != state:
                self.swing_steps = 0
                self.trajectory = self.traj(x_prev=r_in[0], x_d=r_d[0], y_prev=r_in[1], y_d=r_d[1])

            # set target position
            self.target = self.trajectory[:, self.swing_steps]
            self.target = np.hstack(np.append(self.target, self.init_angle))

            self.swing_steps += 1
            # calculate wbc control signal
            u = -self.controller.wb_control(leg=self.robotleg, target=self.target, b_orient=b_orient, force=None)

        elif state == 'stance' or state == 'early':

            if prev_state != state and prev_state != 'early':
                # if contact has just been made, save that contact point as the new target to stay at
                # (stop following through with trajectory)
                self.r_save = r_in
            self.r_save = self.r_save - delp
            self.target = np.hstack(np.append(self.r_save, self.init_angle))
            self.target[2] = -self.hconst  # maintain height estimate at constant to keep ctrl simple

            if skip is True:  # skip force control this time because robot is already in correct pose
                # calculate wbc control signal
                u = -self.controller.wb_control(leg=self.robotleg, target=self.target, b_orient=b_orient, force=None)
            else:
                force = fr_mpc
                u = -self.controller.wb_control(leg=self.robotleg, target=self.target, b_orient=b_orient, force=force)

        elif state == 'late':
            # calculate wbc control signal
            u = -self.controller.wb_control(leg=self.robotleg, target=self.target, b_orient=b_orient, force=None)

        else:
            u = None

        return u

    def traj(self, x_prev, x_d, y_prev, y_d):
        # Generates cubic spline curve trajectory for foot swing

        # number of time steps allotted for swing trajectory
        timesteps = self.t_p * (1 - self.phi_switch) / self.dt
        timesteps = round(timesteps, 2)  # fixes floating point math issues
        if not timesteps.is_integer():
            print("Error: period and timesteps are not divisible")  # if t_p is variable
        timesteps = int(timesteps)
        path = np.zeros(timesteps)

        horizontal = np.array([0.0, timesteps / 2, timesteps])
        # z traj assumed constant body height & flat floor
        # vertical = np.array([-self.hconst, -self.hconst + 0.1325, -self.hconst])
        vertical = np.array([-self.hconst, -self.hconst + 0.1325, -self.hconst])
        # vertical = np.array([-self.hconst, -0.7, -self.hconst])
        cs = CubicSpline(horizontal, vertical)

        # create evenly spaced sample points of desired trajectory
        for t in range(timesteps):
            path[t] = cs(t)
        z_traj = path
        x_traj = np.linspace(x_prev, x_d, timesteps)
        y_traj = np.linspace(y_prev, y_d, timesteps)

        return np.array([x_traj.T, y_traj.T, z_traj.T])
