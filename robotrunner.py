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
import contact
import simulationbridge
import leg
import wbc
import mpc
import statemachine

import time
import sys
import curses

import numpy as np
from scipy.interpolate import CubicSpline

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, dt=1e-3):

        self.dt = dt
        self.u_l = np.zeros(4)
        self.u_r = np.zeros(4)

        left = 1
        right = 0

        self.leg_left = leg.Leg(dt=dt, leg=left)
        self.leg_right = leg.Leg(dt=dt, leg=right)
        controller_class = wbc
        self.controller_left = controller_class.Control(leg=self.leg_left, dt=dt)
        self.controller_right = controller_class.Control(leg=self.leg_right, dt=dt)
        self.mpc_left = mpc.Mpc(leg=self.leg_left, dt=dt)
        self.mpc_right = mpc.Mpc(leg=self.leg_right, dt=dt)
        self.contact_left = contact.Contact(leg=self.leg_left, dt=dt)
        self.contact_right = contact.Contact(leg=self.leg_right, dt=dt)
        self.simulator = simulationbridge.Sim(dt=dt)
        self.state_left = statemachine.Char()
        self.state_right = statemachine.Char()

        self.target_init = np.array([0, 0, -0.8325])  # , self.init_alpha, self.init_beta, self.init_gamma])
        self.target_l = self.target_init
        self.target_r = self.target_init
        self.sh_l = 1  # estimated contact state (left)
        self.sh_r = 1  # estimated contact state (right)
        self.dist_force_l = np.array([0, 0, 0])
        self.dist_force_r = np.array([0, 0, 0])

        self.t_p = 2  # gait period, seconds
        self.phi_switch = 0.75  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.

        self.gait_left = Gait(controller=self.controller_left, robotleg=self.leg_left,
                              t_p=self.t_p, phi_switch=self.phi_switch, dt=dt)
        self.gait_right = Gait(controller=self.controller_right, robotleg=self.leg_right,
                               t_p=self.t_p, phi_switch=self.phi_switch, dt=dt)

        self.target = None

    def run(self):

        steps = 0
        t = 0  # time

        t0_l = t  # starting time, left leg
        t0_r = t0_l + self.t_p / 2  # starting time, right leg. Half a period out of phase with left

        prev_state_l = str("init")
        prev_state_r = prev_state_l

        while 1:
            time.sleep(self.dt)
            # update target after specified period of time passes
            steps += 1
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            # put an if statement here once we have hardware bridge too
            q, base_orientation = self.simulator.sim_run(u_l=self.u_l, u_r=self.u_r)

            q_left = q[0:4]
            q_left[1] *= -1
            q_left[2] *= -1
            q_left[3] *= -1

            q_right = q[4:8]
            q_right[3] *= -1

            # enter encoder values into leg kinematics/dynamics
            self.leg_left.update_state(q_in=q_left)
            self.leg_right.update_state(q_in=q_right)

            # gait scheduler
            s_l = self.gait_scheduler(t, t0_l)
            s_r = self.gait_scheduler(t, t0_r)
            sh_l = self.gait_estimator(self.dist_force_l[2])
            sh_r = self.gait_estimator(self.dist_force_r[2])

            state_l = self.state_left.FSM.execute(s_l, sh_l)
            state_r = self.state_right.FSM.execute(s_r, sh_r)

            # forward kinematics
            pos_l = np.dot(base_orientation, self.leg_left.position()[:, -1])
            pos_r = np.dot(base_orientation, self.leg_right.position()[:, -1])

            # calculate wbc control signal
            self.u_l = self.gait_left.u(state=state_l,
                                        prev_state=prev_state_l, pos_in=pos_l, pos_d=self.target_l,
                                        base_orientation=base_orientation)  # just standing for now

            self.u_r = self.gait_right.u(state=state_r,
                                         prev_state=prev_state_r, pos_in=pos_r, pos_d=self.target_r,
                                         base_orientation=base_orientation)  # just standing for now

            # receive disturbance torques
            dist_tau_l = self.contact_left.disturbance_torque(Mq=self.controller_left.Mq,
                                                              dq=self.leg_left.dq,
                                                              tau_actuated=-self.u_l,
                                                              grav=self.controller_left.grav)
            dist_tau_r = self.contact_right.disturbance_torque(Mq=self.controller_right.Mq,
                                                               dq=self.leg_right.dq,
                                                               tau_actuated=-self.u_r,
                                                               grav=self.controller_right.grav)
            # convert disturbance torques to forces
            self.dist_force_l = np.dot(np.linalg.pinv(np.transpose(self.leg_left.gen_jacEE()[0:3])),
                                       np.array(dist_tau_l))
            self.dist_force_r = np.dot(np.linalg.pinv(np.transpose(self.leg_right.gen_jacEE()[0:3])),
                                       np.array(dist_tau_r))

            prev_state_l = state_l
            prev_state_r = state_r

            # print(self.dist_force_l[2])
            # print(self.reaction_torques()[0:4])

            # tau_d_left = self.contact_left.contact(leg=self.leg_left, g=self.leg_left.grav)

            # fw kinematics
            # print(np.transpose(np.append(np.dot(base_orientation, self.leg_left.position()[:, -1]),
            #                              np.dot(base_orientation, self.leg_right.position()[:, -1]))))
            # joint velocity
            # print("vel = ", self.leg_left.velocity())
            # encoder feedback
            # print(np.transpose(np.append(self.leg_left.q, self.leg_right.q)))

            # sys.stdout.write("\033[F")  # back to previous line
            # sys.stdout.write("\033[K")  # clear line

    def gait_scheduler(self, t, t0):
        # Add variable period later
        phi = np.mod((t - t0) / self.t_p, 1)

        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance

        return s

    def gait_estimator(self, dist_force):
        # Determines whether foot is actually in contact or not

        if dist_force >= 10:
            sh = 1  # stance
        else:
            sh = 0  # swing

        return sh


class Gait:
    def __init__(self, controller, robotleg, t_p, phi_switch, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.t_p = t_p
        self.phi_switch = phi_switch
        self.dt = dt
        self.init_alpha = -np.pi / 2
        self.init_beta = 0  # can't control, ee Jacobian is zeros in that row
        self.init_gamma = 0
        self.controller = controller
        self.robotleg = robotleg

    def u(self, state, prev_state, pos_in, pos_d, base_orientation):

        if state == 'swing':

            if prev_state != state:
                self.swing_steps = 0
                self.trajectory = self.traj(x_prev=pos_in[0], x_d=pos_d[0], y_prev=pos_in[1], y_d=pos_d[1])

            # set target position
            target = self.trajectory[:, self.swing_steps]
            target = np.hstack(np.append(target,
                                         np.array([self.init_alpha, self.init_beta, self.init_gamma])))

            self.swing_steps += 1
            # calculate wbc control signal
            u = -self.controller.control(leg=self.robotleg, target=target, base_orientation=base_orientation)

        elif state == 'stance' or 'early':
            # execute MPC
            target = np.array([0, 0, -0.8235, self.init_alpha, self.init_beta, self.init_gamma])
            # calculate wbc control signal
            u = -self.controller.control(leg=self.robotleg, target=target, base_orientation=base_orientation)

        elif state == 'late':  # position command should freeze at last target position
            target = np.array([0, 0, -0.8235, self.init_alpha, self.init_beta, self.init_gamma])
            # calculate wbc control signal
            u = -self.controller.control(leg=self.robotleg, target=target, base_orientation=base_orientation)

        else:
            u = None  # this'll throw an error if state machine is haywire or something

        return u

    def traj(self, x_prev, x_d, y_prev, y_d):
        # Generates cubic spline curve trajectory for foot swing

        # number of time steps allotted for swing trajectory

        timesteps = self.t_p * (1 - self.phi_switch) / self.dt
        if not timesteps.is_integer():
            print("Error: period and timesteps are not divisible")  # if t_p is variable
        path = np.zeros(int(timesteps))

        horizontal = np.array([0.0, timesteps / 2, timesteps])
        vertical = np.array([-0.8325, -0.7, -0.8325])  # z position assumed constant body height and flat floor
        cs = CubicSpline(horizontal, vertical)

        # create evenly spaced sample points of desired trajectory
        for t in range(int(timesteps)):
            path[t] = cs(t)
        z_traj = path
        x_traj = np.linspace(x_prev, x_d, timesteps)
        y_traj = np.linspace(y_prev, y_d, timesteps)

        return np.array([x_traj.T, y_traj.T, z_traj.T])
