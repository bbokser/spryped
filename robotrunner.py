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

import time
import sys
import curses

import numpy as np

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, leg_left, leg_right,
                 controller_left, controller_right,
                 mpc_left, mpc_right, contact_left, contact_right,
                 simulator, gait_l, gait_r, dt=1e-3):

        self.dt = dt
        self.u_l = np.zeros(4)
        self.u_r = np.zeros(4)
        self.leg_left = leg_left
        self.leg_right = leg_right
        self.controller_left = controller_left
        self.controller_right = controller_right
        self.mpc_left = mpc_left
        self.mpc_right = mpc_right
        self.contact_left = contact_left
        self.contact_right = contact_right
        self.simulator = simulator
        self.gait_l = gait_l
        self.gait_r = gait_r

        self.init_alpha = -np.pi / 2
        self.init_beta = 0  # can't control, ee Jacobian is zeros in that row
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -0.8, self.init_alpha, self.init_beta, self.init_gamma])
        self.target_l = self.target_init
        self.target_r = self.target_init
        self.sh_l = 1  # estimated contact state (left)
        self.sh_r = 1  # estimated contact state (right)
        self.dist_force_l = np.array([0, 0, 0])
        self.dist_force_r = np.array([0, 0, 0])

        self.t_p = 4  # gait period, seconds
        self.phi_switch = 0.75  # switching phase, must be between 0 and 1. Switches b/t swing and stance

    def run(self):

        steps = 0
        t = 0  # time

        t0_l = t  # starting time, left leg
        t0_r = t0_l + self.t_p/2  # starting time, right leg. Half a period out of phase with left

        while 1:
            time.sleep(self.dt)
            # update target after specified period of time passes
            steps = steps + 1
            t = t + self.dt

            # gait scheduler
            s_l = self.gait_scheduler(t, t0_l)
            s_r = self.gait_scheduler(t, t0_r)
            sh_l = self.gait_estimator(self.dist_force_l[2])
            sh_r = self.gait_estimator(self.dist_force_r[2])

            self.gait_l.FSM.execute(s_l, sh_l)
            self.gait_r.FSM.execute(s_r, sh_r)

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

            # self.target_r = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
            self.u_r = -self.controller_right.control(
                leg=self.leg_right, target=self.target_r, base_orientation=base_orientation)

            # self.target_l = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
            self.u_l = -self.controller_left.control(
                leg=self.leg_left, target=self.target_l, base_orientation=base_orientation)

            dist_tau_l = self.contact_left.disturbance_torque(Mq=self.controller_left.Mq,
                                                              dq=self.leg_left.dq,
                                                              tau_actuated=-self.u_l,
                                                              grav=self.controller_left.grav)
            self.dist_force_l = np.dot(np.linalg.pinv(np.transpose(self.leg_left.gen_jacEE()[0:3])),
                                       np.array(dist_tau_l))

            dist_tau_r = self.contact_right.disturbance_torque(Mq=self.controller_right.Mq,
                                                               dq=self.leg_right.dq,
                                                               tau_actuated=-self.u_r,
                                                               grav=self.controller_right.grav)
            self.dist_force_r = np.dot(np.linalg.pinv(np.transpose(self.leg_right.gen_jacEE()[0:3])),
                                       np.array(dist_tau_r))
            # print(self.dist_force_l[2])
            # print(self.reaction_torques()[0:4])
            '''
            if self.statemachine() == 1:
                self.target_r = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
                self.tau_r = self.controller_right.control(leg=self.leg_right, target=self.target_r)
                u_r = self.leg_right.apply_torque(u=self.tau_r, dt=self.dt)

                # u_l = self.mpc_left.mpcontrol(leg=self.leg_left)  # and positions, velocities

            else:
                self.target_l = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
                self.tau_l = self.controller_left.control(leg=self.leg_left, target=self.target_l)
                u_l = self.leg_left.apply_torque(u=self.tau_l, dt=self.dt)

                # u_r = self.mpc_right.mpcontrol(leg=self.leg_right)  # and positions, velocities
            '''

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





