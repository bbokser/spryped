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
# import mpc
import rpc
import statemachine
import gait

import time
# import sys
# import curses

import transforms3d
import numpy as np

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, dt=1e-3, qvis_en=False, freeze_imu=False):

        self.qvis_en = qvis_en
        if qvis_en is True:
            import qvis

        self.freeze_imu = freeze_imu

        self.dt = dt
        self.u_l = np.zeros(4)
        self.u_r = np.zeros(4)

        left = 1
        right = 0

        self.leg_left = leg.Leg(dt=dt, leg=left)
        self.leg_right = leg.Leg(dt=dt, leg=right)
        controller_class = wbc
        self.controller_left = controller_class.Control(dt=dt)
        self.controller_right = controller_class.Control(dt=dt)
        self.force = rpc.Rpc(dt=dt)
        self.contact_left = contact.Contact(leg=self.leg_left, dt=dt)
        self.contact_right = contact.Contact(leg=self.leg_right, dt=dt)
        self.simulator = simulationbridge.Sim(dt=dt)
        self.state_left = statemachine.Char()
        self.state_right = statemachine.Char()

        # gait scheduler values
        self.target_init = np.array([0, 0, -0.8325])  # , self.init_alpha, self.init_beta, self.init_gamma])
        self.target_l = self.target_init
        self.target_r = self.target_init
        self.sh_l = 1  # estimated contact state (left)
        self.sh_r = 1  # estimated contact state (right)
        self.dist_force_l = np.array([0, 0, 0])
        self.dist_force_r = np.array([0, 0, 0])
        self.t_p = 0.5  # gait period, seconds
        self.phi_switch = 0.75  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.gait_left = gait.Gait(controller=self.controller_left, robotleg=self.leg_left,
                              t_p=self.t_p, phi_switch=self.phi_switch, dt=dt)
        self.gait_right = gait.Gait(controller=self.controller_right, robotleg=self.leg_right,
                               t_p=self.t_p, phi_switch=self.phi_switch, dt=dt)

        self.target = None

        # footstep planner values
        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.k_f = 0.03  # Raibert heuristic gain
        self.h = np.array([0, 0, 0.8325])  # height, assumed to be constant
        self.r_l = np.array([0, 0, -0.8325])  # initial footstep planning position
        self.r_r = np.array([0, 0, -0.8325])  # initial footstep planning position

        self.p = np.array([0, 0, 0])  # initial body position
        self.pdot_des = 0  # desired body velocity in world coords
        self.force_control_test = False

    def run(self):

        steps = 0
        t = 0  # time
        p = np.array([0, 0, 0])  # initialize body position

        t0_l = t  # starting time, left leg
        t0_r = t0_l + self.t_p / 2  # starting time, right leg. Half a period out of phase with left

        prev_state_l = str("init")
        prev_state_r = prev_state_l
        prev_contact_l = False
        prev_contact_r = False

        mpc_force = np.zeros(6)
        mpc_dt = 0.025  # mpc period
        mpc_factor = mpc_dt / self.dt  # repeat mpc every x seconds
        mpc_counter = mpc_factor
        skip = False
        t_prev = time.clock()
        time.sleep(self.dt)

        while 1:
            # t_diff = time.clock() - t_prev
            # t_prev = time.clock()
            # print(t_diff)

            # time.sleep(self.dt) # TODO: Check if this is actually necessary
            # update target after specified period of time passes
            steps += 1
            t = t + self.dt
            # print(t)
            # run simulator to get encoder and IMU feedback
            # put an if statement here once we have hardware bridge too
            q, b_orient = self.simulator.sim_run(u_l=self.u_l, u_r=self.u_r)

            if self.freeze_imu is True:
                b_orient = np.identity(3)

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
            # print(state_l, sh_l, self.dist_force_l[2])
            # forward kinematics
            pos_l = np.dot(b_orient, self.leg_left.position()[:, -1])
            pos_r = np.dot(b_orient, self.leg_right.position()[:, -1])

            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates
            p = p + pdot * self.dt  # body position in world coordinates

            theta = np.array(transforms3d.euler.mat2euler(b_orient, axes='sxyz'))

            phi = np.array(transforms3d.euler.mat2euler(b_orient, axes='szyx'))[0]
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            # rotation matrix Rz(phi)
            rz_phi = np.zeros((3, 3))
            rz_phi[0, 0] = c_phi
            rz_phi[0, 1] = s_phi
            rz_phi[1, 0] = -s_phi
            rz_phi[1, 1] = c_phi
            rz_phi[2, 2] = 1

            if state_l == ('stance' or 'early'):
                contact_l = True
            else:
                contact_l = False

            if state_r == ('stance' or 'early'):
                contact_r = True
            else:
                contact_r = False

            # this should activate just before the foot *leaves* contact.
            if contact_l is True and prev_contact_l is False:
                self.r_l = self.footstep(robotleg=1, rz_phi=rz_phi, pdot=pdot, pdot_des=0)

            if contact_r is True and prev_contact_r is False:
                self.r_r = self.footstep(robotleg=0, rz_phi=rz_phi, pdot=pdot, pdot_des=0)

            omega = np.array(self.simulator.omega_xyz)

            x_in = np.hstack([theta, p, omega, pdot]).T  # array of the states for MPC

            x_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # reference pose (desired)

            if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                if np.linalg.norm(x_in - x_ref) > 1e-2:  # then check if the error is high enough to warrant it
                    mpc_force = self.force.rpcontrol(rz_phi=rz_phi, r1=self.r_l, r2=self.r_r, x_in=x_in, x_ref=x_ref,
                                                     s_phi_1=contact_l, s_phi_2=contact_r)
                    skip = False
                else:
                    skip = True  # tells gait ctrlr to default to position control.
                    print("skipping mpc")
                mpc_counter = 0

            mpc_counter += 1

            if self.force_control_test is True:
                state_l = 'stance'
                state_r = 'stance'
                mpc_force = np.zeros(6)
            print(state_r, state_l)

            # calculate wbc control signal
            self.u_l = self.gait_left.u(state=state_l, prev_state=prev_state_l, r_in=pos_l, r_d=self.r_l,
                                        b_orient=b_orient, fr_mpc=mpc_force[0:3], skip=skip)
            # just standing for now
            self.u_r = self.gait_right.u(state=state_r, prev_state=prev_state_r, r_in=pos_r, r_d=self.r_r,
                                         b_orient=b_orient, fr_mpc=mpc_force[3:], skip=skip)
            # just standing for now

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

            prev_contact_l = contact_l
            prev_contact_r = contact_r

            # -------------------------quaternion-visualizer-animation--------------------------------- #
            if self.qvis_en is True:
                q_e = self.controller_left.q_e
                qvis.animate(q_e)
            # ----------------------------------------------------------------------------------------- #

            # print(self.dist_force_l[2])
            # print(self.reaction_torques()[0:4])

            # tau_d_left = self.contact_left.contact(leg=self.leg_left, g=self.leg_left.grav)

            # fw kinematics
            # print(np.transpose(np.append(np.dot(b_orient, self.leg_left.position()[:, -1]),
            #                              np.dot(b_orient, self.leg_right.position()[:, -1]))))
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
        # This is very simple for now, but needs to be revamped later
        if dist_force >= 70:
            sh = 1  # stance
        else:
            sh = 0  # swing

        return sh

    def footstep(self, robotleg, rz_phi, pdot, pdot_des):
        # plans next footstep location
        if robotleg == 1:
            l_i = 0.144  # left hip length
        else:
            l_i = -0.144  # right hip length

        p_hip = np.dot(rz_phi, np.array([0, l_i, 0]))
        t_stance = self.t_p * self.phi_switch
        p_symmetry = t_stance * 0.5 * pdot + self.k_f * (pdot - pdot_des)
        p_cent = 0.5 * np.sqrt(self.h / 9.807) * np.cross(pdot, self.omega_d)
        p = p_hip + p_symmetry + p_cent
        # print("p_symmetry = ", p_symmetry, robotleg)
        # print("p = ", p, robotleg)
        p[2] = -0.8325  # assume constant height for now. TODO: height changes?
        return p


