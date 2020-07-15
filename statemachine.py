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
from scipy.interpolate import CubicSpline
import numpy as np


class State:
    def __init__(self, fsm):
        self.FSM = fsm

    def enter(self):
        pass

    def execute(self):
        # self.s = s
        # self.sh = sh
        pass

    def exit(self):
        pass


class Swing(State):
    def __init__(self, fsm):
        super().__init__(fsm)
        self.steps = 0
        self.target = None
        self.trajectory = None
        self.check = 0

    def enter(self):
        self.trajectory = self.traj(0, 0.5, 0, 0.2)
        self.check = 1

    def execute(self):
        # super().execute(s, sh)
        print("swinging.")
        if self.FSM.s == 1 and self.FSM.sh == 1:
            self.FSM.to_transition("toStance")
        elif self.FSM.s == 0 and self.FSM.sh == 1:
            self.FSM.to_transition("toEarly")
        elif self.FSM.s == 1 and self.FSM.sh == 0:
            self.FSM.to_transition("toLate")

        self.steps += 1
        # set target position
        if self.check == 0:
            self.trajectory = self.traj(0, 0.5, 0, 0.2)  # set initial trajectory if enter() was skipped

        target = self.trajectory[:, self.steps]
        target = np.hstack(np.append(target,
                                     np.array([self.FSM.init_alpha, self.FSM.init_beta, self.FSM.init_gamma])))
        # calculate wbc control signal
        u = -self.FSM.controller.control(leg=self.FSM.leg,
                                         target=target, base_orientation=self.FSM.base_orientation)

        return u

    def exit(self):
        self.steps = 0
        self.check = 0

    def traj(self, x_prev, x_d, y_prev, y_d):
        # Generates cubic spline curve trajectory for foot swing

        # number of time steps allotted for swing trajectory

        timesteps = self.FSM.t_p * (1 - self.FSM.phi_switch) / self.FSM.dt
        if not timesteps.is_integer():
            print("Error: period and timesteps are not divisible")  # if t_p is variable
        path = np.zeros(int(timesteps))

        horizontal = np.array([0.0, timesteps / 2, timesteps])
        vertical = np.array([-0.8325, -0.7, -0.8325])  # z position
        cs = CubicSpline(horizontal, vertical)

        # create evenly spaced sample points of desired trajectory
        for t in range(int(timesteps)):
            path[t] = cs(t)
        z_traj = path
        x_traj = np.linspace(x_prev, x_d, timesteps)
        y_traj = np.linspace(y_prev, y_d, timesteps)

        return np.array([x_traj.T, y_traj.T, z_traj.T])


class Stance(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        # super().execute(s, sh)
        print("standing.")
        if self.FSM.s == 0:
            self.FSM.to_transition("toSwing")
        # Execute MPC
        # set target position
        target = np.array([0, 0, -0.8235, self.FSM.init_alpha, self.FSM.init_beta, self.FSM.init_gamma])
        # calculate wbc control signal
        u = -self.FSM.controller.control(
            leg=self.FSM.leg, target=target, base_orientation=self.FSM.base_orientation)

        # u_l = self.mpc_left.mpcontrol(leg=self.leg_left)  # and positions, velocities

        return u


class Early(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        # super().execute(s, sh)
        print("early.")
        if self.FSM.s == 1:
            self.FSM.to_transition("toStance")
        # Execute MPC
        # set target position
        target = np.array([0, 0, -0.8235, self.FSM.init_alpha, self.FSM.init_beta, self.FSM.init_gamma])
        # calculate wbc control signal
        u = -self.FSM.controller.control(
            leg=self.FSM.leg, target=target, base_orientation=self.FSM.base_orientation)

        # u_l = self.mpc_left.mpcontrol(leg=self.leg_left)  # and positions, velocities

        return u


class Late(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        print("late.")
        if self.FSM.sh == 1:
            self.FSM.to_transition("toStance")
        # position command should freeze at target position
        # set target position
        target = np.array([0, 0, -0.8235, self.FSM.init_alpha, self.FSM.init_beta, self.FSM.init_gamma])
        # calculate wbc control signal
        u = -self.FSM.controller.control(
            leg=self.FSM.leg, target=target, base_orientation=self.FSM.base_orientation)

        return u


class Transition:
    def __init__(self, tostate):
        self.toState = tostate

    def execute(self):
        print("...transitioning...")


class FSM:
    def __init__(self, char, controller):
        self.char = char  # passed in
        self.states = {}
        self.transitions = {}
        self.curState = None
        self.prevState = None
        self.trans = None

        self.controller = controller
        self.s = None
        self.sh = None
        self.t_p = None
        self.phi_switch = 0.75
        self.base_orientation = None
        self.leg = None
        self.init_alpha = -np.pi / 2
        self.init_beta = 0  # can't control, ee Jacobian is zeros in that row
        self.init_gamma = 0
        self.dt = None

    def add_transition(self, transname, transition):
        self.transitions[transname] = transition

    def add_state(self, statename, state):
        self.states[statename] = state

    def setstate(self, statename):
        # look for whatever state we passed in within the states dict
        self.prevState = self.curState
        self.curState = self.states[statename]

    def to_transition(self, to_trans):
        # set the transition state
        self.trans = self.transitions[to_trans]

    def execute(self, s, sh, t_p, base_orientation, leg, dt):
        self.s = s
        self.sh = sh
        self.t_p = t_p
        self.base_orientation = base_orientation
        self.leg = leg
        self.dt = dt

        if self.trans:
            self.curState.exit()
            self.trans.execute()
            self.setstate(self.trans.toState)
            self.curState.enter()
            self.trans = None

        u = self.curState.execute()

        return u


class Char:
    def __init__(self, controller):
        self.FSM = FSM(self, controller)
        self.Swing = True

        self.FSM.add_state("Swing", Swing(self.FSM))
        self.FSM.add_state("Stance", Stance(self.FSM))
        self.FSM.add_state("Early", Early(self.FSM))
        self.FSM.add_state("Late", Late(self.FSM))

        self.FSM.add_transition("toSwing", Transition("Swing"))
        self.FSM.add_transition("toStance", Transition("Stance"))
        self.FSM.add_transition("toEarly", Transition("Early"))
        self.FSM.add_transition("toLate", Transition("Late"))

        self.FSM.setstate("Swing")

    def execute(self):
        self.FSM.execute(s, sh, t_p, base_orientation, leg, dt)
