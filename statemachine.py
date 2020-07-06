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


class State:
    def __init__(self, fsm):
        self.FSM = fsm
        self.s = 0
        self.sh = 0

    def enter(self):
        pass

    def execute(self, s, sh):
        self.s = s
        self.sh = sh

    def exit(self):
        pass


class Swing(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self, s, sh):
        super().execute(s, sh)
        print("swinging.")
        # self.target_l = np.array([0, 0, -0.7, self.init_alpha, self.init_beta, self.init_gamma])
        # self.tau_l = self.controller_left.control(leg=self.leg_left, target=self.target_l)

        if self.s == 1 and self.sh == 1:
            self.FSM.to_transition("toStance")
        elif self.s == 0 and self.sh == 1:
            self.FSM.to_transition("toEarly")
        elif self.s == 1 and self.sh == 0:
            self.FSM.to_transition("toLate")


class Stance(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self, s, sh):
        super().execute(s, sh)
        print("standing.")
        # u_l = self.mpc_left.mpcontrol(leg=self.leg_left)  # and positions, velocities
        # u_r = self.mpc_right.mpcontrol(leg=self.leg_right)  # and positions, velocities
        if self.s == 0:
            self.FSM.to_transition("toSwing")


class Early(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self, s, sh):
        super().execute(s, sh)
        print("early.")
        if self.s == 1:
            self.FSM.to_transition("toStance")


class Late(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self, s, sh):
        super().execute(s, sh)
        print("late.")
        if self.sh == 1:
            self.FSM.to_transition("toStance")


class Transition:
    def __init__(self, tostate):
        self.toState = tostate

    def execute(self):
        print("...transitioning...")


class FSM:
    def __init__(self, char):
        self.char = char  # passed in
        self.states = {}
        self.transitions = {}
        self.curState = None
        self.prevState = None
        self.trans = None

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

    def execute(self, s, sh):
        if self.trans:
            self.curState.exit()
            self.trans.execute()
            self.setstate(self.trans.toState)
            self.curState.enter()
            self.trans = None
        self.curState.execute(s, sh)


class Char:
    def __init__(self):
        self.FSM = FSM(self)
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
        self.FSM.execute(s, sh)
