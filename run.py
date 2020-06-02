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

from sim import Runner

import leg  # from robot import Robot
import osc # from osc import Control
import mpc

left = 1
right = 0
dt = 1e-3

leg_left = leg.Leg(dt=dt, leg=left)
leg_right = leg.Leg(dt=dt, leg=right)

controller_class = osc
controller_left = controller_class.Control(leg=leg_left, dt=dt)
controller_right = controller_class.Control(leg=leg_right, dt=dt)

mpc_left = mpc.Mpc(leg=leg_left, dt=dt)
mpc_right = mpc.Mpc(leg=leg_right, dt=dt)

runner = Runner(dt=dt)
runner.run(leg_left=leg_left, leg_right=leg_right,
           controller_left=controller_left, controller_right=controller_right,
           mpc_left=mpc_left, mpc_right=mpc_right)
