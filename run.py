'''
Copyright (C) 2015 Travis DeWolf
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
'''

from sim import Runner

import robot # from robot import Robot

import osc # from osc import Control

import movement

dt = 1e-3
robot = robot.Robot(dt=dt)
controller_class = osc
task = movement.Task
control_shell = task(robot, controller_class)
runner = Runner(dt=dt)
runner.run(robot=robot, control_shell=control_shell)
