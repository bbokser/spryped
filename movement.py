'''
Copyright (C) 2014 Travis DeWolf
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

import shell

import numpy as np

def Task(arm, controller_class, dt, **kwargs):

    # set robot specific parameters ------------

    kp = 75  # 10 for plain inv kinematics
    ki = 1
    kd = 10
    ko = 100

    # generate control shell -----------------

    controller = controller_class.Control(kp=kp, ki=ki, kd=kd, ko=ko, dt=dt)
    control_shell = shell.Shell(controller=controller)

    return control_shell

    #u = controller.control(arm, **kwargs)
    #return u
