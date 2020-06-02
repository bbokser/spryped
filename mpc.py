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

from casadi import *


class Mpc(control.Control):
    def __init__(self, dt=1e-3, **kwargs):

        super(Control, self).__init__(**kwargs)
        self.dt = dt

    def _control(self, leg, x_dd_des=None):

        theta = MX.sym('theta')  # states
        p = MX.sym('p')  # states
        omega = MX.sym('omega')  # states
        pdot = MX.sym('pdot')  # states
        # grav = MX.sym('grav')  # states
        
        f1 = MX.sym('f1')  # controls
        f2 = MX.sym('f2')  # controls

        xk = ([theta.T, p.T, omega.T, pdot.T]).T

        fk = ([f1, f2]).T

        g = ([MX.zeros(1, 3), MX.zeros(1, 3), MX.zeros(1, 3), grav.T]).T

        A = MX.eye(4)
        A[0, 2] = dot(Rz(phi), dt)  # define
        A[1, 3] = dt

        B = ([MX.zeros(3, 3), MX.zeros(3, 3)],
             [MX.zeros(3, 3), MX.zeros(3, 3)],
             [MX.zeros(3, 3), MX.zeros(3, 3)],
             [MX.zeros(3, 3), MX.zeros(3, 3)])
        B[2, 0] = gIinv1*dt
        B[2, 1] = gIinv2*dt
        B[3, 0] = ones(3, 3)*dt/m
        B[3, 1] = ones(3, 3)*dt/m

        x_next = dot(A, xk) + dot(B, fk) + g  # the discrete dynamics of the system

        qp = {'x': vertcat(x_next, fk),
              'f': dot(dot((x_next-x_d).T, Q), x_next-x_d) + dot(dot(fk.T, R), fk),
              'g': dot(A, xk) + dot(B, fk) + g - x_next}
        S = qpsol('S', 'qpoases', qp)
        r = S(lbg=0)  # initial guess
        x_opt = r['x']
        print('x_opt:', x_opt)
