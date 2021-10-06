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
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
import matplotlib.animation as animation

from pyquaternion import Quaternion
# -------------------------quaternion-visualizer-animation--------------------------------- #
# Matplotlib animation code based on Kieran Wynn's pyquaternion module
# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# use a different color for each axis
colors = ['r', 'g', 'b']

# set up lines and points
lines = sum([ax.plot([], [], [], c=c)
             for c in colors], [])

startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
endpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# prepare the axes limits
ax.set_xlim((-8, 8))
ax.set_ylim((-8, 8))
ax.set_zlim((-8, 8))

ax.view_init(30, 0)  # set point-of-view: specified by (altitude degrees, azimuth degrees)
# q_e = np.array([0., 0., 0., 0.])


def animate(q_e):
    # animation function.  This will be called sequentially with the frame number

    q_in = Quaternion(q_e[0], q_e[1], q_e[2], q_e[3])
    for line, start, end in zip(lines, startpoints, endpoints):
        # end *= 5
        start = q_in.rotate(start)
        end = q_in.rotate(end)

        line.set_data([start[0], end[0]], [start[1], end[1]])
        line.set_3d_properties([start[2], end[2]])

        # pt.set_data(x[-1:], y[-1:])
        # pt.set_3d_properties(z[-1:])

    fig.canvas.draw()
    plt.pause(0.0001)
    return lines
