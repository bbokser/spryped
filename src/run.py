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
import argparse

from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()

# parser.add_argument("ctrl", help="wbc, static_wbc, or static_ik", type=str)
parser.add_argument("--plot", help="whether or not you would like to plot results", action="store_true")
parser.add_argument("--fixed", help="fixed base: True or False", action="store_true")
parser.add_argument("--record", help="add spring: True or False", action="store_true")
parser.add_argument("--qvis", help="add qvis animation: True or False", action="store_true")
args = parser.parse_args()

if args.plot:
    plot = True
else:
    plot = False

if args.fixed:
    fixed = True
else:
    fixed = False

if args.record:
    record = True
else:
    record = False

if args.qvis:
    plot = True
else:
    plot = False

runner = Runner(dt=dt, plot=plot, fixed=fixed, record=record, qvis_animate=args.qvis)
runner.run()
runner = Runner(dt=dt)

runner.run()
