"""
Copyright (C) 2020 Benjamin Bokser
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
