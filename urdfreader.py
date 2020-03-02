'''
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
import numpy as np

import csv

#import xml.etree.ElementTree as ET
#xmltree = ET.parse('spryped_urdf_rev05/urdf/spryped_urdf_rev05.urdf')

ixx, ixy, ixz, iyy, iyz, izz, mass = ([] for i in range(7))

# initializing titles and rows list for csv reader
fields = []
rows = []
with open('spryped_urdf_rev05/urdf/spryped_urdf_rev05.urdf', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = csvreader.next()
    for row in csvreader:
        rows.append(row)
    
#for v in xmltree.iter('mass'):
#    mass.append((v.attrib['value']))
    
for v in xmltree.iter('inertia'):
    ixx.append((v.attrib['ixx']))
    ixy.append((v.attrib['ixy']))
    ixz.append((v.attrib['ixz']))
    iyy.append((v.attrib['iyy']))
    iyz.append((v.attrib['iyz']))
    izz.append((v.attrib['izz']))

class Robot:

    def __init__(self, **kwargs):
        # link lengths
        # Not provided by urdf so update this when you update model
        l1 = 0.05 # femur left
        l2 = 0.199 # tibiotarsus left
        l3 = 0.5 # tarsometatarsus left
        l4 = 0.05 # toe left
        l5 = 0.05 # femur right
        l6 = 0.199 # tibiotarsus right
        l7 = 0.5 # tarsometatarsus right
        l8 = 0.05 # toe right
        self.L = np.array([l1, l2, l3, l4, l5, l6, l7, l8])
        # link masses
        m0 = mass[0] # body
        m1 = mass[1] # femur left
        m2 = mass[2] # tibiotarsus left
        m3 = mass[3] # tarsometatarsus left
        m4 = mass[4] # toe left
        m5 = mass[5] # femur right
        m6 = mass[6] # tibiotarsus right
        m7 = mass[7] # tarsometatarsus right
        m8 = mass[8] # toe left
        if mass[1]!=mass[5]:
            print("WARNING: femur L/R masses unequal, check CAD")
        if mass[2]!=mass[6]:
            print("WARNING: tibiotarsus L/R masses unequal, check CAD")
        if mass[3]!=mass[7]:
            print("WARNING: tarsometatarsus L/R masses unequal, check CAD")
        if mass[4]!=mass[8]:
            print("WARNING: toe L/R masses unequal, check CAD")
        # mass matrices
        self.MM = []
        for i in range(9):
            M = np.zeros((6, 6))
            M[0:3, 0:3] = np.eye(3)*float(mass[i])
            M[3, 3] = float(ixx[i])
            M[3, 4] = float(ixy[i])
            M[3, 5] = float(ixz[i])
            M[4, 3] = -float(ixy[i])
            M[4, 4] = float(iyy[i])
            M[4, 5] = float(iyz[i])
            M[5, 3] = -float(ixz[i])
            M[5, 4] = -float(iyz[i])
            M[5, 5] = float(izz[i])
            #self.MM.insert(i,M)
            self.MM.append(M)
        JCOM1 = np.zeros((6, 3))
        JCOM1[0, 0] = 
robert = Robot()
print(robert.MM[7])
