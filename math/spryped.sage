var('q1,q2,q3,q4,L1,L2,L3,L4,l1,l2,l3,l4')

T1_0 = matrix([[1, 0, 0, 0],
     	       [0, cos(q1), -sin(q1), L1*cos(q1)],
	       [0, sin(q1), cos(q1), L1*sin(q1)],
	       [0, 0, 0, 1]])
	       
T2_1 = matrix([[cos(q2), -sin(q2), 0, L2*sin(q2)],
               [sin(q2), cos(q2), 0, L2*cos(q2)],
	       [0, 0, 1, 0],
	       [0, 0, 0, 1]])

T3_2 = matrix([[cos(q3), -sin(q3), 0, L3*sin(q3)],
               [sin(q3), cos(q3), 0, L3*cos(q3)],
	       [0, 0, 1, 0],
	       [0, 0, 0, 1]])

T4_3 = matrix([[cos(q4), -sin(q4), 0, L4*sin(q4)],
               [sin(q4), cos(q4), 0, L4*cos(q4)],
	       [0, 0, 1, 0],
	       [0, 0, 0, 1]])

com1 = matrix([[0],
               [l1*cos(q1)],
               [l1*sin(q1)],
	       [1]])

com2 = matrix([[l2*sin(q2)],
               [l2*cos(q2)],
               [0],
	       [1]])

com3 = matrix([[l3*sin(q3)],
               [l3*cos(q3)],
               [0],
	       [1]])

com4 = matrix([[l4*sin(q4)],
               [l4*cos(q4)],
               [0],
	       [1]])

#Tcom1 = T1_0*com1 # unsimplified
Tcom1 = matrix([[0], # simplified manually
                [l1*cos(2*q1) + L1*cos(q1)], # trig identity
                [l1*sin(2*q1) + L1*sin(q1)], # trig identity
                [1]])
# Tcom1.apply_map(attrcall('trig_reduce'))

J1 = block_matrix([[Tcom1.derivative(q1), Tcom1.derivative(q2), Tcom1.derivative(q3), Tcom1.derivative(q4)]], subdivide=False)

T2_0 = T1_0*T2_1

#Tcom2 = T2_0*com2 # unsimplified
Tcom2 = matrix([[L2*sin(q2)], # simplified manually
                [cos(q1)*(l2 + L1 + L2*cos(q2))], # trig identity
                [sin(q1)*(l2 + L1 + L2*cos(q2))], # trig identity
                [1]])

J2 = block_matrix([[Tcom2.derivative(q1), Tcom2.derivative(q2), Tcom2.derivative(q3), Tcom2.derivative(q4)]], subdivide=False)

#T3_0 = T3_2*T2_1*T1_0 # unsimplified
T3_0 = matrix([[cos(q2+q3), -sin(q2+q3)*cos(q1), sin(q2+q3)*sin(q1), sin(q2+q3)*L1*cos(q1) + L2*sin(q2-q3) + L3*sin(q3)], # simplified manually
               [sin(q2+q3), cos(q2+q3)*cos(q1), -cos(q2+q3)*sin(q1), sin(q2+q3)*L1*cos(q1) + L2*cos(q2-q3) + L3*cos(q3)],
               [0, sin(q1), cos(q1), L1*sin(q1)],
               [0, 0, 0, 1]])
Tcom3 = T3_0*com3 # unsimplified

J3 = block_matrix([[Tcom3.derivative(q1), Tcom3.derivative(q2), Tcom3.derivative(q3), Tcom3.derivative(q4)]], subdivide=False)

T4_0 = T4_3*T3_2*T2_1*T1_0

#Tcom4 = T4_0*com4
Tcom4 = matrix([[l4*(sin(q2 + q3 + 2*q4)/2 - sin(q2 + q3)/2) + L4*sin(q4) + L3*sin(q3 - q4) + L2*cos(q3 + q4)*sin(q2) - L2*sin(q3 + q4)*cos(q2) - (l4*sin(q2 + q3 + q4)*cos(q1 + q4))/2 - L1*sin(q2 + q3 + q4)*cos(q1) - (l4*cos(q1 - q4)*sin(q2 + q3 + q4))/2],
                [L4*cos(q4) - (l4*(cos(q2 + q3 + 2*q4) - cos(q2 + q3)))/2 + L3*cos(q3 - q4) + L2*cos(q3 + q4)*cos(q2) + L2*sin(q3 + q4)*sin(q2) + (l4*cos(q2 + q3 + q4)*cos(q1 + q4))/2 + L1*cos(q2 + q3 + q4)*cos(q1) + (l4*cos(q1 - q4)*cos(q2 + q3 + q4))/2],
                [sin(q1)*(L1 + l4*cos(q4))],
                [1]])

J4 = block_matrix([[Tcom4.derivative(q1), Tcom4.derivative(q2), Tcom4.derivative(q3), Tcom4.derivative(q4)]], subdivide=False)

#print(Tcom4(q1=1,q2=2,q3=3,q4=4,l1=5,l2=6,l3=7,l4=8,L1=9,L2=10,L3=11,L4=12))
#print(Tcom4simp(q1=1,q2=2,q3=3,q4=4,l1=5,l2=6,l3=7,l4=8,L1=9,L2=10,L3=11,L4=12))
latex.matrix_column_alignment('c')
print("J1 = ", latex(J1))
print("J2 = ", latex(J2))
print("J3 = ", latex(J3))
print("J4 = ", latex(J4))
