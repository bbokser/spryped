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

com1 = matrix([[0],
               [l1*cos(q1)],
               [l1*sin(q1)],
	       [1]])

#Tcom1 = T1_0*com1 # unsimplified
Tcom1 = matrix([[0], # simplified manually
                [l1*cos(2*q1) + L1*cos(q1)], # trig identity
                [l1*sin(2*q1) + L1*sin(q1)], # trig identity
                [1]])

# Tcom1.apply_map(attrcall('trig_reduce'))

J1 = block_matrix([[Tcom1.derivative(q1), Tcom1.derivative(q2), Tcom1.derivative(q3)]], subdivide=False)

T2_0 = T1_0*T2_1

print(J1)