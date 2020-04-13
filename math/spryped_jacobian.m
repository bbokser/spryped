clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms q1 q2 q3 q4 L1 L2 L3 L4 l1 l2 l3 l4

T1_0 = [[1, 0, 0, 0];
     	  [0, cos(q1), -sin(q1), L1*cos(q1)];
	      [0, sin(q1), cos(q1), L1*sin(q1)];
	      [0, 0, 0, 1]];

T2_1 = [[cos(q2), -sin(q2), 0, L2*sin(q2)];
        [sin(q2), cos(q2), 0, L2*cos(q2)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

T3_2 = [[cos(q3), -sin(q3), 0, L3*sin(q3)];
        [sin(q3), cos(q3), 0, L3*cos(q3)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

T4_3 = [[cos(q4), -sin(q4), 0, L4*sin(q4)];
        [sin(q4), cos(q4), 0, L4*cos(q4)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

com1 = [[0];
        [l1*cos(q1)];
        [l1*sin(q1)];
	      [1]];

com2 = [[l2*sin(q2)];
        [l2*cos(q2)];
        [0];
	      [1]];

com3 = [[l3*sin(q3)];
        [l3*cos(q3)];
        [0];
	      [1]];

com4 = [[l4*sin(q4)];
        [l4*cos(q4)];
        [0];
	      [1]];

xee = [[L4*sin(q4)]; # position of the end effector
       [L4*cos(q4)];
       [0];
       [1]];

#Tcom1 = simplify(T1_0*com1) # insufficient simplification
Tcom1 = [[0]; # simplified manually
         [l1*cos(2*q1) + L1*cos(q1)]; # trig identity
         [l1*sin(2*q1) + L1*sin(q1)]; # trig identity
         [1]];

J1 = jacobian(Tcom1, [q1, q2, q3, q4]);
J1(4,1) = 1;
J1(5,:) = 0;
J1(6,:) = 0;

T2_0 = simplify(T1_0*T2_1);

Tcom2 = simplify(T2_0*com2);

J2 = simplify(jacobian(Tcom2, [q1, q2, q3, q4]));
J2(4,1) = 1;
J2(5,:) = 0;
J2(6,2) = 1;

T3_0 = simplify(T3_2*T2_1*T1_0);
        
Tcom3 = simplify(T3_0*com3);

J3 = simplify(jacobian(Tcom3, [q1, q2, q3, q4]));
J3(4,1) = 1;
J3(5,:) = 0;
J3(6,2) = 1;
J3(6,3) = 1;

T4_0 = simplify(T4_3*T3_2*T2_1*T1_0);

Tcom4 = simplify(T4_0*com4);

J4 = simplify(jacobian(Tcom4, [q1, q2, q3, q4]));
J4(4,1) = 1;
J4(5,:) = 0;
J4(6,2) = 1;
J4(6,3) = 1;
J4(6,4) = 1;

Txee = simplify(T4_0*xee);

JEE = simplify(jacobian(Txee, [q1, q2, q3, q4]));
JEE(4,1) = 1;
JEE(5,:) = 0;
JEE(6,2) = 1;
JEE(6,3) = 1;
JEE(6,4) = 1;

for c=1:6
dlmwrite('j1.csv', char(J1(c,:)),'','-append')
end

for c=1:6
dlmwrite('j2.csv', char(J2(c,:)),'','-append')
end

for c=1:6
dlmwrite('j3.csv', char(J3(c,:)),'','-append')
end

for c=1:6
dlmwrite('j4.csv', char(J4(c,:)),'','-append')
end

for c=1:6
dlmwrite('jee.csv', char(JEE(c,:)),'','-append')
end


fprintf('T1_0:\n %s', latex(T1_0))
fprintf('\n')
fprintf('T2_1:\n %s', latex(T2_1))
fprintf('\n')
fprintf('T3_2:\n %s', latex(T3_2))
fprintf('\n')
fprintf('T4_1:\n %s', latex(T4_3))
fprintf('\n')

fprintf('com1:\n %s', latex(com1))
fprintf('\n')
fprintf('com2:\n %s', latex(com2))
fprintf('\n')
fprintf('com3:\n %s', latex(com3))
fprintf('\n')
fprintf('com4:\n %s', latex(com4))
fprintf('\n')
fprintf('com4:\n %s', latex(xee))
fprintf('\n')

%{
fprintf('J1:\n %s', latex(J1))
fprintf('\n')
fprintf('J2:\n %s', latex(J2))
fprintf('\n')
fprintf('J3:\n %s', latex(J3))
fprintf('\n')
fprintf('J4:\n %s', latex(J4))
fprintf('\n')
fprintf('JEE:\n %s', latex(JEE))
fprintf('\n')
}%

%writematrix(J1, 'j.csv') % not supported in Octave yet

%{
fprintf('J1:\n')
J1
fprintf('\n')

fprintf('J2:\n')
J2
fprintf('\n')

fprintf('J3:\n')
J3
fprintf('\n')

fprintf('J4:\n')
J4
fprintf('\n')

fprintf('JEE:\n')
JEE
}%