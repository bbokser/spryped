clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms q0 q1 q2 q3 L0 L1 L2 L3 l0 l1 l2 l3

Torg0 = [[1, 0, 0, 0];
     	  [0, cos(q0), -sin(q0), L0*cos(q0)];
	      [0, sin(q0), cos(q0), L0*sin(q0)];
	      [0, 0, 0, 1]];

T01 = [[cos(q1), -sin(q1), 0, L1*sin(q1)];
        [sin(q1), cos(q1), 0, L1*cos(q1)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

T12 = [[cos(q2), -sin(q2), 0, L2*sin(q2)];
        [sin(q2), cos(q2), 0, L2*cos(q2)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

T23 = [[cos(q3), -sin(q3), 0, L3*sin(q3)];
        [sin(q3), cos(q3), 0, L3*cos(q3)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

com0 = [[0];
        [l0];
        [0];
	      [1]];
        
com1 = [[0];
        [l1]; # l1*cos(q1)
        [0]; # l1*sin(q1)
	      [1]];

com2 = [[0]; # l2*sin(q2)
        [l2]; # l2*cos(q2)
        [0];
	      [1]];

com3 = [[0]; # l3*sin(q3)
        [l3]; # l3*cos(q3)
        [0];
	      [1]];
        
# position of the end effector
xee = [[0];
       [L3];
       [0];
       [1]];
   
Torg1 = simplify(Torg0*T01);    
Torg2 = simplify(Torg0*T01*T12);
Torg3 = simplify(Torg0*T23*T12*T01);

Tcom0 = simplify(Torg0*com0);
Tcom1 = simplify(Torg1*com1);
Tcom2 = simplify(Torg2*com2);
Tcom3 = simplify(Torg3*com3);
Txee = simplify(Torg3*xee);

J0 = simplify(jacobian(Tcom0, [q0, q1, q2, q3]));
J0(4,1) = 1;
J0(5,:) = 0;
J0(6,:) = 0;

J1 = simplify(jacobian(Tcom1, [q0, q1, q2, q3]));
J1(4,1) = 1;
J1(5,:) = 0;
J1(6,2) = 1;

J2 = simplify(jacobian(Tcom2, [q0, q1, q2, q3]));
J2(4,1) = 1;
J2(5,:) = 0;
J2(6,2) = 1;
J2(6,3) = 1;

J3 = simplify(jacobian(Tcom3, [q0, q1, q2, q3]));
J3(4,1) = 1;
J3(5,:) = 0;
J3(6,2) = 1;
J3(6,3) = 1;
J3(6,4) = 1;

JEE = simplify(jacobian(Txee, [q0, q1, q2, q3]));
JEE(4,1) = 1;
JEE(5,:) = 0;
JEE(6,2) = 1;
JEE(6,3) = 1;
JEE(6,4) = 1;

for c=1:6
dlmwrite('j0.csv', char(J0(c,:)),'','-append')
end

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
dlmwrite('jee.csv', char(JEE(c,:)),'','-append')
end



fprintf('Torg0:\n %s', latex(Torg0))
fprintf('\n')
fprintf('T01:\n %s', latex(T01))
fprintf('\n')
fprintf('T12:\n %s', latex(T12))
fprintf('\n')
fprintf('T23:\n %s', latex(T23))
fprintf('\n')

fprintf('com0:\n %s', latex(com0))
fprintf('\n')
fprintf('com1:\n %s', latex(com1))
fprintf('\n')
fprintf('com2:\n %s', latex(com2))
fprintf('\n')
fprintf('com3:\n %s', latex(com3))
fprintf('\n')
fprintf('xee:\n %s', latex(xee))
fprintf('\n')
%}

fprintf('J0:\n %s', latex(J0))
fprintf('\n')
fprintf('J1:\n %s', latex(J1))
fprintf('\n')
fprintf('J2:\n %s', latex(J2))
fprintf('\n')
fprintf('J3:\n %s', latex(J3))
fprintf('\n')
fprintf('JEE:\n %s', latex(JEE))
fprintf('\n')

%writematrix(J1, 'j.csv') % not supported in Octave yet

%{
fprintf('J0:\n')
J0
fprintf('\n')

fprintf('J1:\n')
J1
fprintf('\n')

fprintf('J2:\n')
J2
fprintf('\n')

fprintf('J3:\n')
J3
fprintf('\n')

fprintf('JEE:\n')
JEE
%}