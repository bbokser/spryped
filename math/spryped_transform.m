clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms q0 q1 q2 q3 L0 L1 L2 L3 l0 l1 l2 l3

Torg0 = [[1, 0, 0, 0];
     	   [0, cos(q0), -sin(q0), L0*cos(q0)];
	       [0, sin(q0), cos(q0), L0*sin(q0)];
	       [0, 0, 0, 1]];

T01 = [[cos(q1), -sin(q1), 0, -L1*sin(q1)];
        [sin(q1), cos(q1), 0, L1*cos(q1)];
	      [0, 0, 1, 0];
	      [0, 0, 0, 1]];

T12 = [[cos(q2), -sin(q2), 0, -L2*sin(q2)];
       [sin(q2), cos(q2), 0, L2*cos(q2)];
	     [0, 0, 1, 0];
	     [0, 0, 0, 1]];

T23 = [[cos(q3), -sin(q3), 0, -L3*sin(q3)];
       [sin(q3), cos(q3), 0, L3*cos(q3)];
	     [0, 0, 1, 0];
	     [0, 0, 0, 1]];

Torg1 = simplify(Torg0*T01);    
Torg2 = simplify(Torg0*T01*T12);
Torg3 = simplify(Torg0*T01*T12*T23);


for c=1:4
dlmwrite('Torg3.csv', char(Torg3(c,:)),'','-append')
end

fprintf('Torg0:\n %s', latex(Torg0))
fprintf('\n')
fprintf('T01:\n %s', latex(T01))
fprintf('\n')
fprintf('T12:\n %s', latex(T12))
fprintf('\n')
fprintf('T23:\n %s', latex(T23))
fprintf('\n')