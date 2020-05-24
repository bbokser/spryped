clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms q0 q1 q2 q3

Rorg0 = [[1, 0,       0       ];
     	   [0, cos(q0), -sin(q0)];
	       [0, sin(q0), cos(q0)]];

R01 = [[cos(q1), -sin(q1), 0];
       [sin(q1), cos(q1),  0];
	     [0,       0,        1]];

R12 = [[cos(q2), -sin(q2), 0];
       [sin(q2), cos(q2),  0];
	     [0,       0,        1]];

R23 = [[cos(q3), -sin(q3), 0];
       [sin(q3), cos(q3),  0];
	     [0,       0,        1]];
   
Rorg1 = simplify(Rorg0*R01);    
Rorg2 = simplify(Rorg0*R01*R12);
Rorg3 = simplify(Rorg0*R01*R12*R23);

for c=1:3
dlmwrite('R.csv', char(Rorg3(c,:)),'','-append')
end

fprintf('Rorg0:\n %s', latex(Rorg0))
fprintf('\n')
fprintf('R01:\n %s', latex(R01))
fprintf('\n')
fprintf('R12:\n %s', latex(R12))
fprintf('\n')
fprintf('R23:\n %s', latex(R23))
fprintf('\n')

fprintf('Rorg3:\n %s', latex(Rorg3))
fprintf('\n')