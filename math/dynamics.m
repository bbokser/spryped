clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms theta_x theta_y theta_z p_x p_y p_z omega_x omega_y omega_z pdot_x ...
     pdot_y pdot_z f1_x f1_y f1_z f2_x f2_y f2_z ...
     rz_phi dt i_inv r1 r2 mass gravity

x = [[theta_x];  % states
     [theta_y];
     [theta_z];
     [p_x];
     [p_y];
     [p_z];
     [omega_x];
     [omega_y];
     [omega_z];
     [pdot_x];
     [pdot_y];
     [pdot_z]];

f = [[f1_x];  % controls
     [f1_y];
     [f1_z];
     [f2_x];
     [f2_y];
     [f2_z]];

g = zeros(12, 1);
g(12) = gravity;

A = zeros(12, 12);
A(1:3, 1:3) = ones(3, 3);
A(4:6, 4:6) = ones(3, 3);
A(7:9, 7:9) = ones(3, 3);
A(10:12, 10:12) = ones(3, 3);

A(1:3, 7:9) = rz_phi * dt;
A(4:6, 10:12) = dt;

B = zeros(12, 6);
B(7:9, 1:3) = i_inv * r1 * dt;
B(7:9, 4:6) = i_inv * r2 * dt;
B(10:12, 1:3) = dt / mass;
B(10:12, 4:6) = dt / mass;

x_next = mtimes(A, x) + mtimes(B, f) + g;

for c=1:4
dlmwrite('dynamics.csv', char(x_next(c,:)),'','-append')
end

fprintf('x = \n %s', latex(x))
fprintf('\n')
fprintf('f = \n %s', latex(f))
fprintf('\n')
fprintf('g = \n %s', latex(g))
fprintf('\n')
fprintf('A = \n %s', latex(A))
fprintf('\n')
fprintf('B = \n %s', latex(B))
fprintf('\n')
fprintf('x(k+1) = \n %s', latex(x_next))