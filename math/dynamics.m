clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms theta_x theta_y theta_z p_x p_y p_z omega_x omega_y omega_z pdot_x ...
     pdot_y pdot_z f1_x f1_y f1_z f2_x f2_y f2_z ...
     dt i_inv r1x r1y r1z r2x r2y r2z mass gravity...
     i11 i12 i13 i21 i22 i23 i31 i32 i33...
     rz11 rz12 rz13 rz21 rz22 rz23 rz31 rz32 rz33
     
% i_inv = sym('i_inv', [3, 3]);
% rz_phi = sym('rz_phi', [3, 3]);

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

g = [[0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [0];
     [gravity]];

% g = zeros(12, 1);
% g(12, :) = gravity;
A = [[1,  1,  1,  0,  0,  0,  rz11*dt,  rz12*dt,  rz13*dt,  0,  0,  0];
     [1,  1,  1,  0,  0,  0,  rz21*dt,  rz22*dt,  rz23*dt,  0,  0,  0];
     [1,  1,  1,  0,  0,  0,  rz31*dt,  rz32*dt,  rz33*dt,  0,  0,  0];
     [0,  0,  0,  1,  1,  1,  0,  0,  0,  dt,  dt,  dt];
     [0,  0,  0,  1,  1,  1,  0,  0,  0,  dt,  dt,  dt];
     [0,  0,  0,  1,  1,  1,  0,  0,  0,  dt,  dt,  dt];
     [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1]];

% A(1:3, 7:9) = rz_phi * dt;
% A(4:6, 10:12) = repmat(dt, 3, 3);
B = [[0, 0, 0, 0, 0, 0];
     [0, 0, 0, 0, 0, 0];
     [0, 0, 0, 0, 0, 0];
     [0, 0, 0, 0, 0, 0];
     [0, 0, 0, 0, 0, 0];
     [0, 0, 0, 0, 0, 0];
     [(i12*r1z - i13*r1y)*dt, (-i11*r1z + i13*r1x)*dt, (i11*r1y - i12*r1x)*dt, ... 
        (i12*r2z - i13*r2y)*dt, (-i11*r2z + i13*r2x)*dt, (i11*r2y - i12*r2x)*dt];
     [(i22*r1z - i23*r1y)*dt, (-i21*r1z + i23*r1x)*dt, (i21*r1y - i22*r1x)*dt, ... 
        (i22*r2z - i23*r2y)*dt, (-i21*r2z + i23*r2x)*dt, (i21*r2y - i22*r2x)*dt];
     [(i32*r1z - i33*r1y)*dt, (-i31*r1z + i33*r1x)*dt, (i31*r1y - i32*r1x)*dt, ... 
        (i32*r2z - i33*r2y)*dt, (-i31*r2z + i33*r2x)*dt, (i31*r2y - i32*r2x)*dt];
     [dt/mass, dt/mass, dt/mass, dt/mass, dt/mass, dt/mass];
     [dt/mass, dt/mass, dt/mass, dt/mass, dt/mass, dt/mass];
     [dt/mass, dt/mass, dt/mass, dt/mass, dt/mass, dt/mass]];

x_next = mtimes(A, x) + mtimes(B, f) + g;

for c=1:12
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
