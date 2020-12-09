clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms theta_x theta_y theta_z p_x p_y p_z omega_x omega_y omega_z pdot_x ...
     pdot_y pdot_z f1_x f1_y f1_z f2_x f2_y f2_z ...
     dt i_inv r1x r1y r1z r2x r2y r2z mass g...
     i11 i12 i13 i21 i22 i23 i31 i32 i33...
     rzt11 rzt12 rzt13 rzt21 rzt22 rzt23 rzt31 rzt32 rzt33...
     s_phi_1 s_phi_2 dt2 m
     
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

f1 = [[f1_x];  % controls
     [f1_y];
     [f1_z]];

f2 = [[f2_x];
     [f2_y];
     [f2_z]];

A = [[1,  0,  0,  0,  0,  0,  dt,  0,  0,  0,  0,  0];
     [0,  1,  0,  0,  0,  0,  0,  dt,  0,  0,  0,  0];
     [0,  0,  1,  0,  0,  0,  0,  0,  dt,  0,  0,  0];
     [0,  0,  0,  1,  0,  0,  0,  0,  0,  dt,  0,  0];
     [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  dt,  0];
     [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  dt];
     [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0];
     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]];

% dt2 = 0.5*(dt^2)  % but this has floating point so better just make it sym
     
B = [[m*dt2, 0, 0, 0, 0, 0]
     [0, m*dt2, 0, 0, 0, 0]
     [0, 0, m*dt2, 0, 0, 0]
     [0, 0, 0, i11*dt2, i12*dt2, i13*dt2];
     [0, 0, 0, i21*dt2, i22*dt2, i23*dt2];
     [0, 0, 0, i21*dt2, i22*dt2, i23*dt2];
     [m*dt, 0, 0, 0, 0, 0]
     [0, m*dt, 0, 0, 0, 0]
     [0, 0, m*dt, 0, 0, 0]
     [0, 0, 0, i11*dt, i12*dt, i13*dt];
     [0, 0, 0, i21*dt, i22*dt, i23*dt];
     [0, 0, 0, i21*dt, i22*dt, i23*dt]];   
     
d = [[0];
     [0];
     [dt2*g];
     [0];
     [0];
     [0]
     [0];
     [0];
     [dt*g];
     [0];
     [0];
     [0]];
     
h_1 = [[1, 0, 0];
       [0, 1, 0];
       [0, 0, 1];
       [(rzt12*r1z - rzt13*r1y)*dt, (-rzt11*r1z + rzt13*r1x)*dt, (rzt11*r1y - rzt12*r1x)*dt];
       [(rzt22*r1z - rzt23*r1y)*dt, (-rzt21*r1z + rzt23*r1x)*dt, (rzt21*r1y - rzt22*r1x)*dt];
       [(rzt32*r1z - rzt33*r1y)*dt, (-rzt31*r1z + rzt33*r1x)*dt, (rzt31*r1y - rzt32*r1x)*dt]];

h_2 = [[1, 0, 0];
       [0, 1, 0];
       [0, 0, 1];
       [(rzt12*r2z - rzt13*r2y)*dt, (-rzt11*r2z + rzt13*r2x)*dt, (rzt11*r2y - rzt12*r2x)*dt];
       [(rzt22*r2z - rzt23*r2y)*dt, (-rzt21*r2z + rzt23*r2x)*dt, (rzt21*r2y - rzt22*r2x)*dt];
       [(rzt32*r2z - rzt33*r2y)*dt, (-rzt31*r2z + rzt33*r2x)*dt, (rzt31*r2y - rzt32*r2x)*dt]];

% forces and torques acting on the CoM
h = s_phi_1*mtimes(h_1, f1) + s_phi_2*mtimes(h_2, f2);  

x_next = mtimes(A, x) + mtimes(B, h) + d;

for c=1:12
dlmwrite('dynamics_rpc.csv', char(x_next(c,:)),'','-append')
end

fprintf('x = \n %s', latex(x))
fprintf('\n')
fprintf('f = \n %s', latex(h))
fprintf('\n')
fprintf('A = \n %s', latex(A))
fprintf('\n')
fprintf('B = \n %s', latex(B))
fprintf('\n')
fprintf('x(k+1) = \n %s', latex(x_next))
