clear all
clc

pkg load symbolic
% warning: delete old csvs before running this or it will just append to them

syms r_z r_y r_x

r = [[0, -r_z, r_y];
     [r_z, 0, -r_x];
     [-r_y, r_x, 0]];
     
i_inv = sym('i', [3, 3]);

mtimes(i_inv, r)