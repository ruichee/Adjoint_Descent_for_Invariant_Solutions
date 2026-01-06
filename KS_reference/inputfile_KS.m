%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Terms of use:
% This set of MATLAB codes illustrate the application of the 
% adjoint method for finding equilibrium solutions of the 
% Navier-Stokes equaitons. 
% The application is illustrated here on a unidirectional 
% Kuramoto-Sivashinsky (KS) equation. 
% This code may be used with appropriate citation to 
% "Farazmand M., An adjoint-based approach for finding invariant solutions
% of Navier-Stokes equations, 2015 arXiv:1508.06363"
% This code has been tested on MATLAB R2015b.
% ##########################################
% Variables: 
% There are 5 global variables.
% n  (integer scalar)  = number of collocation points
% kx (double vector)   = wave vector
% x  (double scalar)   = coordinates of the collocation points
% L  (double scalar)   = length of the domain
% EQ (Boolean scalalr) = determines whether KS eq. (EQ=0) 
%                        or the adjoint KS eq. (EQ=1) should be solved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
global n kx x L EQ

%% set up parameters
n=128; % number of collocation points
L=22;  % Length of the domain
EQ=1;  % 0=KS equaiton, 1=adjoint KS equation

%% generate collocation points (x) and the wave vector (kx)
[kx, x]=gvars(n,L);

%% Integration time (T) and the saving interval (dt)
T = 200; dt = 1;

%% set up initial condition u0
% m=positive integer
% change m to converge to different equilibria
m=2;
u0=2*sin(m*2*pi*x/L);

%% main body: integrate the equation
tic
u=dns_KS(u0,T,dt);
toc

%% Post-processing: computing the residue, plotting, etc.
res=plot_data(T,dt,u,u0);