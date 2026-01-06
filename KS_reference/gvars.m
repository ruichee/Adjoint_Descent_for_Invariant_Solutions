function [kx, x]=gvars(n,L)
% defines the wave vector (kx)
% and the coordinates for the collocation points (x)

dx=L/n;
x=linspace(0,L-dx,n)';
kx = [0:n/2-1 0 -n/2+1:-1]'*(2*pi/L);