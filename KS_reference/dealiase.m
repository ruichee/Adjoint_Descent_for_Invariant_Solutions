function ff = dealiase(ff)
% 2/3 dealiasing
global kx

k=abs(kx);
k0=1/3*max(k);
ind=(k>=k0);
ff(ind)=0.0;