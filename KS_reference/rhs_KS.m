function rhs_u=rhs_KS(u)
% This subroutine evaluates the right hand side F(u) of 
% the KS equation. 

global kx

% to Fourier space
fu=fft(u); fu=dealiase(fu);

% compute the nonlinear term -u\partial_x u
usqr = u.^2;
fusqr = fft(usqr);
fusqr_x = complex(0,kx).*fusqr;
usqr_x  = ifft(fusqr_x,'symmetric');
rhs=-.5*usqr_x;

% take to Fourier domain
frhs = fft(rhs);

% add the linear terms
frhs = frhs + (kx.^2-kx.^4).*fu;

% dealiase
frhs = dealiase(frhs);

% mean-flow=0
ind=(abs(kx)==0); frhs(ind)=0.0;

% back to physical space
rhs_u = ifft(frhs,'symmetric');