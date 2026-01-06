function Lrhs_u=rhs_adjKS(u,v)
% Thsi subroutine evaluates the right hand side G(u) of 
% the adjoint-KS equation. Note that v=F(u)
global kx

% to Fourier space
fv=fft(v); fv=dealiase(fv);

% compute the mixed term -u\partial_x v
fvx = complex(0,kx).*fv;
vx  = ifft(fvx,'symmetric');
rhs = -u.*vx;

% take to Fourier space
frhs = fft(rhs);

% add linear terms
frhs = frhs - (kx.^2-kx.^4).*fv;

% dealiase
frhs = dealiase(frhs);

% mean-flow=0
ind=(abs(kx)==0); frhs(ind)=0.0;

% take to physical space
Lrhs_u = ifft(frhs,'symmetric');