function u=dns_KS(u,T,dt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This subroutine embodies the main part
% of the code where the numerical integration
% is carried out. We use MATLAB's ode15s for the
% time integration.
%###################
% NOTE: The parameter dt does NOT determine the length
% of the time step. It only determines how often the
% solution is saved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% set up the time interval
nt=T/dt+1;
tspan=linspace(0,T,nt);

% integrate
options = odeset('RelTol',1e-10,'AbsTol',1e-10);
[~, F]=ode15s(@oderhs,tspan,u,options);
u = F;

% uncomment to save the solution to disc
% save('KS_u_all','u');