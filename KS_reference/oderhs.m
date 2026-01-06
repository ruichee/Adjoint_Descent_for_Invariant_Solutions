function dy=oderhs(t,y)
global EQ
% t
u=y(:);
Nu = rhs_KS(u);
if EQ==0
    dy = Nu;
elseif EQ==1
    Lu = rhs_adjKS(u,Nu);
    dy = Lu;
end