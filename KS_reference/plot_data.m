function err=plot_data(T,dt,u,u0)
global x L

figure;
nt=T/dt+1;
tspan=linspace(0,T,nt);
err = zeros(nt,1);
for j=1:nt
    subplot(2,1,1)
    plot(x,u(j,:),'k'); axis([0 L 1.1*min(u(:)) 1.1*max(u(:))]);
    hold on
    plot(x,u0,'--r')
    title(['$\tau=$' num2str((j-1)*dt)],'interpreter','latex','fontsize',22);
    xlabel('$x$','interpreter','latex','fontsize',22);
    ylabel('$u(x,\tau)$','interpreter','latex','fontsize',22);
    hold off
    w = rhs_KS(u(j,:)');
    err(j) = sum(w.^2);
    subplot(2,1,2)
    semilogy(tspan(1:j),err(1:j));
    axis([0 T 1e-16 1e3])
    xlabel('$\tau$','interpreter','latex','fontsize',22);
    ylabel('$L^2$ residue','interpreter','latex','fontsize',22);
    pause(1e-9)
end