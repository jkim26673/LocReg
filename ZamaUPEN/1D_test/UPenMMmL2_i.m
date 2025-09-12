function [xpwL2,errML2,resML2,Q,INd_exit,ck,Lam,time]= UPenMMmL2_i(A,b,xex,noise_norm,beta_0,Kmax,tol_lam)
%
%  UPenMMmL2 Inexact
%
% Set up parameters
[N,M]=size(A);
L2 = get_l(M+2,2); 
L2(:,1)=[];
L2(:,end)=[];

beta_c = 1;


% initial iterate
% [x0,iter] = grad_proj_noreg1d(A,b,0, zeros(size(b)), 1e-5, 50000); x = x0;
x = b;

Rsqrd = norm(A*x-b)^2;  Rsqrd = Rsqrd/N;
c = L2*x; c=LocMax(abs(c));
ck = Rsqrd./(beta_0+beta_c*c.^2); ck_old = ck+10; 
Qk = (norm(A*x-b)^2 + x'*L2'*diag(ck)*L2*x + beta_0*sum(ck)) /  (prod(ck.^(1/(2*N))));


k = 0;
Tol_outer = 0e-4; 
flag=1;
tic
while k<=Kmax && norm(ck-ck_old)>norm(ck)*Tol_outer
   
    k = k+1; 
    ck_old = ck;
    Qk_old = Qk;
      
    x = (A'*A+L2'*diag(ck)*L2)\(A'*b); 
    
    % update regularization parameters
    Rsqrd = norm(A*x-b)^2;  Rsqrd = Rsqrd/N;
    c = L2*x; c=LocMax(abs(c));
    ck = Rsqrd./(beta_0+beta_c*c.^2); ck_inex = ck;
    Qk = (norm(A*x-b)^2 + x'*L2'*diag(ck)*L2*x + beta_0*sum(ck)) /  (prod(ck.^(1/(2*N))));
    
    j=0; jmax = 20;      
    c = L2*x; ck_exact = Rsqrd./(beta_0+beta_c*c.^2); 

    while Qk>=Qk_old && j<=jmax
        j = j+1;
 %       disp(['Non decreasing at k=' num2str(k) 'j=' num2str(j)])
 %         ck = 0.9*ck+0.1*ck_exact;
        ck = 0.9^j*ck_inex+(1-0.9^j)*ck_exact;
        Qk = (norm(A*x-b)^2 + x'*L2'*diag(ck)*L2*x + beta_0*sum(ck)) /  (prod(ck.^(1/(2*N))));
    end
    
    errML2(k) = norm(xex-x)/norm(xex);
    resML2(k) = norm(A*x-b);
    Q(k) = Qk;
    Lam(k)=norm(ck-ck_old)/norm(ck);
     if Lam(k)<tol_lam && flag
         INd_exit=k;
         sol_exit=x;flag=0;
     end
end
time=toc;
xpwL2 = sol_exit;

% fprintf('multi    & %e & %d \n',errML2(end),k);
% figure; plot(xpwL2,'LineWidth',2); hold on; plot(xex,'r--','LineWidth',2); hold off; legend('multipenalty','exact','FontSize',15)
% axis tight; xticks([0 50 100])
% figure; plot(ck,'LineWidth',2);  title('Multipenalty - Regularization parameters');
% figure; plot(errML2,'o:','LineWidth',2); title('Multipenalty - Relative Error'); xlabel('iter')
% figure; plot(resML2,'o:','LineWidth',2); hold on;
% plot(ones(size(resML2))*noise_norm,'r'); hold off; title('Multipenalty'); xlabel('iter'); legend('Residual norm','noise norm','FontSize',13);
% figure; plot(Q,'o-','LineWidth',2); title('Surrogate function');
[mm,mj]=max(diff(Q)); %disp([mm,mj]) % check decrease
end
