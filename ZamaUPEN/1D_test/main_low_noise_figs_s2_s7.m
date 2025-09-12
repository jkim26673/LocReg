% TEST PROBLEM 1: HEAT from Regularization Tool
% figures S2-S7
%
clear
close all
clc
%addpath ../regu
% tests
test='T1';%
%test='T2';%
%test='T3'
fprintf('\n test : %s \n',test)
% methods
i_multiL2 = 0;      % inexact method GupenMM
i_multiL2_nneg = 0; % inexact method GupenMM + non neg constraint
multiL2 = 0;        % UpenMM
multiL2_nneg = 1;   % UpenMM + non neg constraint
%
multi_fig=1;
switch(test)
    case 'T1'
% Create test problem
[A,b,xex] = heat(100);

% add gaussian noise
b_exact=A*xex;
noise_level = 0.01;
randn('seed',0);
noise = randn(size(b_exact));
noise = noise/norm(noise);
noise = noise_level*noise*norm(b_exact); noise_norm = norm(noise);
b = noise+b_exact;
%
beta_0 = 1e-5; tol_lam=5.e-2;
%
    case 'T2'
        noise_level = 0.01;
        [A,b,xex,noise_norm] =T2(100,noise_level);beta_0 = 1e-7; tol_lam=5.e-2;
    case 'T3'
        noise_level = 0.01;
        [A,b,xex,noise_norm] =T3(100,noise_level);beta_0 = 1e-7;tol_lam=1.e-2; 
end
%%
itermax=20;format short e
     fprintf('\n method           Rel Err             Res            lambda        iterations    ');

if i_multiL2 
    method='GUpenMM';
     [xpwL2,errML2,resML2,Q,INd_exit,ck,Lam]=UPenMMmL2_i(A,b,xex,noise_norm,beta_0,itermax,tol_lam);
     N=numel(xpwL2);
     
     
     %%
     figure; plot(ck,'LineWidth',2);  title([method '  - Regularization parameters']);grid on
     xticks([0 round(N/2) N]);

     figure; plot(1:numel(errML2),errML2,'o:',INd_exit,errML2(INd_exit),'*r','LineWidth',2);
     title([method ' - Relative Error']); xlabel('k');
     set(gca,"FontWeight","bold")
     figure; plot(resML2,'o:','LineWidth',2); hold on;
     plot(ones(size(resML2))*noise_norm,'r'); hold off; title(method); 
     xlabel('k'); legend('Residual norm','noise norm','FontSize',13);
     set(gca,"FontWeight","bold")
     figure; plot(Q,'o-','LineWidth',2); title([method 'Surrogate function']);
     set(gca,"FontWeight","bold")
     WWp=[errML2(INd_exit), resML2(INd_exit), Lam(INd_exit), INd_exit];
     fprintf('\n  %s        %e        %e   %e     %d  ',method,WWp);
      ERR_GUPenMM_L2=errML2;INd_gu=INd_exit;Res_GUPenMM_L2=resML2;
      Q_GUPenMM_L2=Q;
      X_GUPenMM_L2=xpwL2;

end
if i_multiL2_nneg 
    method='GUpenMM\_nn';
    [xpwL2_nn,errML2,resML2,Q,INd_exit,ck,Lam,it_pn]=UPenMMmL2nn_i(A,b,xex,noise_norm,beta_0,itermax,tol_lam);
    N=numel(xpwL2_nn);
    
     %
     figure; plot(xpwL2_nn,'LineWidth',2); hold on; 
     plot(xex,'r--','LineWidth',2); hold off; 
     legend('multipenalty','exact','FontSize',15)
     axis tight; xticks([0 round(N/2) N]);set(gca,"FontWeight","bold")
     %
     fig=figure;ax = axes('FontSize', 16); 
     plot(ck,'LineWidth',2);  xticks([0 round(N/2) N]);
     title([method '  - Regularization parameters']);set(gca,"FontWeight","bold")
     set(gca,"FontWeight","bold","FontSize",16)
     %
     figure; plot(1:numel(errML2),errML2,'o:',INd_exit,errML2(INd_exit),'*r','LineWidth',2);
     title([method ' - Relative Error']); xlabel('k');set(gca,"FontWeight","bold")
     figure; plot(resML2,'o:','LineWidth',2); hold on;
     plot(ones(size(resML2))*noise_norm,'r'); hold off; title(method); 
     xlabel('k'); legend('Residual norm','noise norm','FontSize',13);
     set(gca,"FontWeight","bold")
     figure; plot(Q,'o-','LineWidth',2); title([method 'Surrogate function']);
     xlabel('k');
     set(gca,"FontWeight","bold")
     
     WWr=[errML2(INd_exit), resML2(INd_exit), Lam(INd_exit), INd_exit,sum(it_pn(1:INd_exit))];
       fprintf('\n  %s    %e        %e   %e     %d   %d ',method,WWr);
      ERR_GUPenMM_L2nn=errML2;INd_gu_nn=INd_exit;Res_GUPenMM_L2nn=resML2;
      Q_GUPenMM_L2nn=Q;
      X_GUPenMM_L2nn=xpwL2_nn;
end
if multiL2 
    method='UPenMM';
     [xpwL2,errML2,resML2,Q,INd_exit,ck,Lam]=UPenMMmL2(A,b,xex,noise_norm,beta_0,itermax,tol_lam);
   
     
     %
     figure; plot(xpwL2,'LineWidth',2); hold on; plot(xex,'r--','LineWidth',2); hold off; legend('multipenalty','exact','FontSize',15)
     axis tight; xticks([0 round(N/2) N]);set(gca,"FontWeight","bold")
     %
%
     figure; plot(ck,'LineWidth',2);  xticks([0 round(N/2) N]);
     title([method '  - Regularization parameters']);set(gca,"FontWeight","bold")
     %
     figure; plot(1:numel(errML2),errML2,'o:',INd_exit,errML2(INd_exit),'*r','LineWidth',2);
     title([method ' - Relative Error']); xlabel('k');set(gca,"FontWeight","bold")
     figure; plot(resML2,'o:','LineWidth',2); hold on;
     plot(ones(size(resML2))*noise_norm,'r'); hold off; title(method); 
     xlabel('k'); legend('Residual norm','noise norm','FontSize',13);
     set(gca,"FontWeight","bold")
     figure; plot(Q,'o-','LineWidth',2); title([method 'Surrogate function']);
     xlabel('k');
     set(gca,"FontWeight","bold")
     %
     WWi=[errML2(INd_exit), resML2(INd_exit), Lam(INd_exit), INd_exit];
      fprintf('\n  %s         %e        %e   %e     %d  ',method,WWi);
     ERR_UPenMM_L2=errML2;Res_UPenMM_L2=resML2;
      Q_UPenMM_L2=Q;INd_u=INd_exit;
      X_UPenMM_L2=xpwL2;

end
if multiL2_nneg 
    method='UPenMM\_nn';
    [xpwL2_nn,errML2,resML2,Q,INd_exit,ck,Lam,it_pn]=UPenMMmL2nn(A,b,xex,noise_norm,beta_0,itermax,tol_lam);
    N=numel(xpwL2_nn);
     figure; plot(xpwL2_nn,'LineWidth',2); hold on; plot(xex,'r--','LineWidth',2); hold off; legend('multipenalty','exact','FontSize',15)
     axis tight; xticks([0 round(N/2) N]);set(gca,"FontWeight","bold")
    %
     figure; plot(ck,'LineWidth',2);  xticks([0 round(N/2) N]);
     title([method '  - Regularization parameters']);set(gca,"FontWeight","bold")
     %
     figure; plot(1:numel(errML2),errML2,'o:',INd_exit,errML2(INd_exit),'*r','LineWidth',2);
     title([method ' - Relative Error']); xlabel('k');set(gca,"FontWeight","bold")
     figure; plot(resML2,'o:','LineWidth',2); hold on;
     plot(ones(size(resML2))*noise_norm,'r'); hold off; title(method); 
     xlabel('k'); legend('Residual norm','noise norm','FontSize',13);
     set(gca,"FontWeight","bold")
     figure; plot(Q,'o-','LineWidth',2); title([method 'Surrogate function']);
     xlabel('k');
     set(gca,"FontWeight","bold")

     WW=[errML2(INd_exit), resML2(INd_exit), Lam(INd_exit), INd_exit,sum(it_pn(1:INd_exit))];
      fprintf('\n  %s     %e        %e   %e     %d   %d',method,WW);fprintf('\n ')
     ERR_UPenMM_L2nn=errML2;Res_UPenMM_L2nn=resML2;
      Q_UPenMM_L2nn=Q;INd_u_nn=INd_exit;
      X_UPenMM_L2nn=xpwL2_nn;

end
%%
if i_multiL2_nneg && multiL2_nneg
figure; 
plot(1:numel(ERR_GUPenMM_L2nn),ERR_GUPenMM_L2nn,'--b',INd_gu_nn,ERR_GUPenMM_L2nn(INd_gu_nn),'*r','LineWidth',2);hold on
plot(1:numel(ERR_UPenMM_L2nn),ERR_UPenMM_L2nn,'-.k',INd_u_nn,ERR_UPenMM_L2nn(INd_u_nn),'+r','LineWidth',2);hold on
xlabel('k');
legend('GUPenMM\_L2nn', 'exit','UPenMM\_L2nn','exit');grid on 
set(gca,"FontWeight","bold")
%
figure; 
plot(1:numel(Q_GUPenMM_L2nn),Q_GUPenMM_L2nn,'--b','LineWidth',2);hold on
plot(1:numel(Q_UPenMM_L2nn),Q_UPenMM_L2nn,'-.k','LineWidth',2);hold on
xlabel('k');
legend('Q GUPenMM\_L2nn',' Q UPenMM\_L2nn');grid on 
% set(gca,"FontWeight","bold")
set(gca,"FontWeight","bold")
figure; 
MG=numel(Res_GUPenMM_L2nn);MU=numel(Res_UPenMM_L2nn);
plot(1:MG,Res_GUPenMM_L2nn,'--b','LineWidth',2);hold on
plot(1:MU,Res_UPenMM_L2nn,'-.k','LineWidth',2);hold on
plot(ones(max(MG,MU),1)*noise_norm,'r');
xlabel('k');
legend('Res GUPenMM\_L2nn',' Res UPenMM\_L2nn','noise norm');grid on 
% set(gca,"FontWeight","bold")
set(gca,"FontWeight","bold")
figure; plot(X_GUPenMM_L2nn,'-b','LineWidth',2); hold on; 
         plot(X_UPenMM_L2nn,'-.k','LineWidth',2); hold on; 
        plot(xex,'r--','LineWidth',2); hold off; 
        legend('GUPenMM\_L2','UPenMM\_L2','exact','FontSize',15)
     axis tight; xticks([0 round(N/2) N]);
     set(gca,"FontWeight","bold")
    
elseif i_multiL2 && multiL2
    figure; 
    plot(1:numel(ERR_GUPenMM_L2),ERR_GUPenMM_L2,'--b',INd_gu,ERR_GUPenMM_L2(INd_gu),'*r','LineWidth',2);hold on
    plot(1:numel(ERR_UPenMM_L2),ERR_UPenMM_L2,'-.k',INd_u,ERR_UPenMM_L2(INd_u),'+r','LineWidth',2);hold on
    xlabel('k');
    legend('GUPenMM\_L2', 'exit','UPenMM\_L2','exit');grid on 
    set(gca,"FontWeight","bold")
%
    figure; 
    plot(1:numel(Q_GUPenMM_L2),Q_GUPenMM_L2,'--b','LineWidth',2);hold on
    plot(1:numel(Q_UPenMM_L2),Q_UPenMM_L2,'-.k','LineWidth',2);hold on
    xlabel('k');
    legend('Q GUPenMM\_L2',' Q UPenMM\_L2');grid on 
% set(gca,"FontWeight","bold")
    set(gca,"FontWeight","bold")
    figure; 
    MG=numel(Res_GUPenMM_L2);MU=numel(Res_UPenMM_L2);
    plot(1:MG,Res_GUPenMM_L2,'--b','LineWidth',2);hold on
    plot(1:MU,Res_UPenMM_L2,'-.k','LineWidth',2);hold on
    plot(ones(max(MG,MU),1)*noise_norm,'r');
    xlabel('k');
    legend('Res GUPenMM\_L2',' Res UPenMM\_L2','noise norm');grid on 
% set(gca,"FontWeight","bold")
    set(gca,"FontWeight","bold")
    figure; plot(X_GUPenMM_L2,'-b','LineWidth',2); hold on; 
     plot(X_UPenMM_L2,'-.k','LineWidth',2); hold on; 
     plot(xex,'r--','LineWidth',2); hold off; 
     legend('GUPenMM\_L2','UPenMM\_L2','exact','FontSize',15)
     axis tight; xticks([0 round(N/2) N]);
     set(gca,"FontWeight","bold")
    
end


