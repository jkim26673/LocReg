%########################################################################################
% Main for figures 1(a) (b)  and 2 (UpenMM)
%########################################################################################
%########################################################################################
close all
clear all
clc
%
% Folder input data 
%
addpath funzioni/
Data_folder='Synth_data_2pk_1/';
%
% par and data read
%
script_data_setup
%%
% Parameters setting
%
if(FL_UPEN2D==2)  % upenfista    
X_true=zeros(nx,ny);
%=======================================================================
% 
  X_true=dlmread([Data_folder 'True_2Dmap.dat']); 
  normexact = norm((Kc*X_true*Kr'-s)/Amp_scale,'fro')^2; 
  fprintf('normexact scal=%e \n',normexact)
%=======================================================================
%
% fista
%
par.Amp_scale=Amp_scale;
par.fista_maxiter = 500000;
par.fista_tol     = 1.e-7;
par.fista_crit    = 1;
par.fista_verb_fista = 0;
par.scale_fact  = scale_fact;
par.true        = X_true;
par.upen_tol_x    = 1.e-4;%
par.upen_tol_res    = 1.e-6;% f
par.upen.beta_p =0;
 [x,LAMBDA,hist] = upenMM_2D(Kc,Kr,s,par);Testo='UpenMM';
%%
     fprintf('%s   \n',Testo)
     Err=hist.err(end);
     Res = norm((Kc*x*Kr'-s)/Amp_scale,'fro')^2;IT_int=sum(hist.it_int);
 % 
 %
 fprintf('\n \n res=%e it_fista %d time=%e s. \n\n',hist.res(end),IT_int,sum(hist.tempi));
grafici_vb(x,T1,T2,0, FL_typeKernel, 80, Testo);
set(gca,"FontWeight","bold")
%
figure; plot(hist.err,LineWidth=2);xlabel('k');ylabel('Relative Error');
set(gca,"FontWeight","bold")
end
%
%%
figure;surf(log10(hist.ck));title('log(\Lambda)');grid on% Check vars

%################################## END MAIN #########################################################

function [x,ck,hist]=upenMM_2D(Kc,Kr,s,par)
%
% 
%
scale_fact = par.scale_fact;
TOL_x        = par.upen_tol_x;
TOL_res        = par.upen_tol_res;

Kmax       = par.upen.iter;

if isfield(par,'Amp_scale')
      Amp_scale=par.Amp_scale;
else
      Amp_scale=1;
end
%--------------------------------------------------------------------------
% 
tic
 if (par.weightB==1)
   B=par.upen.B;
   Kr=B*Kr;
   s=eye(size(s,1))*s*B;  % (fz) 17/7/17 Piu' efficiente
 end
  [Uc,Sc,Vc]=svd(Kc); 
  [Ur,Sr,Vr]=svd(Kr); 
  Svals = Sc*Sr'; L = max(max(Svals.^2));
 % filtro
 if par.svd.svd
   %--------------------------------------------------------------------------
   soglia=par.svd.soglia;
   Sc=diag(Sc);hist.Sc=Sc;
   Sr=diag(Sr);hist.Sr=Sr;
   %
   if soglia < min(Sc)
     nc=length(Sc);
    else
     nc=find(Sc<=soglia,1);
   end
   if soglia < min(Sr)
     nr=length(Sr);
    else
     nr=find(Sr<=soglia,1);
   end
   %
   %--------------------------------------------------------------------------
   Uc=Uc(:,1:nc); Vc=Vc(:,1:nc); Sc=Sc(1:nc);
   Ur=Ur(:,1:nr); Vr=Vr(:,1:nr); Sr=Sr(1:nr);
   s=Uc'*s*Ur;
   Kc=Uc'*Kc;
   Kr=Ur'*Kr;
 end
%--------------------------------------------------------------------------
[N_T1,N_T2]=size(s); NN = (N_T1*N_T2);
nx = size(Kc,2); ny = size(Kr,2);

% L2 and L1 differential Matrices
[L1nx,L1ny,L2] = get_diff(nx,ny,'cost'); 

% Set up parameters
beta_00 = par.upen.beta00;
beta_p  = beta_00*par.upen.beta_p; 
beta_c  = beta_00*par.upen.beta_c;
beta_0  = beta_00*par.upen.beta0;
% Initialize
[x0, iter]=grad_proj_noreg(Kc,Kr,s,0, zeros(nx,ny), par); 
err0 = norm(par.true(:)-x0(:))/norm(par.true(:));
x = x0;
Rsqrd = norm((Kc*x*Kr'-s),'fro')^2; 
% 
c = reshape(L2*(x(:)),nx,ny); psi=c+beta_0;
% 
px = L1nx*(x(:)); %
py = L1ny*(x(:)); %
v = sqrt(px.^2+py.^2);
p = reshape(v,nx,ny);
beta_0 = beta_0*Rsqrd/(NN); 
% L2 regularization parameters 
ck = (Rsqrd)./((NN+1)*(beta_0+beta_p*p.^2+beta_c*c.^2)); 
%  L1 regularization parameter
lambda=1*Rsqrd/norm((NN+1)*x(:),1);
% Parameters fo inner fista solvers
maxiter = par.fista_maxiter;
tol = par.fista_tol;
crit = par.fista_crit;
verb_fista = par.fista_verb_fista;  
 
% stopping criteria
continua = 1; i=1;cond_res=1;X_init=zeros(size(x0));
par_fista.Thresh=10^(-7);tempo0=toc;
Q1=[];

%-----
% surrogate test
%-------
     pars_reg=[ck(:); lambda];
     format short e
     %
     %  senza scalatura
     %
     const=1/par.upen.beta0;
     % 
     [min(pars_reg) max(pars_reg) const]
     % 
     [sv, iv]=sort(pars_reg,'descend');
     % 
     indx=find(sv<const);
     P=prod(sv(indx));  % 
     Num_const=indx(1)-1; %
     Prod=1;flag=1;v=sv(indx); 
     %
     for ii=numel(v):-1:1
         val=v(ii);
         pp=Prod*val;
         if pp < realmax && flag
             Prod=pp;
         else
             ind_excl=Num_const+indx(ii);% salvo l'inizio degli esclusi
             flag=0;
         end
     end
     P_e=1;
     %fprintf(' Num_const =%d ind_excl=%d \n',Num_const,ind_excl)
     for ii=Num_const+1:ind_excl-Num_const+1
            P_e=P_e*sv(ii);
     end
     %[Num_const*const P_e Prod]
      vv=(Kc*x*Kr'-s);
      psi=[psi(:); norm(x(:),1)];
      NUM=norm(vv)+pars_reg'*psi;
     
     
     sv_old=sv;
     NUM_old=NUM;
      NUM=norm(vv)+pars_reg'*psi;
      DEN=P_e;
      vq1=(NUM^2)/(DEN^(1/NN));
      
      Q1=[Q1 vq1];
 
%
 
while continua

    xold = x; 
    par_fista.tau=lambda;
     if i>2
         par_fista.Thresh=max(par_fista.Thresh/100, 10^(-10));
     end
    [x, par_out] = my_Thresh_fista_l1_upen(X_init,s,Kr,Kc,ck,L2,par_fista,L,crit, tol, maxiter, par.true, 0,Amp_scale);
    X_init=x;
    res_int{i}=par_out.nres;
    tempi(i)=par_out.times;
    mse_int{i}=par_out.mses;erel_int{i}=par_out.erel_int;
    fobj{i}=par_out.objective;
    it_inter(i)=numel(par_out.nres);
    crit_int{i}=par_out.crit_out;
    x = reshape(x,nx,ny);
    % 
    c = reshape(L2*(x(:)),nx,ny); c_true=c;psi=c_true+beta_0;
    c = ordfilt2(abs(c),9,ones(3));
    
    % 
    px = L1nx*(x(:)); 
    py = L1ny*(x(:)); 
    v = sqrt(px.^2+py.^2);
    p = reshape(v,nx,ny);
    p = ordfilt2(p,9,ones(3));

        % output
    res(i) = norm((Kc*x*Kr'-s)/Amp_scale,'fro');
    if ~isempty(par.true)
        err(i) = norm(par.true(:)-x(:))/norm(par.true(:));
    end
    % fprintf('  Err=%e Res=%e it_fista=%d \n',err(i),res(i),it_inter(i))
    cond_err=norm(x-xold,'fro')/norm(x,'fro')>TOL_x;
     
    continua =  cond_err && i<Kmax;
   
        Rsqrd = norm((Kc*x*Kr'-s),'fro')^2; 
        ck =Rsqrd./(NN*(beta_0+beta_p*p.^2+beta_c*c.^2)); 
        ck_true =Rsqrd./((NN+1)*(beta_0+beta_c*c_true.^2)); 
         lambda=Rsqrd/norm((NN+1)*x(:),1);
         lambda_inner(i)=lambda;
    
     %-----
     %  test  surrogata
     %-------
     pars_reg=[ck(:); lambda];pars_true=[ck_true(:); lambda/(NN+1)];
     
     const=max(pars_reg);
     [sv, iv]=sort(pars_reg,'descend');
     
     indx=find(sv<const);
     Num_const=indx(1)-1; 
     Prod=1;v=sv(indx); v_old=sv_old(indx);
     Prod_old=1;flag=1;
      
     for ii=numel(v):-1:1
         val=v(ii);
         pp=Prod*val;
         if pp < realmax && flag
             Prod=pp;
             Prod_old=min(Prod_old*v_old(ii),realmax);
         else
             ind_excl=min(Num_const+indx(ii),numel(sv));
             flag=0;
         end
     end
    
     P_e=1;Pe_old=1;
     for ii=Num_const+1:ind_excl-Num_const
            P_e=P_e*sv(ii);
            Pe_old=Pe_old*sv_old(ii);
     end
     vv=(Kc*x*Kr'-s);
      psi=[psi(:); norm(x(:),1)];
      NUM=norm(vv)+pars_reg'*psi;
      DEN=P_e;DEN_old=Pe_old;
      vq1=(NUM^2)/(DEN^(1/NN));
      
      cond_Q1 = vq1 < (NUM_old^2)/(DEN_old^(1/NN));
       [(NUM^2)/(DEN^(1/NN)), (NUM_old^2)/(DEN_old^(1/NN))];
     if cond_Q1
      Q1=[Q1; vq1];
      sv_old=sv;
          i = i+1;
     else
       pars_tilde=pars_reg;j=1;
       pars_reg=0.9^j*pars_tilde+0.1^j*pars_true;
       check_Q;
       Q1=[Q1; vq1];
       sv_old=sv;
       ck=reshape(pars_reg(1:end-1),nx,ny);lambda=pars_reg(end);
     end
end
figure;plot(log(Q1),LineWidth=2);xlabel('k');ylabel('log(Q_k)');
set(gca,"FontWeight","bold");


hist.ssize=[N_T1,N_T2];
hist.err = [err0 err];
hist.res = res;
hist.res_int=res_int;
hist.obj= fobj;
hist.tempi=[tempo0 tempi];
hist.mse=mse_int;hist.erel_int=erel_int;
hist.it_int=it_inter;
hist.lambda_inner=lambda_inner;
hist.it_cg=iter;
hist.crit_int=crit_int;
hist.ck=ck;
errupen=[err0 err];
Q_upen=Q1;
end

%
% upen fista modificato con soglia di uscita
%
function [x, par_out] = ...
    my_Thresh_fista_l1_upen(x,b,Kr,Kc,ck,L2,par,L,stopcriterion, tolerance, maxiters, true, verbose,Amp_scale)

tau=par.tau;
Thresh=par.Thresh; % siglia di arresto
[~,nx]=size(Kr); [~,ny]=size(Kc);
b=[b(:); zeros(nx*ny,1)];   
true=true(:);
T_reg=sqrt(ck);
 x=x(:);
y = x;
t = 1;
%
% Correzione L con max eigenvalues
%
L_A=L;
%Lc=max(max(abs(ck)));
Lc=max(max(abs(T_reg)));
L=L_A+Lc;
%times(1) = 0;
%t0 = cputime;
tic
objective(1) = 0.5*norm(A(x,Kr,Kc,L2,T_reg)-b)^2 + tau*sum(abs(x) );
mses(1) = norm((x-true))^2/numel(true);erel_int(1)=  norm(x-true)/norm(true);
if (verbose)
    fprintf('iter = %d, obj = %3.3g\n', 1, objective(1))
end
exit_cond=0;continua=1;k=1;
while continua
%for k = 2:maxiters
    k=k+1;
    x_old = x;
    t_old = t;
    y1=A(y,Kr,Kc,L2,T_reg) - b; 
    y = y - (1/L)*AT(y1,Kr,Kc,L2,T_reg );
    x = soft(y,tau/L);
    

    t = 0.5*( 1 + sqrt(1+4*t_old^2) );
    y = x + ( (t_old - 1)/t )*(x - x_old);
    nres(k)=norm(y1/Amp_scale);
    objective(k) = 0.5*norm(A(x,Kr,Kc,L2,T_reg)-b)^2 + tau*sum(sum( abs(x) ));
    mses(k) = norm(x-true)^2/numel(true);
    erel_int(k)=  norm(x-true)/norm(true);

 %   times(k) = cputime - t0;
           
    switch (stopcriterion)
        case 1
            criterion = abs(objective(k)-objective(k-1))/objective(k);
        case 2
           criterion = norm((x-x_old),'fro')/norm(x,'fro');
       case 3
            criterion = objective(k);
        otherwise
            error('Invalid stopping criterion!');
    end
    crit(k)=criterion;
    
    if k>2
        Delta_crit=abs((crit(k)-crit(k-1))/crit(k));
        if Delta_crit <Thresh
              exit_cond=1;%k_exit=k;
        end
    end
    if (verbose)
        fprintf('iter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\n', k, objective(k), criterion, tolerance)
    end
    continua= criterion > tolerance && ~exit_cond && k <= maxiters;
%

end
     times=toc;
     par_out.it_int=k;
     par_out.objective= objective;
     par_out.times= times;
     par_out.mses=mses;
     par_out.nres=nres;
     par_out.erel_int=erel_int;
     par_out.crit_out=crit;   
     %
end
%--------------------------------------------------------------------------

function y = soft(x,T)
y=sign(x-T).*(max((abs(x)-T),0));

% if sum(abs(T(:)))==0
%    y = x;
% else
%    y = max(abs(x) - T, 0);
%    y = y./(y+T) .* x;
% end
end

function y=A(x,Kr,Kc,L2,theta2)
% x un vettore 

[~,nx]=size(Kr); [~,ny]=size(Kc);
y = Kc*reshape(x,nx,ny)*Kr'; y=y(:);
y=[y;theta2(:).*(L2*x)];
end

function y=AT(x,Kr,Kc,L2,theta2)
% x  un vettore
[N1,nx]=size(Kr);
[N2,ny]=size(Kc);
VV=reshape(x(1:N1*N2),N2,N1);y=Kc'*VV*Kr; y=y(:);  %% Correzione FZ 10/10/17
y=y+theta2(:).*(L2*x(N1*N2+1:nx*ny+N1*N2));
end

%###################################################################################################
%NAME    :grad_proj_noreg.m
%PURPOSE :Metodo del Gradiente Proiettato.
%DATE    :
%VERSION :1.1 [03/01/2016](vb) Cosmetic changes.
%###################################################################################################
%
function [x, k, norma_grad]=grad_proj_noreg(Kc,Kr,s, lb, x0, par)
 %
 % set up tolerances
 tol=par.gpnr.tol;
 maxk=par.gpnr.maxiter;
 %STEP 1 (inizializzazione)
 x = max(x0,lb);
 [nx,ny] = size(x0);
 alpha_min=1e-10; 
 alpha_max=1e10; %intervallo nel quale varia alpha
 alpha=1;        %inizializzo alpha
 % Gradient of the objective function
 temp = Kc*x*Kr'-s; 
 res = temp;
 grad=Kc'*temp*Kr;   %A'(Ax-b) 
 %norma_res(1)=norm(res(:)); %norma del gradiente 
 norma_res(1)=norm(res(:)/par.Amp_scale); %norma del residuo pesata
 k=1; 
 continua = 1;
 while continua
    %STEP 2 (Proiezione)
    d=max(x-alpha*grad,lb)-x;
    %STEP 3(Ricerca della direzione tramite la regola della
    %minimizzazione limitata)
    temp = Kc*d*Kr'; temp=Kc'*temp*Kr;
    Ad = temp;
    if norm(Ad(:))>eps*norm(d(:))
        lambda=min(-(grad(:)'*d(:))/(d(:)'*Ad(:)), 1);     
    else
        lambda=1;
    end
    x=x+lambda*d;
    grad=grad+lambda*Ad;
    res = Kc*x*Kr'-s;
    %STEP 4 (Aggiornamento di alpha)
    if norm(Ad(:))>eps*norm(d(:))
        if mod(k, 6)<3    
            alpha=(d(:)'*Ad(:))/(Ad(:)'*Ad(:));
        else
            alpha=(d(:)'*d(:))/(d(:)'*Ad(:));
        end
        alpha=max(alpha_min,min(alpha_max, alpha));
    else
        alpha=alpha_max;
    end
    k=k+1;
    norma_grad(k)=norm(grad(:)); %norma del gradiente   
    %norma_res(k)=norm(res(:)); %norma del gradiente 
    norma_res(k)=norm(res(:)/par.Amp_scale); %norma residuo pesata 
    Diff_res= abs(norma_res(k)-norma_res(k-1));
    continua = k<maxk && Diff_res>=tol;
 end 
 if (k >= maxk),
      fprintf('*** Uscita max iter grad proj noreg Diff res = %e  k=%d \n',Diff_res, k)
 end
 return;
end


