  function [u_mat,pcgiter] = cg1dn(F,H,L2,ck,b_mat,pcg_max,pcg_tol,u0)
  % PCG for linear system with coefficient matrix of the form:
  %                     (K^T*K+L^T*lambda*L)
  % where k=kron(K_r,K_c)
  
  % INPUT: KcTKc = Kc^T*Kc
  %        KrTKr = Kr^T*Kr
  %        LTL = L^t*L

%   [nx,ny]=size(b_mat); 

%  PCG initialization.
  E=1-F;
  if isempty(u0)
      u_mat = zeros(size(b_mat));
      resid_mat = b_mat;
  else 
      u_mat=u0;
      temp = H*(E.*u_mat); 
      temp1 = ck(:).*(L2*(E(:).*u_mat(:))); temp1 = L2'*temp1;
      resid_mat =E.*( H'*temp+temp1)+F.*u_mat;
  end

%  Perform PCG iterations to solve Ah*u = bh. 
%  Keep track of the norm of the residual and the solution error.

  pcgiter = 0;
  residrat = norm(resid_mat(:));
  pcg_tol = pcg_tol*norm(resid_mat(:));
  while (pcgiter < pcg_max && residrat > pcg_tol)

    pcgiter = pcgiter + 1;
    d_mat = resid_mat;

    %  Compute conjugate direction p and update u, residual.

    rd = resid_mat(:)'*d_mat(:);
    if pcgiter == 1,
       p_mat = d_mat; 
     else
       betak = rd / rdlast;
       p_mat = d_mat + betak * p_mat;
    end
      temp = H*(E.*p_mat); 
      temp1 = ck(:).*(L2*(E(:).*p_mat(:))); temp1 = L2'*temp1;
      Ap_mat =E.*( H'*temp+temp1)+F.*p_mat;
    
    alphak = rd / (p_mat(:)'*Ap_mat(:));
    u_mat = u_mat + alphak*p_mat;
    resid_mat = resid_mat - alphak*Ap_mat;
    rdlast = rd;
    residrat = norm(resid_mat(:));
    NormR(pcgiter)=residrat;
    R_mean(pcgiter)=mean(NormR(1:pcgiter));
  end

% figure;semilogy(1:pcgiter,NormR,'-',1:pcgiter,R_mean,'-r');pause
