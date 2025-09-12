function [x, k, norma_grad]=newt_proj1d(H,ck,s,L2,x0, tol, maxk)
%METODO DI NEWTON PROIETTATO


% Active set parameter
psi = 1e-10;

% line search parameter
maxarm = 40;
eta=1.e-4;

% Initialize
x = max(x0,0);
% [nx,ny] = size(x0);
n=length(x0);

% Compute objective and gradient
temp0 = H*x-s; temp = H'*temp0;
temp1 = ck(:).*(L2*x(:)); temp1 = L2'*temp1;
grad = temp+temp1;
objf = 0.5*(norm(temp0(:))^2+x(:)'*temp1);

k=0; eflag = 1; tol=tol*norm(grad(:)); continua = 1;
while continua
    
    k=k+1;
    
    % Evaluate "active set"
    wk = norm( x-max(0,x-grad),'fro' ); 
    epsilonk = min([psi; wk]);
    Ik = ( x<=epsilonk & grad>0 ); %''Active set''
    
    % Compute descent direction
    [d,iter] = cg1dn(Ik,H,L2,ck,-grad,n,1e-5,[]); 
        
    % Constrained line search
    alpha = 1; iarm = 1; 
    xt = max(x + alpha*d,0);
    temp0 = H*xt-s; 
    temp1 = ck(:).*(L2*xt(:)); temp1 = L2'*temp1;
    objft = 0.5*(norm(temp0(:))^2+xt(:)'*temp1);
    stept = alpha * d;
    stept(Ik) = -( x(Ik) - xt(Ik) );
    while objft >= objf+eta*grad(:)'*stept(:) && iarm <= maxarm 
        alpha = alpha * 0.5;
        xt = max(x + alpha*d,0);
        temp0 = H*xt-s; 
        temp1 = ck(:).*(L2*xt(:)); temp1 = L2'*temp1;
        objft = 0.5*(norm(temp0(:))^2+xt(:)'*temp1);
        stept = alpha * d;
        stept(Ik) = -( x(Ik) - xt(Ik) );
        iarm = iarm+1;
        if iarm > maxarm
            %fprintf('\n   *** Armijo failure *** \n');
            eflag = 0;
        end
    end  % end line search

    % Update
    x = xt; 
    rho=(objf-objft)/objft; % relative increment in the objective
    objf = objft;
    grad = H'*temp0+temp1;
    norma_grad(k)=norm(grad(:)); %norma del gradiente    
    
    % Check stopping conditions
    continua = k<maxk && norm(grad(:))>=tol && eflag == 1 && rho>1e-6;
end  
% figure;semilogy(norma_grad); pause

