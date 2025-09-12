function [x, k, norma_res, norma_grad]=grad_proj_noreg1d(H,s, lb, x0, tol, maxk)
%METODO DEL GRADIENTE PROIETTATO

%STEP 1 (inizializzazione)
x = max(x0,lb);

alpha_min=1e-10; alpha_max=1e10; %intervallo nel quale varia alpha
alpha=1; %inizializzo alpha
 

% Gradient of the objective function
temp =H*x-s; 
res = temp;
grad=H'*temp;   %A'(Ax-b) 
norma_res(1)=norm(res); %norma del gradiente 
norma_grad(1) = norm(grad);
k=1; %tol=tol*norm(grad(:));

continua = 1;
while continua

    %STEP 2 (Proiezione)
    d=max(x-alpha*grad,lb)-x;

    %STEP 3(Ricerca della direzione tramite la regola della
    %minimizzazione limitata)
    temp = H'*H*d; 
    Ad = temp;
    if norm(Ad(:))>eps*norm(d(:))
        lambda=min(-(grad(:)'*d(:))/(d(:)'*Ad(:)), 1);     
    else
        lambda=1;
    end
    x=x+lambda*d;
    grad=grad+lambda*Ad;
    res = H*x-s;

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
    norma_res(k)=norm(res(:)); %norma del gradiente 
    
    continua = k<maxk && abs(norma_res(k)-norma_res(k-1))>=tol;
end  
%figure;semilogy(norma_grad); 
%figure;semilogy(norma_res);

