%###################################################################################################
%NAME    :get_dif.m
%PURPOSE :
%DATE    :
%VERSION :1.1 [03/01/2016](vb) Cosmetic changes.
%                          
%IMPUT   :nx =
%         ny =
%         BC =
%         hx =
%         hy =
%
%OUTPUT  :L1nx =
%         L1ny =
%         L2 =
%                          
%NOTES   :
%
%###################################################################################################
%
function [L1nx,L1ny,L2] = get_diff(nx,ny,BC,hx,hy)
  if nargin ==3,
    hx=1; hy=1;
  end
  switch BC
    case 'null'
        % Operatore divergenza
        D1nx = get_l(nx+1,1)/hx; D1nx = D1nx(:,1:end-1); D1nx=0.5*(D1nx-D1nx');
        D1ny = get_l(ny+1,1)/hy; D1ny = D1ny(:,1:end-1);D1ny=0.5*(D1ny-D1ny'); 
        L1nx = kron(D1ny,speye(nx)); L1ny = kron(speye(ny),D1nx);
        % Laplacian 
        D2nx = get_l(nx+2,2)/hx^2; D2nx = D2nx(:,2:end-1); 
        D2ny = get_l(ny+2,2)/hy^2; D2ny = D2ny(:,2:end-1); 
        L2 = kron(D2ny,speye(nx))+kron(speye(ny),D2nx);
    case 'cost'
        % Operatore divergenza
        D1nx = get_l(nx+1,1)/hx; D1nx = D1nx(:,1:end-1); 
        %         D1nx=0.5*(D1nx-D1nx'); D1nx(1,1)=-0.5/hx; D1nx(nx,nx)=0.5/hx;
        D1ny = get_l(ny+1,1)/hy; D1ny = D1ny(:,1:end-1); 
        %         D1ny=0.5*(D1ny-D1ny'); D1ny(1,1)=-0.5/hy; D1ny(ny,ny)=0.5/hy;
        L1nx=kron(D1ny,speye(nx)); L1ny=kron(speye(ny),D1nx);
        
        % Laplacian 
        D2n = get_l(nx+2,2)/hx^2; D2n = D2n(:,2:end-1); D2n(1,1)=-1/hx^2;  D2n(nx,nx)=-1/hx^2;
        D2m = get_l(ny+2,2)/hy^2; D2m = D2m(:,2:end-1); D2m(1,1)=-1/hy^2;  D2m(ny,ny)=-1/hy^2;
        L2 = kron(D2m,speye(nx))+kron(speye(ny),D2n);
    case 'peri'
        % Operatore divergenza
        D1n = get_l(nx+1,1); D1n = D1n(:,1:end-1); D1n=0.5*(D1n-D1n'); D1n(1,nx)=-0.5; D1n(nx,1)=0.5;
        D1m = get_l(ny+1,1); D1m = D1m(:,1:end-1); D1m=0.5*(D1m-D1m'); D1m(1,ny)=-0.5; D1m(ny,1)=0.5;
        L1nx=kron(D1m,speye(nx)); L1ny=kron(speye(ny),D1n);
        % Laplacian 
        D2n = get_l(nx+2,2); D2n = D2n(:,2:end-1); D2n(1,nx)=1; D2n(nx,1)=1; 
        D2m = get_l(ny+2,2); D2m = D2m(:,2:end-1); D2m(1,ny)=1; D2m(ny,1)=1; 
        L2 = kron(D2m,speye(nx))+kron(speye(ny),D2n);
 end
 return;
end