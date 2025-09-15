%###################################################################################################
%NAME    :SetPar.m
%PURPOSE :Set/load parameters from file.
%DATE    :12/07/2017
%VERSION :1.1 [15/07/2017] manages file of parameters
%
%IMPUT   :InputFileName = name of the file with parameters
%         
%         
%
%OUTPUT  :           
%                          
%NOTES   :
%
%###################################################################################################
%
function [par]= SetPar(InputFileName, s, B, FL_UseMatrixB, par, UseDefault)
  if UseDefault
    %Set Defaulf
    % [GRADIENTE PROIETTATO]
    par.gpnr.tol          =0.1;
    par.gpnr.maxiter      =50000;
    %
    % [NEWTON PROIETTATO]
    %par.nwtp.maxiter      =500;
    %par.nwtp.tolrho       =1.0E-4;
    par.nwtp.maxiter      =1E5;
    par.nwtp.tolrho       =1.0E-8;
    %
    % [CG]
    %par.cgn2d.tol        = 0.5;
    %par.cgn2d.maxiter    = length(s(:));
    par.cgn2d.tol         =1.E-3;
    par.cgn2d.maxiter     =3000;
    %
    %[SVD]
    par.svd.svd           =1; % per proiezione-filtro SVD
    par.svd.soglia        =1.0E-6; % SOLO per par.svd=1;
    % 
    %[UPEN]
    par.upen.tol          =0.001;   % !
    par.upen.iter         =100;
    par.upen.beta00       =1;
    par.upen.beta0        =1.0E-4; 
    par.upen.beta_p       =1.0E-5;
    par.upen.beta_c       =1.0E-1;
    %
    %[TIKHONOV]
    par.tikh.lambda =2.0E2;
    %
    %Load from file if file exists.
  else 
    fprintf('InputName=%s \n',InputFileName);
    fid = fopen(InputFileName);  %
    %
    CommentTS = fgetl(fid);       % a row of comment
    % %extract parameters
    while(1)
      stringa=fgetl(fid);
      stringa=strtrim(stringa);
      if(strfind(stringa, 'END')==1) break; end   %stops reading parameters.
      % [GRADIENTE PROIETTATO]
      if(strfind(stringa, 'par.gpnr.tol          =')==1) 
         par.gpnr.tol=str2double(strrep(stringa,'par.gpnr.tol          =',''));
      end
      if(strfind(stringa, 'par.gpnr.maxiter      =')==1) 
          par.gpnr.maxiter=str2double(strrep(stringa,'par.gpnr.maxiter      =',''));
      end
      % [NEWTON PROIETTATO]
      if(strfind(stringa, 'par.nwtp.maxiter      =')==1) 
          par.nwtp.maxiter=str2double(strrep(stringa,'par.nwtp.maxiter      =',''));
      end
      if(strfind(stringa, 'par.nwtp.tolrho       =')==1) 
          par.nwtp.tolrho=str2double(strrep(stringa,'par.nwtp.tolrho       =',''));
      end
      % [CG]
      if(strfind(stringa, 'par.cgn2d.tol         =')==1) 
          par.cgn2d.tol=str2double(strrep(stringa,'par.cgn2d.tol         =',''));
      end
      if(strfind(stringa, 'par.cgn2d.maxiter     =')==1) 
          par.cgn2d.maxiter=str2double(strrep(stringa,'par.cgn2d.maxiter     =',''));
      end
      %[SVD]
      if(strfind(stringa, 'par.svd.svd           =')==1) 
          par.svd.svd=str2double(strrep(stringa,'par.svd.svd           =',''));
      end
      if(strfind(stringa, 'par.svd.soglia        =')==1) 
          par.svd.soglia=str2double(strrep(stringa,'par.svd.soglia        =',''));
      end
      %[UPEN]
      if(strfind(stringa, 'par.upen.tol          =')==1) 
          par.upen.tol=str2double(strrep(stringa,  'par.upen.tol          =',''));
      end    
      if(strfind(stringa, 'par.upen.iter         =')==1) 
          par.upen.iter=str2double(strrep(stringa, 'par.upen.iter         =',''));
      end
      if(strfind(stringa, 'par.upen.beta00       =')==1)
         par.upen.beta00=str2double(strrep(stringa,'par.upen.beta00       =',''));
      end
      if(strfind(stringa, 'par.upen.beta0        =')==1)
         par.upen.beta0=str2double(strrep(stringa,'par.upen.beta0        =',''));
      end
      if(strfind(stringa, 'par.upen.beta_p       =')==1)
          par.upen.beta_p=str2double(strrep(stringa,'par.upen.beta_p       =',''));
      end
      if(strfind(stringa, 'par.upen.beta_c       =')==1) 
          par.upen.beta_c=str2double(strrep(stringa,'par.upen.beta_c       =',''));
      end
      %[TIKHONOV]
      if(strfind(stringa, 'par.tikh.lambda       =')==1) 
          par.tikh.lambda=str2double(strrep(stringa,'par.tikh.lambda       =',''));
      end
    end
    fclose(fid);
  end
  %
  %[VARIOUS]
  if (FL_UseMatrixB)
     par.weightB=1; % 1 uses weight B matrix
   else
     par.weightB=0; % 0 does not use weight B matrix
  end
  par.upen.B = B;
  return;
%
end

