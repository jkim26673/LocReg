
NameFileFlags=[Data_folder 'FileFlag_fista.par'];
NameFileSetInput=[Data_folder 'FileSetInput.par'];
Name_Par_Fista=[Data_folder 'FilePar_fista.par'];
addpath(Data_folder)
%############################### FLag parameters #######################################
% FL_typeKernel=4;          %1 IR-CPMG; 2 SR-CPMG; 3 D-T2; 4 T2-T2
% FL_UPEN2D=1;              %1 I2DUPEN   2: I2DUPEN+FISTA
% FL_TIKHONOV=0;            %1 yes
% FL_Stelar_Magriteck=0;    %1 Stelar, 0 Magriteck, 2 Simulated
% FL_InversionTimeLimits=0; %1 autmomatic, 0 manually selection inversion times
% FL_LoadMatrixB=0;         %1 load matrix from file, 0 generate unitary matrix
% FL_UseMatrixB=0;          %1 uses B matrix, 0 not uses B matrix
% FL_OutputData=0;          %1 create output data file for ILT2D
% load flag fom file
[CommentS, FL_typeKernel, FL_UPEN2D, FL_TIKHONOV, FL_Stelar_Magriteck, FL_InversionTimeLimits, ...
      FL_LoadMatrixB, FL_UseMatrixB, FL_OutputData, FL_NoContour]= LoadFlags(NameFileFlags,0);  
  %%
%########################## Declaration parameter strucuture ###########################
% creates a 1-by-1 structure with no fields.
par=struct;
parFile=struct;
%
%#################################### Load Data Set ####################################
[parFile]=SetInputFile(NameFileSetInput, parFile, 0);
%%
Amp_scale=1;scale_fact=1;
switch FL_Stelar_Magriteck
  case 0
    [CommentTS, N_T1, N_T2, Tau1, Tau2, s] = LoadDataFileMagritek(parFile.filenamedata, parFile.filenameTimeY,parFile.filenameTimeX);
    if(FL_LoadMatrixB)
       [CommentB, B] = LoadBMatrix(parFile.FileNameMatrixB); scale_fact=1E3;
       B = sqrt(B);
    end   
  case 1
    [CommentTS, N_T1, N_T2, Tau1, Tau2, s] = LoadInputDataFile3(parFile.filenamedata, 1,1);
    if(FL_LoadMatrixB)
       [CommentB, B] = LoadBMatrix(parFile.FileNameMatrixB); 
       B = sqrt(B);
       Amp_scale=1E4;
    end
  case 2
     [CommentTS, N_T1, N_T2, Tau1, Tau2, s] = LoadDataFileMagritek(parFile.filenamedata, parFile.filenameTimeY,parFile.filenameTimeX);
     Amp_scale=1E4;
     if(FL_LoadMatrixB)
     [CommentB, B] = LoadBMatrix(parFile.FileNameMatrixB); 
     %
     B = sqrt(B);
     else
         B=eye(N_T2);
     end
end

%#########################################################################################
%  Use of B matrix
% (vb) 06/11/2017 fixed bug
if FL_UseMatrixB
    if(FL_LoadMatrixB)
       Testo='B File';
    else
       Testo='B Identity';
       B=eye(size(s,2));    
    end
 else
    Testo='No B';
    B=0;
end
%###################################### Set problem dimension ############################
nx=parFile.nx;
ny=parFile.ny;
N=nx*ny;
%
%######################## Set times of the inversion channels ############################
%Set times of the inversion channels. Two modalities: authomatic setting or fixed setting.
%Times are in milliseconds
if(FL_InversionTimeLimits==1)
   if(FL_typeKernel==1||FL_typeKernel==2||FL_typeKernel==4)
     % Relaxation Time 
     Tau1=scale_fact*Tau1; Tau2=scale_fact*Tau2;
     q1 = exp((1/(nx-1))*log(4*Tau1(end)/(0.25*Tau1(1))));
     T1 = 0.25*Tau1(1)*q1.^(0:nx-1);
   else
     %diffusion inversion point
     q1 = exp((1/(nx-1))*log(10/0.001));
     T1 = 0.001*q1.^(0:nx-1);
   end
   q2 = exp((1/(ny-1))*log(4*Tau2(end)/(0.25*Tau2(1))));
   T2 = 0.25*Tau2(1)*q2.^(0:ny-1);
else
   T1min=parFile.T1min;
   T1max=parFile.T1max;
   T2min=parFile.T2min;
   T2max=parFile.T2max;
   q1 = exp((1/(nx-1))*log(T1max/T1min));
   T1 = T1min*q1.^(0:nx-1);
   q2 = exp((1/(ny-1))*log(T2max/T2min));
   T2 = T2min*q2.^(0:ny-1);
end
%
%############################# Set the Kernel #######################################
if(FL_typeKernel==1) %IR-CPMG
   Kernel_1 = inline('1-2*exp(- Tau * (1./ T1))','Tau','T1');
 elseif(FL_typeKernel==2)%SR-CPMG
   Kernel_1 = inline('1-exp(- Tau * (1./ T1))','Tau','T1');
 elseif(FL_typeKernel==3)%D-T2
   Kernel_1 = inline('exp(- Tau * (1.* T1))','Tau','T1');
 elseif(FL_typeKernel==4)%T2-T2
   Kernel_1 = inline('exp(- Tau * (1./ T1))','Tau','T1');
end
Kc = Kernel_1 (Tau1,T1); 
Kernel_2 = inline('exp( - Tau * (1./ T2))','Tau','T2');
Kr = Kernel_2(Tau2,T2);
%
%############################# Set the Parameter structure ##########################
%
[par]=SetPar(Name_Par_Fista,s, B, FL_UseMatrixB, par,0);