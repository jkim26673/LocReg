%###################################################################################################
%NAME    :LoadFlags.m
%PURPOSE :loads flags from file.
%DATE    :15/07/2017
%VERSION :1.1 [00/00/0000]
%         1.2 [19/10/2017] (vb) added FLNoContour flag to plot with or without countur.
%
%IMPUT   :InputFileName = name of the file with Flag values
%         
%         
%
%OUTPUT  :           
%                          
%NOTES   :
%
%###################################################################################################
%
function [CommentTS, FL_typeKernel, FL_UPEN2D, FL_TIKHONOV, FL_Stelar_Magriteck, FL_InversionTimeLimits, ...
      FL_LoadMatrixB, FL_UseMatrixB, FL_OutputData, FL_NoContour]= LoadFlags(InputFileName, UseDefault) 

  %Load from file if file exists.
  %   fprintf('InputName=%s \n',InputFileName);
  if(UseDefault)
     CommentTS='Default Flags';
     FL_typeKernel=4;          %1 IR-CPMG; 2 SR-CPMG; 3 D-T2; 4 T2-T2
     FL_UPEN2D=1;              %1 yes
     FL_TIKHONOV=0;            %1 yes
     FL_Stelar_Magriteck=0;    %1 Stelar, 0 Magriteck
     FL_InversionTimeLimits=0; %1 autmomatic, 0 manually selection inversion times
     FL_LoadMatrixB=0;         %1 load matrix from file, 0 generate unitary matrix
     FL_UseMatrixB=0;          %1 uses B matrix, 0 not uses B matrix
     FL_OutputData=0;          %1 create output data file for ILT2D
     FL_NoContour=1;           %1 no image with contour
   else
    fid = fopen(InputFileName);  %
    %
    CommentTS = fgetl(fid);       % a row of comment
    %extract flags values
    while(1)
       stringa=fgetl(fid);
       stringa=strtrim(stringa);
       if(strfind(stringa, 'END')==1) break; end   %stops reading parameters.
       if(strfind(stringa, 'FL_typeKernel         =')==1) 
          FL_typeKernel=str2double(strrep(stringa,'FL_typeKernel         =',''));
       end
       if(strfind(stringa, 'FL_UPEN2D             =')==1) 
          FL_UPEN2D=str2double(strrep(stringa,'FL_UPEN2D             =',''));
       end
       if(strfind(stringa, 'FL_TIKHONOV           =')==1) 
          FL_TIKHONOV=str2double(strrep(stringa,'FL_TIKHONOV           =',''));
       end
       if(strfind(stringa, 'FL_Stelar_Magriteck   =')==1) 
          FL_Stelar_Magriteck=str2double(strrep(stringa,'FL_Stelar_Magriteck   =',''));
       end
       if(strfind(stringa, 'FL_InversionTimeLimits=')==1) 
          FL_InversionTimeLimits=str2double(strrep(stringa,'FL_InversionTimeLimits=',''));
       end
       if(strfind(stringa, 'FL_LoadMatrixB        =')==1) 
          FL_LoadMatrixB=str2double(strrep(stringa,'FL_LoadMatrixB        =',''));
       end
       if(strfind(stringa, 'FL_UseMatrixB         =')==1) 
          FL_UseMatrixB=str2double(strrep(stringa,'FL_UseMatrixB         =',''));
       end
       if(strfind(stringa, 'FL_OutputData         =')==1) 
          FL_OutputData=str2double(strrep(stringa,'FL_OutputData         =',''));
       end
       if(strfind(stringa, 'FL_NoContour          =')==1) 
          FL_NoContour=str2double(strrep(stringa,'FL_NoContour          =',''));
       end
       
    end
    fclose(fid);
  end
  return;
end

