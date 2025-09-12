function [CommentTS, N_T1, N_T2, t_T1, t_T2, S] = LoadDataFileMagritek(DataFileName, TimeRowFileName, TimeColumnFileName)
%NAME    :LoadDataFileMagritek.m
%PURPOSE :Extracts data and times from Magritek data files.
%VERSION: 1.0 [10/04/2017]
%DATE    :10/04/2017
%CHANGES :1.0 [] 
%
%AUTHOR  :VB.
%
%
%CommentTS = fgetl(fid);       % a row of comment
%extract parameters
%fprintf('InputName=%s \n',InputFileName);
fid = fopen(TimeRowFileName);  %
 t_T1 = fscanf(fid,'%f');
fclose(fid);
N_T1=size(t_T1);
%
fid = fopen(TimeColumnFileName);  %
 t_T2 = fscanf(fid,'%f');
fclose(fid);
N_T2=size(t_T2);
%
S = dlmread(DataFileName);
%
CommentTS='';
%
return;
%
end

