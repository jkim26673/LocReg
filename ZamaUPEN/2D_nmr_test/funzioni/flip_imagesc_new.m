%###################################################################################################
%NAME    :flip_imagesc_new.m
%PURPOSE :
%DATE    :
%VERSION :1.1 [03/01/2016](vb) Cosmetic changes.
%         1.2 [11/07/2017](fz; vb) Changed plot style (similar to Challagan).
%         1.2 [12/07/2017](vb) fixed some bugs.
%
%                          
%IMPUT   :X =
%         S =
%
%OUTPUT  : 
%                          
%NOTES   :
%
%###################################################################################################
%
function flip_imagesc_new(X, T1, T2, S, Contur_or_Surf, FL_Kernel)
 [Ty,Tx]=meshgrid(T1,T2);
 if Contur_or_Surf
    %surf(log10(Tx/1000),log10(Ty/1000),X);
    surf(log10(Tx),log10(Ty),X);
 else
    %contour(log10(Tx/1000),log10(Ty/1000),X);
    contour(log10(Tx),log10(Ty),X);
 end
 shading flat;
 az = 90;
 el = -90;
 view(az, el);
 if (FL_Kernel==1 || FL_Kernel==2)
     ylabel('log_{10}(T_2) [T_2 in ms]');%xlabel('log(T1) (ms)'); %%%%
     xlabel('log_{10}(T_1) [T_1 in ms]'); 
  elseif FL_Kernel==3
     %xlabel('log(T1) (ms)'); %%%%  
     ylabel('log_{10}(T_2) [T_2 in ms]'); 
     xlabel('log_{10}(D) [D in \mum^2/ms]');
  elseif FL_Kernel==4
     ylabel('log_{10}(T_{22}) [T_{22} in ms]');%xlabel('log(T1) (ms)'); %%%%  
     xlabel('log_{10}(T_{21}) [T_{21} in ms]'); 
 end
 %ylabel('log(T2) [T2 in s]');
 colorbar;
 title(S);
%  imagesc(X);
%  [ny,nx]=size(X);
%  T2_tick=[1 10:10:nx];
%  if T2_tick(end)<nx
%     T2_tick = [T2_tick nx];
%  end
%  T1_tick=[1 10:10:ny];
%  if T1_tick(end)<ny
%     T1_tick = [T1_tick ny];
%  end
%  T2_label=T2_tick;
%  T1_label=T1_tick;
%  set(gca,...
%     'Xdir','normal', ...
%     'XTick', T2_tick,...
%     'XTickLabel', ...
%     arrayfun(@num2str, T2_label(:), 'UniformOutput', false),...
%     'Ydir','normal', ...
%     'YTick', T1_tick,...
%     'YTickLabel', ...
%     arrayfun(@num2str, T1_label(:), 'UniformOutput', false));
% 
%  % set(gca,'FontSize',16);
%  % xlabel('T2 (ms)');
%  % ylabel('T1 (ms)'); 
%  colorbar;
%  title(S);
 return;
end