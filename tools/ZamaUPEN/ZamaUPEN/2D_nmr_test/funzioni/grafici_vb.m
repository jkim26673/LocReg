%##########################################################################
%Name    : grafici_vb
%Changes : 
%                       
%Date    : [11/07/2017]  Code partially derived by the Challagan code.
%
function grafici_vb(X,tauv,tauh,Contur_or_Surf, FL_typeKernel, numline, Titolo)
    fig=figure;
    axes('FontSize',12);
    set(gcf,'Renderer','zbuffer');
    set(fig,'DoubleBuffer','on');
    set(gca,'NextPlot','replace','Visible','off')
    taulh = log10(tauh);
    sta = size(taulh);
    taulv = log10(tauv);
    stb = size(taulv);
    %
    if Contur_or_Surf
      surf(taulh,taulv',X);
     else
      contour(taulh,taulv',X,numline);
    end
    %
    caxis('auto');
	%caxis([0 1]);
    shading interp;
    %axis([taulh(1),taulh(sta(2)),taulv(1),taulv(stb(2))]);
    axis([taulh(1),taulh(sta(2)),taulv(1),taulv(stb(2))]);grid on
    colorbar;
    if (FL_typeKernel==1 || FL_typeKernel==2)
      xlabel('Log_{10}(T_2)  [T_2 in ms]'); %xlabel('log(T1) (ms)'); %%%%
      ylabel('Log_{10}(T_1)  [T_1 in ms]'); %xlabel('log(T1) (ms)'); %%%%
     elseif FL_typeKernel==3
      xlabel('Log_{10}(T_2)  [T_2 in ms]'); %
      ylabel('Log_{10} D (\mum^2/ms)'); %
     elseif FL_typeKernel==4
      xlabel('Log_{10}(T_{22})  [T_{22} in ms]'); %xlabel('log(T1) (ms)'); %%%% 
      ylabel('Log_{10}(T_{21})  [T_{21} in ms]'); %xlabel('log(T1) (ms)'); %%%%
    end
    title(Titolo);
%     if FL_typeKernel == 1 
%         title('T1-T2 correlation - IRCPMG','FontSize',16);
%       elseif FL_typeKernel == 2
%         title('T1-T2 correlation - SRCPMG','FontSize',16);
%       elseif FL_typeKernel == 3
%         title('D-T2 correlation','FontSize',16);
%       elseif FL_typeKernel == 4
%         title('T2-T2 correlation','FontSize',16);
%     end
  return;
end   