%%         1.1 [12/07/2017] (vb) Adapted to use the new flip_imagesc_new function
function grafico_1(x,T1,T2,metodo, FL_typeKernel)

 [nx,ny]=size(x);
 % picco
 [~,iy] = max(max(x));
 [~,ix] = max(max(x'));
 picco = x(ix,iy);
 M_picco=x(max(ix-5,1):min(ix+5,nx),max(iy-5,1):min(iy+5,ny)); Perc=100*sum(M_picco(:))/sum(x(:));
 fprintf('%s T2=%0.2f T1=%0.2f picco=%0.2f  PercTot=%0.2f  \n',metodo,T2(iy),T1(ix),picco,Perc);
 fprintf('   (T2= %0.2f %0.2f %0.2f, T1 = %0.2f %0.2f %0.2f) \n',T2(iy-1),T2(iy),T2(iy+1),T1(ix-1),T1(ix),T1(ix+1));

 Titolo=[metodo ' T2(' num2str(iy) ')=' num2str(T2(iy),'%0.2f') ...
    ' T1(' num2str(ix) ')=' num2str(T1(ix),'%0.2f') ' peak =' num2str(picco,'%0.2f')];
 figure; flip_imagesc_new(x,T1,T2, Titolo, 1, FL_typeKernel);
 figure; surf(x); grid on %title(metodo); 

 analisi_T1=sum(x,2);
 analisi_T2=sum(x,1);
 figure
 semilogx(T1,analisi_T1); axis([T1(1) T1(end) min(analisi_T1) max(analisi_T1)]);grid on
 xlabel('T1 (ms)');%xlabel('log(T1) (ms)'); %%%%
 ylabel('Probability Density (a.u.)');%ylabel('Densit''a di probabilit''a (u.a.)')   
 %title(metodo);
 figure; semilogx(T2,analisi_T2); axis([T2(1) T2(end) min(analisi_T2) max(analisi_T2)]);grid on
 xlabel('T2 (ms)');%xlabel('log(T2) (ms)'); %%%%
 ylabel('Probability Density (a.u.)');%ylabel('Densit''a di probabilit''a (u.a.)')
 %title(metodo);
 return;
end
