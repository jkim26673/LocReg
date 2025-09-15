      const=max(pars_reg);
     %[min(pars_reg) max(pars_reg) const]
     [sv, iv]=sort(pars_reg,'descend');
     indx=find(sv<const);
     Num_const=indx(1)-1; 
     Prod=1;v=sv(indx); v_old=sv_old(indx);
     Prod_old=1;flag=1;
     %;
     for ii=numel(v):-1:1
         val=v(ii);
         pp=Prod*val;
         if pp < realmax && flag
             Prod=pp;
             Prod_old=min(Prod_old*v_old(ii),realmax);
         else
             ind_excl=Num_const+indx(ii);
             flag=0;
         end
     end
     P_e=1;Pe_old=1;
%
     for ii=Num_const:ind_excl-Num_const
            P_e=P_e*sv(ii);
            Pe_old=Pe_old*sv_old(ii);
     end
      DEN=P_e;DEN_old=Pe_old;
      vq1=(NUM^2)/(DEN^(1/NN));
      cond_Q = vq1 < (NUM_old^2)/(DEN_old^(1/NN));
       [(NUM^2)/(DEN^(1/NN)), (NUM_old^2)/(DEN_old^(1/NN))];
