function [A,b,xex,noise_norm,b_exact]=T2(N,noise_level)
% Create test problem
[A,b,xex] = heat(100);
[A,b,bb] = baart(76); bb=bb-min(bb);
[a,i]=max(xex);
xx = [xex(1:i); flipud(xex(1:i))];
xx = xx-min(xx);  xx=xx/max(xx); 
xx = [zeros(100,1); xx; zeros(30,1); 0.75*xx; zeros(100,1); bb; zeros(50,1)];
xx(121:159)=xx(121);
% 100 24 30
plot(xx);
xex = xx;


%generate blurring kernel
N = numel(xex);
X = (1:N)';
sblur = 5; %standard deviation
g = exp(-( ((X-N/2-1).^2)/(2*sblur^2) )); %Gaussian shifted by N/2 (centered blur kernel)
kc = g; %blur kernel (point spread function) equals Gaussian
kc = kc/(sum(sum(kc))); %normalze
figure; plot(-N/2:N/2-1,kc); title(['Gaussian blur kernel k, \sigma = ' num2str(sblur)']); %plot blur kernel
A = circulant(fftshift(kc),1);


% add gaussian noise
b_exact=A*xex;

randn('seed',0);
noise = randn(size(b_exact));
noise = noise/norm(noise);
noise = noise_level*noise*norm(b_exact); noise_norm = norm(noise);
b = noise+b_exact;
end