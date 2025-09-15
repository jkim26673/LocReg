function [A,b,xex,noise_norm,b_exact] = T1(N,noise_level)
[A,b,xex] = heat(N);

% add gaussian noise
b_exact=A*xex;

randn('seed',0);
noise = randn(size(b_exact));
noise = noise/norm(noise);
noise = noise_level*noise*norm(b_exact); noise_norm = norm(noise);
b = noise+b_exact;
end