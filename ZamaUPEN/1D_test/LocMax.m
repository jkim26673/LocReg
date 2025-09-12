function w = LocMax(z)

w = zeros(size(z));
n = length(z);
w(1) = max(z(1:2));
for k=3:n-2
    w(k)=max(z(k-1:k+1));
end
w(n) = max(z(n-1:n));
end
