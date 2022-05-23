function [ x ] = chase_method( A, b )

T = A;
for i = 2 : size(T,1)
    T(i,i-1) = T(i,i-1)/T(i-1,i-1);
    T(i,i) = T(i,i) - T(i-1,i) * T(i,i-1);
end

L = zeros(size(T));
L(logical(eye(size(T)))) = 1;   %diagonal = 1
for i = 2:size(T,1)
    for j = i-1:size(T,1)
        L(i,j) = T(i,j);
        break;
    end
end

U = zeros(size(T));
U(logical(eye(size(T)))) = T(logical(eye(size(T))));
for i = 1:size(T,1)
    for j = i+1:size(T,1)
        U(i,j) = T(i,j);
        break;
    end
end


y = zeros(size(b));
y(1) = b(1);
for i = 2 : length(b)
    y(i) = b(i) - L(i, i-1) * y(i-1);
end

x(length(b)) = y(length(b)) / U(length(b), length(b));
for i = length(b)-1 : -1: 1
    x(i) = (y(i) - U(i, i+1) * x(i+1)) / U(i,i);
end

end
