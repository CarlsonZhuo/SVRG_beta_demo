function [ grad ] = full_grad( w, X, y, lambda )
[~, N] = size(X);
grad = 0;
for i=1:N
    sigmoid_term = 1/(1+exp(-y(i)*w'*X(:,i)));
    grad = grad + (sigmoid_term-1) * y(i) * X(:,i);
end
grad = grad + lambda*w;

end
