function [hist, x] = Alg_SVRG(samples, labels, lambda, Lmax, max_it, mb)

if nargin < 6
    mb = 1;  
end 

% initialization
[d, N] = size(samples);
x      = zeros(d,1);
xold   = x;
rnd_pm = [randperm(N)];
m      = fix(2*N/mb);
mu   = zeros(d,1); 
max1   = fix(max_it/(2*N))+1;
alpha = 1 / (10*N*Lmax);
hist = zeros(max1,1);

for k = 1:max1
    
    if k > 0     
        mu = feval(@full_grad, xold, samples, labels, lambda); 
        mu = mu/N;
    end
    numx = 0;
    xx = zeros(d,1);
    
    for j = 1:m 
        %%% randomly choose minibatch   
        idx = ceil(N*rand);
        if idx <= N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
        I = sort(rnd_pm(ix));
        sample = samples(:,I);
        label  = labels(I);    

        % Gradient
        if mb > 1            
            if k==1
                gg = feval(@msub_grad, x, sample, label, lambda);
                gg = gg/mb;
            else
                gg = feval(@msub_grad,x,sample,label,lambda) - feval(@msub_grad,xold,sample,label,lambda);
                gg = gg/mb + mu;
            end
        else
            if k==1
                gg = feval(@msub_grad, x, sample, label, lambda);
            else
                gg = feval(@msub_grad,x,sample,label,lambda) - feval(@msub_grad,xold,sample,label,lambda);
                gg = gg + mu;
             end
        end
        x = x - alpha * gg;
    end 
    xold = x;

    hist(k) = Eval_loss(samples, labels, xold, 'logistic', lambda)
end
end