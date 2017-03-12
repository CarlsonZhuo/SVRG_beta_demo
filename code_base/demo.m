
clear;

%% Get the data X, y
%   1. X: [m*n], each column of X is one sample data;
%   2. y: [n*1], is the label of each sample data.A(i,:).
%   3. w: [m*1], is the number of features.

% load('../data/rcv1_train.binary.mat');
load('../data/adult.mat');
X = [ones(size(X,1),1) X];
[n, d] = size(X);
X = X';

% addpath('..\libsvm');
% [y, X] = libsvmread('..\data\australian');

% addpath('../libsvm-3.21/matlab');
% [y, X] = libsvmread('../data/australian');
% [n, d] = size(X);
% X = X';    
% X = full(X); 

% Data normalization 
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2, 1)));
end


%% Get the approximation of the best parameter
% lambda = 1/(n);
lambda = 1 / 100000;
Lmax   = (0.25 * max(sum(X.^2,2)) + lambda);

max_it = 50*2*n;

% PSVRG

w_PSVRG = zeros(d, 1);
tic;
[histPSVRG_l2, w_PSVRG] = Alg_PSVRG(X, y, lambda, Lmax, max_it, 5);
time_PSVRG = toc;
fprintf('Time spent on SVRG: %f seconds \n', time_PSVRG);



% SVRG
w_SVRG = zeros(d, 1);
tic;
[histSVRG_l2, w_SVRG] = Alg_SVRG(X, y, lambda, Lmax, max_it,5);
time_SVRG = toc;
fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);

semilogy(histSVRG_l2);
hold on;
semilogy(histPSVRG_l2)