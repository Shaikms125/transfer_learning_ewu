% This code is part of the supplementary material to the paper
%
% @article{FernandoPRL2015,
% author = {Basura Fernando, Tatiana Tommasi, Tinne Tuytelaars},
% title = {Joint cross domain classification and subspace learning for unsupervised adaptation},
% journal = {Pattern Recognition Letters},
% year = {2015},
% }
%
% Copyright (c) 2015, Basura Fernando
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
% list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation and/or
% other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function acc = JCSL(Xs, Ys, Xt, Yt, dim) %my_edit: function signature compatible with demo.m
% Adaptor for multi-class cross-position activity recognition in demo.m

%my_edit start
if nargin < 5 || isempty(dim)
    dim = 30; % default subspace dimension
end

% Hyperparameters (can be changed manually here)
Params.C = dim;    % dimensionality of subspace
Params.L = 1.0;    % hinge loss weight (use 1.0 or 0.01)
Params.B = 0.5;    % weight of Frobenius norm ||U-V||_fro

% Compute initial source subspace U_o using PCA
[U_o, ~, ~] = pca(Xs);
U_o = U_o(:, 1:dim);

% One-vs-all (OVA) multi-class classification
classes = unique(Ys);
num_classes = numel(classes);
W = zeros(dim, num_classes);

for c = 1 : num_classes
    % Binary label: +1 for class c, -1 for all other classes
    Ys_binary = -ones(size(Ys));
    Ys_binary(Ys == classes(c)) = 1;
    
    % Call original JCSL core routine for binary training
    [w, ~, V] = JCSL_core(Xs, Xt, Ys_binary, Params, U_o);
    W(:, c) = w;
end

% Project target samples into target subspace V and predict
scores = (Xt * V) * W;
[~, pred_idx] = max(scores, [], 2);
pred = classes(pred_idx);

% Calculate target test accuracy (percentage)
acc = mean(pred == Yt) * 100;
%my_edit end
end


function [w,U,V,overall_obj] = JCSL_core(Xs,Xt,Ys,Params,U_o) %my_edit: renamed to JCSL_core to preserve original implementation
ns = size(Xs,1); % number of source samples
[V,~,~] = pca(Xt); % PCA for the target %my_edit: replaced princomp with pca for compatibility
d = Params.C; % dimensionality of subspace
V = V(:,1:d); % target subpsace
%---Initialization
U = U_o; % initialize U
niter = 50; %my_edit: 50 epochs for fast and practical convergence
overall_obj = zeros(1,niter);
L = Params.L; %0.01;
B = Params.B; %0.5;
%--- Initialize w
% w = randn(d,1);
% [w, ~,~] = vl_svmtrain( (Xs * U)', Ys', L); % better initialize with vl_feat
w = zeros(d,1); %my_edit: zero initialization avoids external vl_feat dependency
batch_size = 25;
learn_rate_0 = 10^-2; %my_edit: 10^-2 provides stable learning rate for SGD
D = d; % dimensionality
learn_rate_w = learn_rate_0;
learn_rate_u = learn_rate_0;
obj_old = w'*w + L * mean(max(0,1-Ys.*(Xs*U*w))) + B * norm(U-V,'fro'); %my_edit: syntax fix for * and .*
for iter = 1 : niter
 batches = ceil(ns / batch_size);
 rn = (randperm(ns));
 for bi = 1 : batches
 indx = [(bi-1)*batch_size + 1: min(bi * batch_size,ns)]; %my_edit: fixed (bi-1)*batch_size
 indx = rn(indx);
 shi = Ys(indx).*(Xs(indx,:)*U*w); %my_edit: added missing operators .* and *
 error_indx = indx(find(shi < 1));
 ne = numel(error_indx); 
 if ne > 0
 gradient_w = 2*w + L * mean( -(Xs(error_indx,:)*U)' .* repmat(Ys(error_indx)',D,1) , 2 ); %my_edit: syntax fix for * and .*
 gradient_U = zeros(size(U)); %my_edit: initialize gradient accumulator
 for jj = 1 : ne
 gradient_U = gradient_U + Xs(error_indx(jj),:)' * w' .* Ys(error_indx(jj),:); %my_edit: accumulate gradient over error samples
 end
 gradient_U = L * gradient_U/ne + B * 2*(U-V) ; 
 else %my_edit: handle batch with no margin violations
 gradient_w = 2*w; %my_edit
 gradient_U = B * 2*(U-V); %my_edit
 end 
 w = w - learn_rate_w * gradient_w; 
 U = U - learn_rate_u * gradient_U; 
 end
 obj_new = w'*w + L * mean(max(0,1-Ys.*(Xs*U*w))) + B * norm(U-V,'fro'); %my_edit: syntax fix for * and .*
 overall_obj(iter) = obj_new;
 if (obj_old - obj_new) < 10^-6 % convergence criteria
 return; %my_edit: return upon convergence
 end
 obj_old = obj_new; %my_edit: update obj_old across iterations
end
end
