%function [Y_tar_predition] = DMDAJFR(X_src,Y_src,X_tar,Y_tar_pseudo,parameter)
function [acc, Y_tar_predition] = DMDAJFR(X_src,Y_src,X_tar,Y_tar,dim)
%Inputs:
    %%% X_src  :source feature matrix, n_src * m
    %%% Y_src  :source label vector, n_src * 1
    %%% X_tar  :target feature matrix, n_tar * m
    %%% Y_tar  :target label vector, n_tar * 1 (true target labels, used ONLY to compute the accuracy)
    %%% dim    :kept so DMDAJFR can be called exactly like STL(X_src,y_src,X_tar,y_tar,dim); not used, GMDA keeps the full feature dimension
%outputs:
    %%% acc             :accuracy on the target domain (fraction, same convention as STL)
    %%% Y_tar_predition :predicted target labels

%%% parameter.lambda;
 %parameter.beta;
 %parameter.noises;
 %parameter.layers;
 %my edit
 parameter.lambda  = 1;    % ridge on mapping W (bias excluded) — numerical stabilizer
 parameter.beta    = 1;    % weight of MMD domain-alignment term vs. mDA reconstruction
 parameter.noises  = 0.2;  % mDA corruption probability (0–1)
 parameter.layers  = 5;    % stacked mDA + pseudo-label refinement rounds

 if nargin < 5
     dim = [];  %#ok<NASGU>  % unused by GMDA (m -> m feature mapping)
 end

 %initial target pseudo labels: 1-NN on the raw features (wrapper addition, not in the og code)
 [Y_tar_pseudo] = Pseudolable(X_src, X_tar, Y_src);

 %ensure every source class appears in the initial pseudo labels: GMDA builds one
 %MMD block per class and divides by zero if a class is empty (GMDA.m:34-43)
 for c = setdiff(unique(Y_src), unique(Y_tar_pseudo))'
     d_c = pdist2(X_tar, mean(X_src(Y_src==c,:),1));           %distance to the class-c source centroid
     n_c = max(1, round(sum(Y_src==c)/size(Y_src,1)*size(X_tar,1)));
     u = unique(Y_src);
     cnt = arrayfun(@(t) sum(Y_tar_pseudo==t), u);
     [cmax, jmax] = max(cnt);
     if cmax < 2, continue; end                                %cannot donate without becoming empty
     cand = find(Y_tar_pseudo==u(jmax));
     [~,k] = mink(d_c(cand), min(n_c, cmax-1));                %donate from the largest class, keeping it non-empty
     Y_tar_pseudo(cand(k)) = c;
 end
 %my edit

  [X_src_globalx,X_tar_globalx,W] =GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
  [X_src_localx,X_tar_localx,Wc] =GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
  %X_src_new,: source joint feature representations;
   %X_tar_new : target joint feature representations,
  X_src_new=[X_src_globalx,X_src_localx];
  X_tar_new=[X_tar_globalx,X_tar_localx];
  [Y_tar_predition]= Pseudolable(X_src_new,  X_tar_new, Y_src);
  acc = mean(Y_tar_predition(:)==Y_tar(:));   %accuracy, same convention as STL.m %my edit
end
