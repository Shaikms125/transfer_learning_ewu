function [acc, y_pred, Xs_coral] = CORAL(Xs, Ys, Xt, Yt)
% CORAL aligns source features to the target covariance and classifies the
% target samples with Euclidean 1-nearest-neighbour.
%
% Inputs:
%   Xs : source feature matrix, ns-by-m
%   Ys : source label vector, ns-by-1
%   Xt : target feature matrix, nt-by-m
%   Yt : target label vector, nt-by-1
%
% Outputs:
%   acc      : target classification accuracy in [0, 1]
%   y_pred   : predicted target labels, nt-by-1
%   Xs_coral : CORAL-aligned source feature matrix, ns-by-m

    Xs = double(Xs);
    Xt = double(Xt);
    Ys = double(Ys(:));
    Yt = double(Yt(:));

    cov_src = cov(Xs) + eye(size(Xs, 2));
    cov_tar = cov(Xt) + eye(size(Xt, 2));
    A_coral = cov_src^(-1/2) * cov_tar^(1/2);
    Xs_coral = Xs * A_coral;

    classifier_dir = fullfile(fileparts(mfilename('fullpath')), ...
        'base', 'classifier');
    if exist('jdknn', 'file') ~= 2
        addpath(classifier_dir);
    end

    [y_pred, acc] = jdknn(Xs_coral, Ys, Xt, Yt, 1);
end
