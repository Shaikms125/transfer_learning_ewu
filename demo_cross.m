clear;

addpath(genpath('data/'))
file_name = 'OPP';         %% 'OPP' | 'DSADS' | 'PAMAP'
file_name2 = 'DSADS'; %my change
dim = 30;                  %% < 81
% tasks = { 'PAMAP', 'OPP',   2, 1;      % PAMAP2 -> OPP
%           'DSADS', 'PAMAP', 1, 2;      % DSADS  -> PAMAP2
%           'OPP',   'DSADS', 1, 1};    % OPP    -> DSADS

for i = 1 : 1             
    for j = 1 : 1
        if strcmp(file_name,file_name2) && i == j %my change
            continue;
        end
        %% Get source and target domain according to dataset
        [src_domain] = load_data(file_name,i); %my change
        [tar_domain] = load_data(file_name2,j); %my change
        n1 = size(src_domain,1);
        n2 = size(tar_domain,1);
        seq1 = randperm(n1);
        seq2 = randperm(n2);
        r_source_domain = src_domain(seq1,:);
        r_target_domain = tar_domain(seq2,:);
        x_src = r_source_domain(:,1:end - 1);
        y_src = r_source_domain(:,end);
        x_tar = r_target_domain(:,1:end - 1);
        y_tar = r_target_domain(:,end);
        %               
        x_src = zscore(x_src);
        x_tar = zscore(x_tar);

        acc_stl = STL(x_src,y_src,x_tar,y_tar,dim);
        fprintf('Acc: %.2f\n',acc_stl);

        % %TCA Options for TCA (intra-class transfer in STL)
        % tca_options = [];                   %my_change
        % tca_options.lambda      = 0.1;      %my_change
        % tca_options.dim         = dim;      %my_change
        % tca_options.kernel_type = 'primal'; %my_change  % 'primal' | 'linear' | 'rbf'
        % tca_options.gamma       = 0.1;      %my_change   % only used by the 'rbf' kernel
        % 
        % [x_src_tca, x_tar_tca] = TCA(x_src, x_tar, tca_options);
        % [y_pred_tca, acc_tca] = jdknn(x_src_tca, y_src, x_tar_tca, y_tar, 1);
        % fprintf('TCA + 1NN Acc: %.2f%%\n', acc_tca * 100);

    end
end


function [src_domain] = load_data(file_name,src_index)
data = []; 
col_sets = {[1:81],[82:162],[163:243],[244:324],[325:405]};
src_set = col_sets{src_index};
%tar_set = col_sets{tar_index};
switch(file_name)
    case 'DSADS'
        load('data/cross_dsads.mat');
        data = data_dsads;
        % index = data(:,end - 1) == 1 | data(:,end - 1) == 2 | data(:,end - 1) == 3 ...
        % | data(:,end - 1) == 4 | data(:,end - 1) == 5 | data(:,end - 1) == 6 ...
        % | data(:,end - 1) == 7 | data(:,end - 1) == 9 | data(:,end - 1) == 12 ...
        % | data(:,end - 1) == 18;
        % % Original filter kept only selected activity IDs.

        index = true(size(data,1),1); %my_change: keep all DSADS rows

        %src_domain = data(index,[src_set,end - 1]);
        %tar_domain = data(index,[tar_set,end - 1]);
        data = data(index,:); %my_change
        src_domain = data(:,[src_set,end]); %my_change
    case 'OPP'
        load('data/cross_opp.mat');
        
        % index = data_opp_loco(:,end) ~= 4;
        % Original filter excluded subject 4 because the last column is the subject ID.
        index = true(size(data_opp,1),1); %my_change: keep all OPP rows and subjects

        data = data_opp(index,:); %my_change

        %data = feature_norm(index,:);
        
        %src_domain = data(:,[src_set,end - 2]);
        %tar_domain = data(:,[tar_set,end - 2]);
        src_domain = data(:,[src_set,end]); %my_change
    case 'PAMAP'
        load('data/cross_pamap.mat');
        data = data_pamap;
        % index = data(:,end - 1) == 1 | data(:,end - 1) == 2 | data(:,end - 1) == 3 ...
        % | data(:,end - 1) == 4 | data(:,end - 1) == 5 | data(:,end - 1) == 6 ...
        % | data(:,end - 1) == 7 | data(:,end - 1) == 9 | data(:,end - 1) == 12 ...
        % | data(:,end - 1) == 13;

        % Original filter kept only selected activity IDs.
        index = true(size(data,1),1); %my_change: keep all PAMAP rows

        %src_domain = data(index,[src_set,end - 1]);
        %tar_domain = data(index,[tar_set,end - 1]);
        pamap_to_common = [4 3 1 2];
        data(:,end) = pamap_to_common(data(:,end)); %my_change
        src_domain = data(:,[src_set,end]); %my_change
end
end
