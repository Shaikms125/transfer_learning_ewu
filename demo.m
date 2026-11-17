clear;
addpath(genpath('base/classifier/'));
addpath(genpath('data/'))
file_name = 'OPP';         %% 'OPP' | 'DSADS' | 'PAMAP'
dim = 30;                  %% < 81
%% Use i and j to denote the different body positions in each dataset
% 'DSADS', 2, 3;   % RA  -> LA
% 'DSADS', 4, 5;   % RL  -> LL
% 'OPP',   2, 4;   % RUA -> LUA
% 'OPP',   3, 5;   % RLA -> LLA
% 'DSADS', 2, 1;   % RA  -> T
% 'PAMAP', 1, 2;   % H   -> C
% 'OPP',   3, 1;   % RLA -> T
% 'OPP',   2, 1;   % RUA -> T


for i = 2 : 2             
    for j = 1 : 1
        if i == j
            continue;
        end
        %% Get source and target domain according to dataset
        [src_domain,tar_domain] = load_data(file_name,i,j);
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

            
            %TCA
            % Options for TCA (intra-class transfer in STL)
            % tca_options = [];                   %my_change
            % tca_options.lambda      = 0.1;      %my_change
            % tca_options.dim         = dim;      %my_change
            % tca_options.kernel_type = 'primal'; %my_change  % 'primal' | 'linear' | 'rbf'
            % tca_options.gamma       = 0.1;      %my_change   % only used by the 'rbf' kernel
            % 
            % [x_src_tca, x_tar_tca] = TCA(x_src, x_tar, tca_options);
            % [y_pred_tca, acc_tca] = jdknn(x_src_tca, y_src, x_tar_tca, y_tar, 1);
            % fprintf('TCA + 1NN Acc: %.2f%%\n', acc_tca * 100);


            %dapca
            % [~, ~, x_src_dapca, x_tar_dapca] = DAPCA(x_src, y_src, x_tar, dim);
            % 
            % learner = templateSVM('KernelFunction', 'linear', 'Standardize', false);
            % model = fitcecoc(x_src_dapca, y_src, 'Learners', learner);
            % y_pred_dapca = predict(model, x_tar_dapca);
            % 
            % acc_dapca = mean(y_pred_dapca == y_tar) * 100;
            % fprintf('DAPCA Acc: %.2f\n', acc_dapca);
            %dapca

            % x_src_coral = CORAL(x_src, x_tar);
            % 
            % learner = templateSVM('KernelFunction','linear','Standardize',false);
            % model = fitcecoc(x_src_coral, y_src, 'Learners', learner);
            % 
            % y_pred_coral = predict(model, x_tar);
            % acc_coral = mean(y_pred_coral == y_tar) * 100;
            % 
            % fprintf('CORAL Acc: %.2f\n', acc_coral);

           

    end
end


function [src_domain,tar_domain] = load_data(file_name,src_index,tar_index)
    data = []; 
    col_sets = {[1:81],[82:162],[163:243],[244:324],[325:405]};
    src_set = col_sets{src_index};
    tar_set = col_sets{tar_index};
    switch(file_name)
        case 'DSADS'
            load('data/dsads.mat');
            data = data_dsads;
            % index = data(:,end - 1) == 1 | data(:,end - 1) == 2 | data(:,end - 1) == 3 ...
            % | data(:,end - 1) == 4 | data(:,end - 1) == 5 | data(:,end - 1) == 6 ...
            % | data(:,end - 1) == 7 | data(:,end - 1) == 9 | data(:,end - 1) == 12 ...
            % | data(:,end - 1) == 18;
            % Original filter kept only selected activity IDs.

            index = true(size(data,1),1); %my_change: keep all DSADS rows

            src_domain = data(index,[src_set,end - 1]);
            tar_domain = data(index,[tar_set,end - 1]);
        case 'OPP'
            load('data/opp_loco.mat');
            % index = data_opp_loco(:,end) ~= 4;
            % Original filter excluded subject 4 because the last column is the subject ID.
            index = true(size(data_opp_loco,1),1); %my_change: keep all OPP rows and subjects

            %data = feature_norm(index,:);
            data = data_opp_loco(index,:);          % my_change
            src_domain = data(:,[src_set,end - 2]);
            tar_domain = data(:,[tar_set,end - 2]);
        case 'PAMAP'
            load('data/pamap.mat');
            data = data_pamap;
            % index = data(:,end - 1) == 1 | data(:,end - 1) == 2 | data(:,end - 1) == 3 ...
            % | data(:,end - 1) == 4 | data(:,end - 1) == 5 | data(:,end - 1) == 6 ...
            % | data(:,end - 1) == 7 | data(:,end - 1) == 9 | data(:,end - 1) == 12 ...
            % | data(:,end - 1) == 13;

            % Original filter kept only selected activity IDs.
            index = true(size(data,1),1); %my_change: keep all PAMAP rows

            src_domain = data(index,[src_set,end - 1]);
            tar_domain = data(index,[tar_set,end - 1]);
    end
end
