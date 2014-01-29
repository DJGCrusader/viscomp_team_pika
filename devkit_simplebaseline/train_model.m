function model = train_model(class, reset)
    globals;
    
    train_cache_file = 'train_data.mat';
    train_cache_feat_file = 'train_feat.mat';
    addpath(genpath('liblinear-1.4'));
    
    % Extract annotations
    if exist(train_cache_file, 'file') && (reset < 2)
        load(train_cache_file);
    else
        % load list of train files and list of classes
        load('filelists.mat', 'train_data');

        fprintf('loading annotations...');
        annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
        image_list = train_data.images;
        fprintf('done\n');
        
        save(train_cache_file, 'annotations', 'image_list');
    end
    
    % Extract features
    if exist(train_cache_feat_file, 'file') && ~reset
        load(train_cache_feat_file);
    else
        %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!extract_feats!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        feats= extract_feats(annotations, image_list);
        
        save(train_cache_feat_file, 'feats');
    end

    % Extract labels/features
    labels = cellfun(@(x) str2double(x.annotation.classes.(class)), annotations);
    
    %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!C PARAMETER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model = train(labels, sparse(double(feats)), ['-s 2 -c 50 -B 0 -q']);
end