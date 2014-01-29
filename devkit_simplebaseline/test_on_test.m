function [conf,image_names] = test_on_test(model, reset)
    globals;
    
    cache_file = 'test_data.mat';
    cache_feat_file = 'test_feat.mat';
    
    % Extract annotations
    if exist(cache_file, 'file') && (reset < 2)
        load(cache_file);
    else
        % load list of test files and list of classes
        load('filelists.mat', 'test_data');

        fprintf('loading annotations...');
        annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), test_data.annotations', 'UniformOutput', false);
        image_list = test_data.images;
        fprintf('done\n');
        
        save(cache_file, 'annotations', 'image_list');
    end
    
    % Extract features
    if exist(cache_feat_file, 'file') && ~reset
        load(cache_feat_file);
    else
        feats = extract_feats(annotations, image_list);
        save(cache_feat_file, 'feats');
    end

    % Extract labels/features
    fake_labels = zeros(size(feats,1),1);
    [~,~,conf] = predict(fake_labels, sparse(double(feats)), model, '-q');
    if model.Label(1) ==0
        conf = -conf;
    end
        
    % get the list of image names for test data as shown below
    image_names = cellfun(@(x) x.annotation.filename, annotations, 'UniformOutput', false);
end

