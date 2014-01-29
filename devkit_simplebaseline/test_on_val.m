function ap = test_on_val(class, model, reset)
    globals;

    cache_file = 'val_data.mat';
    cache_feat_file = 'val_feat.mat';
    
    
    % Extract annotations
    if exist(cache_file, 'file') && (reset < 2)
        load(cache_file);
    else
        % load list of validation files and list of classes
        load('filelists.mat', 'val_data');

        fprintf('loading annotations...');
        annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), val_data.annotations', 'UniformOutput', false);
        image_list = val_data.images;
        fprintf('done\n');
        
        save(cache_file, 'annotations', 'image_list');
    end
    
    % Extract features
    if exist(cache_feat_file, 'file') && ~reset
        load(cache_feat_file);
    else
        feats= extract_feats(annotations, image_list);
        save(cache_feat_file, 'feats');
    end

    % Extract labels/features
    labels = cellfun(@(x) str2double(x.annotation.classes.(class)), annotations);        
    [~,~,conf] = predict(labels, sparse(double(feats)), model, '-q');
    if model.Label(1) ==0
        conf = -conf;
    end
        
    ap = computeAP(conf, labels, 1)*100;
end

