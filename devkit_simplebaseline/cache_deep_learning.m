function [feats, images] = cache_deep_learning()
    globals;

    cache_deep_file = 'deep_feats.mat';
    
    if exist(cache_deep_file, 'file')
        load(cache_deep_file);
    else
        fprintf('caching some of deep learning features...');
        
        feat_files = dir([feature_folder '*.mat']);
        feats = []; images= [];
        for i = 1:length(feat_files)
            if mod(i,10) == 0
                fprintf('.');
            end
            load([feature_folder feat_files(i).name]);
            
            feats = [feats; scores];
            images = [images, image_list];
        end
        save(cache_deep_file, 'feats', 'images');
        
        fprintf('done.\n');
    end
end