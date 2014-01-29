function ret_feats = extract_feats(annotations, image_names)
    globals;
    addpath(genpath('feature-extraction'));

     % LOAD CACHE DEEP LEARNING FEATURES
    [feats, images] = cache_deep_learning();
    
    fprintf('Extracting features..');
    for i = 1:length(image_names)        
        tic;
        if mod(i,100)==0
            fprintf('.');
        end
        
        % COMPUTE YOUR OWN FEATURE
        %
        % % your own feature extraction code
        % feat1 = YOUR OWN WAY
        %
        feat_custom = [];
        im = imread([dataset_folder image_names{i}]);
        
        
        % Load the configuration and set dictionary size to n
        feature = 'hog3x3';
        c = conf();
        c.verbosity = 1;
        c.feature_config.(feature).dictionary_size = 30;
        c.feature_config.(feature).num_desc = 2e4;
        c.feature_config.(feature).descPerImage = 250;
        
        feat1 = extract_hog3x3(im,c);
        feat_custom(1) = feat1(1);
        
        % LOAD PRE-COMPUTED DEEP LEARNING FEATURE        
        ind=find(~cellfun(@isempty, strfind(images, image_names{i})));
        assert(length(ind)==1);
        
        ret_feats(i,:) = [feat_custom, feats(ind,:)];
    end    
    fprintf('Done.\n');
end