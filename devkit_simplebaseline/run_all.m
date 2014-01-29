function run_all(reset)
% reset - flag to use cache or not
%            0: use cache
%            1: reset feature computation
%            2: reset features+annotations

    addpath(genpath('liblinear-1.94'));
    
    load('classes.mat');

    upload_file_name = 'simple_result.txt';
    total = 0;
    % For each class
    for i = 1:length(classes)
        % Train a model
        model = train_model(classes{i}, reset);
        
        % Test on validation
        ap = test_on_val(classes{i}, model, reset);
        fprintf('class: %s, average precision: %.02f%%\n', classes{i}, ap);
        total = total+ap;
        % Test on testset
        [conf(:,i), image_names] = test_on_test(model, reset);
    end
    
    disp(['Average: ', num2str(total/length(classes)), '%']);
    
    % Create upload file
    createUploadFile(image_names, classes, conf, upload_file_name);
end