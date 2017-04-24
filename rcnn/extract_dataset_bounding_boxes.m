function extract_dataset_bounding_boxes(image_set)

    switch image_set
        case 'flickr8k'
            dataset = 'flickr8k';
            image_dir_name = 'images';
        case 'mscoco/train2014'
            dataset = 'mscoco';
            image_dir_name = 'train2014';
        case 'mscoco/val2014'
            dataset = 'mscoco';
            image_dir_name = 'val2014';

        case 'mscoco/train2014/0'
            dataset = 'mscoco';
            image_dir_name = 'train2014_0';
        case 'mscoco/train2014/1'
            dataset = 'mscoco';
            image_dir_name = 'train2014_1';
        case 'mscoco/train2014/2'
            dataset = 'mscoco';
            image_dir_name = 'train2014_2';
        case 'mscoco/train2014/3'
            dataset = 'mscoco';
            image_dir_name = 'train2014_3';
        case 'mscoco/train2014/4'
            dataset = 'mscoco';
            image_dir_name = 'train2014_4';

        case 'mscoco/val2014/0'
            dataset = 'mscoco';
            image_dir_name = 'val2014_0';
        case 'mscoco/val2014/1'
            dataset = 'mscoco';
            image_dir_name = 'val2014_1';
        case 'mscoco/val2014/2'
            dataset = 'mscoco';
            image_dir_name = 'val2014_2';
        case 'mscoco/val2014/3'
            dataset = 'mscoco';
            image_dir_name = 'val2014_3';
        case 'mscoco/val2014/4'
            dataset = 'mscoco';
            image_dir_name = 'val2014_4';

        otherwise
            error(['Unknown image set ' image_set]);
    end

    % Initialize R-CNN model
    rcnn_model_file = './data/rcnn_models/ilsvrc2013/rcnn_model.mat';
    rcnn_model = rcnn_load_model(rcnn_model_file, true);
    
    dataset_root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'datasets', dataset);
    image_root = fullfile(dataset_root, image_dir_name);
    % Create file to store boxes
    bbox_file = [image_root '_rcnn_boxes.txt'];
    % If file exists, get files that were already processed
    if exist(bbox_file, 'file')
        M = readtable(bbox_file, 'ReadVariableNames', false);
        processed_files = M{:, 1};
    end
    f = fopen(bbox_file, 'a');
    d = dir(image_root);
    % Remove . and ..
    d = d(3:end);
    % Get full image paths
    image_paths = cell(numel(d), 1);
    for j=1:numel(d)
        image_paths{j} = fullfile(image_root, d(j).name);
    end

    for j=1:numel(image_paths)
        image_path = image_paths{j};
        % Skip if the image was processed
        if exist('processed_files', 'var') && any(ismember(image_path, processed_files))
            fprintf('Already processed %s, skipping\n', image_path);
            continue;
        else
            fprintf('Working on image %s\n', image_path);
        end
        boxes = get_bounding_boxes(image_path, rcnn_model);
        % Take transpose for easy writing
        boxes = boxes';
        fprintf(f, '%s,', image_path);
        for k=1:numel(boxes)
            fprintf(f, '%d,', boxes(k));
        end
        fprintf(f, '\n');
        % Print progress
        fprintf('Finished %d/%d images\n', j, numel(image_paths));
    end

end

function best_det_boxes = get_bounding_boxes(image_path, rcnn_model)

    % Load image
    im = imread(image_path);
    if ndims(im) == 2
        im = cat(3, im, im, im);
    end
    % Get values on all object proposals with > 0 confidence in weird format
    dets = rcnn_detect(im, rcnn_model, -Inf);
    % Get detections in normal format
    all_dets = [];
    for i = 1:length(dets)
      all_dets = cat(1, all_dets, ...
          [i * ones(size(dets{i}, 1), 1) dets{i}]);
    end

    % Extract 1-indexed coordinates (x_left, y_top, x_right, y_bottom) of top 19 bounding boxes
    [~, ord] = sort(all_dets(:,end), 'descend');
    best_det_boxes = round(all_dets(ord(1:19), 2:5));
    % Add full image box
    best_det_boxes(end+1, :) = [1, 1, size(im, 2), size(im, 1)];

end