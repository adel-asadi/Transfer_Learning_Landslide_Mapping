% Load the input image and its corresponding binary label
img = Post_event_image;
label = double(rec_recl_clip111);

% get the size of the input image
[img_height, img_width, ~] = size(img);

% define the tile size and no overlap
tile_size = 224;
no_overlap = 0;

% calculate the number of tiles in the x and y directions
num_tiles_x = floor((img_width - tile_size)/(tile_size - no_overlap)) + 1;
num_tiles_y = floor((img_height - tile_size)/(tile_size - no_overlap)) + 1;

% initialize empty cell arrays to store the tiles and labels
tiles = cell(num_tiles_x * num_tiles_y, 1);
labels = cell(num_tiles_x * num_tiles_y, 1);

% extract the tiles and corresponding labels
tile_index = 1;
for y = 1:num_tiles_y
    for x = 1:num_tiles_x
        % calculate the tile's starting and ending x and y coordinates
        start_x = (x-1)*(tile_size - no_overlap) + 1;
        end_x = start_x + tile_size - 1;
        start_y = (y-1)*(tile_size - no_overlap) + 1;
        end_y = start_y + tile_size - 1;
        
        % extract the tile and corresponding label
        tile = img(start_y:end_y, start_x:end_x, :);
        tile_label = label(start_y:end_y, start_x:end_x, :);
        
        % store the tile and corresponding label in the cell arrays
        tiles{tile_index} = tile;
        labels{tile_index} = tile_label;
        
        tile_index = tile_index + 1;
    end
end

% randomly divide the tiles and corresponding labels into training, validation, and testing sets
num_tiles = length(tiles);
random_indices = randperm(num_tiles);

train_indices = random_indices(1:round(0.7*num_tiles));
val_indices = random_indices(round(0.7*num_tiles)+1:round(0.9*num_tiles));
test_indices = random_indices(round(0.9*num_tiles)+1:end);

train_tiles = tiles(train_indices);
train_labels = labels(train_indices);
val_tiles = tiles(val_indices);
val_labels = labels(val_indices);
test_tiles = tiles(test_indices);
test_labels = labels(test_indices);

%%

for i=34:35
tile_orig=augm_origin_tile{i};
a = zeros(size(tile_orig),'uint8');
    a(:,:,1) = tile_orig(:,:,1);
    a(:,:,2) = tile_orig(:,:,2);
    a(:,:,3) = tile_orig(:,:,3);
    % Display the tile and label
    subplot(7, 10, i);
    imshow(a);
end

%%

% initialize empty cell arrays to store the augmented tiles and labels
%augmented_tiles = cell(num_augmentations * length(train_tiles), 1);
%augmented_labels = cell(num_augmentations * length(train_labels), 1);

% loop over the training tiles and labels
aug_index = 1;
origin_index = 1;
for i = 1:length(train_tiles)
    % get the tile and corresponding label
    tile = train_tiles{i};
    label = train_labels{i};
    % check if the tile contains the minority class (1 on the labels)
    if any(label(:) == 1)
        % perform data augmentation on the tile
            % rotate the tile by 90, 180, and 270 degrees
            augmented_tile_1 = imrotate(tile, 90);
            augmented_label_1 = imrotate(label, 90);
            augmented_tile_2 = imrotate(tile, 180);
            augmented_label_2 = imrotate(label, 180);
            augmented_tile_3 = imrotate(tile, 270);
            augmented_label_3 = imrotate(label, 270);
            
            % flip the tile horizontally and vertically
            augmented_tile_4 = flip(tile, 2);
            augmented_label_4 = flip(label, 2);
            augmented_tile_5 = flip(tile, 1);
            augmented_label_5 = flip(label, 1);
            
            % apply brightness change (higher and lower)
            augmented_tile_6 = jitterColorHSV(uint8(augmented_tile_1),Brightness=[0.1 0.3]);
            augmented_tile_7 = jitterColorHSV(uint8(augmented_tile_2),Brightness=[-0.3 -0.1]);
            
            % apply contrast change (higher and lower)
            augmented_tile_8 = jitterColorHSV(augmented_tile_3,Contrast=[1.2 1.4]);
            augmented_tile_9 = jitterColorHSV(augmented_tile_4,Contrast=[0.6 0.8]);
            
            % apply randomized Gaussian blur
            sigma = 1+5*rand;
            augmented_tile_10 = imgaussfilt(augmented_tile_5, sigma);

            % apply randomized Gaussian noise
            augmented_tile_11 = imnoise(uint8(tile),"gaussian");
            
            % store the augmented tiles and corresponding labels in the cell arrays
            augmented_tiles{aug_index} = augmented_tile_1;
            augmented_labels{aug_index} = augmented_label_1;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_2;
            augmented_labels{aug_index} = augmented_label_2;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_3;
            augmented_labels{aug_index} = augmented_label_3;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_4;
            augmented_labels{aug_index} = augmented_label_4;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_5;
            augmented_labels{aug_index} = augmented_label_5;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_6;
            augmented_labels{aug_index} = augmented_label_1;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_7;
            augmented_labels{aug_index} = augmented_label_2;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_8;
            augmented_labels{aug_index} = augmented_label_3;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_9;
            augmented_labels{aug_index} = augmented_label_4;
            aug_index = aug_index + 1;
            
            augmented_tiles{aug_index} = augmented_tile_10;
            augmented_labels{aug_index} = augmented_label_5;
            aug_index = aug_index + 1;

            augmented_tiles{aug_index} = augmented_tile_11;
            augmented_labels{aug_index} = label;
            aug_index = aug_index + 1;

            augm_origin_tile{origin_index}=tile;
            augm_origin_label{origin_index}=label;
            origin_index=origin_index+1;
    end
end

% remove any empty cells in the augmented_tiles and augmented_labels cell arrays
augmented_tiles = augmented_tiles(~cellfun('isempty',augmented_tiles));
augmented_labels = augmented_labels(~cellfun('isempty',augmented_labels));

% combine the original training tiles and labels with the augmented tiles and labels
train_tiles_combined = [train_tiles; augmented_tiles'];
train_labels_combined = [train_labels; augmented_labels'];

% generate random indices to shuffle the training data
train_indices = randperm(length(train_tiles_combined));

% shuffle the training data
train_tiles_combined = train_tiles_combined(train_indices);
train_labels_combined = train_labels_combined(train_indices);

%%

% Display a few of the augmented image tiles and their labels

augmentation_techniques = ["90 deg Rotation", "180 deg Rotation", "270 deg Rotation", "Horizontal Flip", "Vertical Flip", "Brightening", "Darkening", "Higher Contrast", "Lower Contrast", "Gaussian Blur", "Gaussian Noise"];

num_samples = 10;
j_ind=[3,5,7,9,11,13,15,17,19,21,23];
for i = 1:num_samples
    figure;
    % Select a random index
    idx = randi(size(augm_origin_tile, 2));
    % Select a random augmented tile and label
    tile_orig = augm_origin_tile{idx};
    label_orig = augm_origin_label{idx};

    a = zeros(size(tile_orig),'uint8');
    a(:,:,1) = tile_orig(:,:,1);
    a(:,:,2) = tile_orig(:,:,2);
    a(:,:,3) = tile_orig(:,:,3);
    % Display the tile and label
    subplot(6, 4, 1);
    imshow(a);
    title('Original Tile');

    subplot(6, 4, 2);
    imagesc(label_orig);
    title('Original Label');
    colorbar

    for j=1:11
    tile_augm{j} = augmented_tiles{((idx-1)*11)+j};
    label_augm{j} = augmented_labels{((idx-1)*11)+j};

    a = zeros(size(tile_augm{j}),'uint8');
    a(:,:,1) = tile_augm{j}(:,:,1);
    a(:,:,2) = tile_augm{j}(:,:,2);
    a(:,:,3) = tile_augm{j}(:,:,3);
    % Display the tile and label
    subplot(6, 4, j_ind(j));
    imshow(a);
    title(augmentation_techniques(j));
    
    subplot(6, 4, (j_ind(j)+1));
    imagesc(label_augm{j});
    title('Augmented Label');
    colorbar
    end
end

%%

% initialize empty cell arrays to store the augmented tiles and labels
%augmented_tiles = cell(num_augmentations * length(train_tiles), 1);
%augmented_labels = cell(num_augmentations * length(train_labels), 1);

% loop over the validation tiles and labels
aug_index = 1;
origin_index = 1;
for i = 1:length(val_tiles)
    % get the tile and corresponding label
    tile = val_tiles{i};
    label = val_labels{i};
    % check if the tile contains the minority class (1 on the labels)
    if any(label(:) == 1)
        % perform data augmentation on the tile
            % rotate the tile by 90, 180, and 270 degrees
            augmented_tile_1 = imrotate(tile, 90);
            augmented_label_1 = imrotate(label, 90);
            augmented_tile_2 = imrotate(tile, 180);
            augmented_label_2 = imrotate(label, 180);
            augmented_tile_3 = imrotate(tile, 270);
            augmented_label_3 = imrotate(label, 270);
            
            % flip the tile horizontally and vertically
            augmented_tile_4 = flip(tile, 2);
            augmented_label_4 = flip(label, 2);
            augmented_tile_5 = flip(tile, 1);
            augmented_label_5 = flip(label, 1);
            
            % apply brightness change (higher and lower)
            augmented_tile_6 = jitterColorHSV(uint8(augmented_tile_1),Brightness=[0.1 0.3]);
            augmented_tile_7 = jitterColorHSV(uint8(augmented_tile_2),Brightness=[-0.3 -0.1]);
            
            % apply contrast change (higher and lower)
            augmented_tile_8 = jitterColorHSV(augmented_tile_3,Contrast=[1.2 1.4]);
            augmented_tile_9 = jitterColorHSV(augmented_tile_4,Contrast=[0.6 0.8]);
            
            % apply randomized Gaussian blur
            sigma = 1+5*rand;
            augmented_tile_10 = imgaussfilt(augmented_tile_5, sigma);

            % apply randomized Gaussian noise
            augmented_tile_11 = imnoise(uint8(tile),"gaussian");
            
            % store the augmented tiles and corresponding labels in the cell arrays
            augmented_tiles_val{aug_index} = augmented_tile_1;
            augmented_labels_val{aug_index} = augmented_label_1;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_2;
            augmented_labels_val{aug_index} = augmented_label_2;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_3;
            augmented_labels_val{aug_index} = augmented_label_3;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_4;
            augmented_labels_val{aug_index} = augmented_label_4;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_5;
            augmented_labels_val{aug_index} = augmented_label_5;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_6;
            augmented_labels_val{aug_index} = augmented_label_1;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_7;
            augmented_labels_val{aug_index} = augmented_label_2;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_8;
            augmented_labels_val{aug_index} = augmented_label_3;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_9;
            augmented_labels_val{aug_index} = augmented_label_4;
            aug_index = aug_index + 1;
            
            augmented_tiles_val{aug_index} = augmented_tile_10;
            augmented_labels_val{aug_index} = augmented_label_5;
            aug_index = aug_index + 1;

            augmented_tiles_val{aug_index} = augmented_tile_11;
            augmented_labels_val{aug_index} = label;
            aug_index = aug_index + 1;

            augm_origin_tile_val{origin_index}=tile;
            augm_origin_label_val{origin_index}=label;
            origin_index=origin_index+1;
    end
end

% remove any empty cells in the augmented_tiles and augmented_labels cell arrays
augmented_tiles_val = augmented_tiles_val(~cellfun('isempty',augmented_tiles_val));
augmented_labels_val = augmented_labels_val(~cellfun('isempty',augmented_labels_val));

% combine the original training tiles and labels with the augmented tiles and labels
validation_tiles_combined = [val_tiles; augmented_tiles_val'];
validation_labels_combined = [val_labels; augmented_labels_val'];

% generate random indices to shuffle the training data
val_indices = randperm(length(validation_tiles_combined));

% shuffle the training data
validation_tiles_combined = validation_tiles_combined(val_indices);
validation_labels_combined = validation_labels_combined(val_indices);

%%

% Display a few of the augmented image tiles and their labels

augmentation_techniques = ["90 deg Rotation", "180 deg Rotation", "270 deg Rotation", "Horizontal Flip", "Vertical Flip", "Brightening", "Darkening", "Higher Contrast", "Lower Contrast", "Gaussian Blur", "Gaussian Noise"];

num_samples = 1;
j_ind=[3,5,7,9,11,13,15,17,19,21,23];
for i = 1:num_samples
    figure;
    % Select a random index
    idx = randi(size(augm_origin_tile_val, 2));
    % Select a random augmented tile and label
    tile_orig = augm_origin_tile_val{idx};
    label_orig = augm_origin_label_val{idx};

    a = zeros(size(tile_orig),'uint8');
    a(:,:,1) = tile_orig(:,:,1);
    a(:,:,2) = tile_orig(:,:,2);
    a(:,:,3) = tile_orig(:,:,3);
    % Display the tile and label
    subplot(6, 4, 1);
    imshow(a);
    title('Original Tile');

    subplot(6, 4, 2);
    imagesc(label_orig);
    title('Original Label');
    colorbar

    for j=1:11
    tile_augm{j} = augmented_tiles_val{((idx-1)*11)+j};
    label_augm{j} = augmented_labels_val{((idx-1)*11)+j};

    a = zeros(size(tile_augm{j}),'uint8');
    a(:,:,1) = tile_augm{j}(:,:,1);
    a(:,:,2) = tile_augm{j}(:,:,2);
    a(:,:,3) = tile_augm{j}(:,:,3);
    % Display the tile and label
    subplot(6, 4, j_ind(j));
    imshow(a);
    title(augmentation_techniques(j));
    
    subplot(6, 4, (j_ind(j)+1));
    imagesc(label_augm{j});
    title('Augmented Label');
    colorbar
    end
end

%%

% Display the sizes of the resulting cells
fprintf('Training set size: %d tiles, %d labels\n', numel(train_tiles_combined), numel(train_labels_combined));
fprintf('Validation set size: %d tiles, %d labels\n', numel(validation_tiles_combined), numel(validation_labels_combined));
fprintf('Testing set size: %d tiles, %d labels\n', numel(test_tiles), numel(test_labels));

%%

for k=1:7049
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Training_tiles\train_tiles',k);
    imwrite(train_tiles_combined{k},filename, 'BitDepth', 16);
end

for k=1:2179
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Validation_tiles\val_tiles',k);
    imwrite(validation_tiles_combined{k},filename, 'BitDepth', 16);
end

for k=1:336
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Testing_tiles\test_tiles',k);
    imwrite(test_tiles{k},filename, 'BitDepth', 16);
end

%%

for k=1:7049
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Training_labels\train_labels',k);
    imwrite(train_labels_combined{k},filename, 'BitDepth', 1, 'Compression', 'none');
end

for k=1:2179
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Validation_labels\val_labels',k);
    imwrite(validation_labels_combined{k},filename, 'BitDepth', 1, 'Compression', 'none');
end

for k=1:336
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Testing_labels\test_labels',k);
    imwrite(test_labels{k},filename, 'BitDepth', 1, 'Compression', 'none');
end

%%

tile = train_tiles_1;
a = zeros(size(tile),'uint8');
a(:,:,1) = tile(:,:,1);
a(:,:,2) = tile(:,:,2);
a(:,:,3) = tile(:,:,3);
label = train_labels_1;
subplot(1, 2, 1);
imshow(a);
subplot(1, 2, 2);
imagesc(label);

%%

imds_train = imageDatastore('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Training_tiles');

imds_validation = imageDatastore('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Validation_tiles');

imds_test = imageDatastore('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Testing_tiles');

%%

classNames=["Non_LS" "LS"];
pixelLabelIDs=[0,1];

location_train='C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Training_labels';
pxds_train = pixelLabelDatastore(location_train,classNames,pixelLabelIDs);

location_validation='C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Validation_labels';
pxds_validation = pixelLabelDatastore(location_validation,classNames,pixelLabelIDs);

location_test='C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Kumamoto_v2_Testing_labels';
pxds_test = pixelLabelDatastore(location_test,classNames,pixelLabelIDs);

%%

pximds_train = pixelLabelImageDatastore(imds_train,pxds_train);
pximds_validation = pixelLabelImageDatastore(imds_validation,pxds_validation);
pximds_test = pixelLabelImageDatastore(imds_test,pxds_test);

%%

I = readimage(imds_train,3200);
I = histeq(I);
figure;imshow(I)

%%

CC = readimage(pxds_train,3200);
%cmap = camvidColorMap;
%B = labeloverlay(I,CC,'Colormap','gray');
B = labeloverlay(I,CC);
imshow(B)
%pixelLabelColorbar(cmap,classes);

%%

tbl = countEachLabel(pximds_train);

frequency = tbl.PixelCount/sum(tbl.PixelCount);
inverseFrequency = 1./frequency;
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%%

classNames = {'Non_LS','LS'};

% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [224 224 3];

% Specify the number of classes.
numClasses = 2;

% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50");

%%

pxLayer = pixelClassificationLayer('Name','labels','Classes',classNames,'ClassWeights',inverseFrequency);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

%%

analyzeNetwork(lgraph)

%%

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'OutputNetwork','best-validation-loss', ...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximds_validation,...
    'MaxEpochs',5, ...  
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 8);

%%

[net, info] = trainNetwork(pximds_train,lgraph,options);

%%

%ind_nums=[12 30 49 45 43 87 81 99 130 121 137];
ind_nums=[27 35 54 61 97 104];

for i=1:length(ind_nums)
    figure;
id_number=ind_nums(i);
I = readimage(imds_test,id_number);
C = semanticseg(I, net);

subplot(1, 4, 1);
I = histeq(I);
imshow(I)

CC = readimage(pxds_test,id_number);
B1 = labeloverlay(I,CC);
subplot(1, 4, 2);
imshow(B1)

B = labeloverlay(I,C,'Transparency',0.4);
subplot(1, 4, 3);
imshow(B)

expectedResult = readimage(pxds_test,id_number);
actual = uint8(C);
expected = uint8(expectedResult);
subplot(1, 4, 4);
imshowpair(actual, expected)
end

%%

classes = ["Non_LS";"LS"];
iou = jaccard(C,expectedResult);
table(classes,iou)

%%

pxdsResults = semanticseg(pximds_test,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

%%

metrics = evaluateSemanticSegmentation(pxdsResults,pxds_test,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics

%%

tile_orig=augm_origin_tile{34};
II = zeros(size(tile_orig),'uint8');
II(:,:,1) = tile_orig(:,:,1);
II(:,:,2) = tile_orig(:,:,2);
II(:,:,3) = tile_orig(:,:,3);
% Display the tile and label
imshow(II);

%%

CC = semanticseg(II, net);

%%

%featureLayer = "dec_relu4";
featureLayer = "res2a_branch2c";
reductionLayer = "softmax-out";

%%

[gradCAMMap,featureLayer,reductionLayer] = gradCAM(net,II,classes, ...
    ReductionLayer=reductionLayer, ...
    FeatureLayer=featureLayer);

%%

figure
subplot(1,3,1)
imshow(II)
title("Test Image")
subplot(1,3,2)
imshow(II)
hold on
imagesc(gradCAMMap(:,:,1),AlphaData=0.5)
colorbar
title("Grad-CAM: " + 'Non-Landslide')
colormap jet
subplot(1,3,3)
imshow(II)
hold on
imagesc(gradCAMMap(:,:,2),AlphaData=0.5)
colorbar
title("Grad-CAM: " + 'Landslide')
colormap jet

%%

layers = ["res2a_branch2c","res2b_branch2c","res2c_branch2c","res3a_branch2c","res3a_branch1","res3b_branch2c","res3c_branch2c","res3d_branch2c","res4a_branch2c","res4b_branch2c","res4d_branch2c","res4c_branch2c","res4e_branch2c","res4f_branch2c","res5a_branch2c","res5b_branch2c","res5c_branch2c","aspp_Conv_1","aspp_Conv_2","aspp_Conv_3","aspp_Conv_4","dec_c1","dec_c2","dec_c3","dec_c4","scorer"];
layer_num=[12, 24, 34, 44, 45, 56, 66, 76, 86, 98, 108, 118, 128, 138, 148, 160, 170, 174, 177, 180, 183, 187, 192, 196, 199, 202];

numLayers = length(layers);

%classes = ["Car" "Road"];
numClasses = length(classes);

gradCAMMaps = [];
for i = 1:numLayers
    gradCAMMaps(:,:,:,i) = gradCAM(net,II,classes, ...
        ReductionLayer=reductionLayer, ...
        FeatureLayer=layers(i));
end

figure;
idx = 1;
for i=1:numLayers
    for j=1:numClasses
        subplot(numLayers,numClasses,idx)
        imshow(II)
        hold on
        imagesc(gradCAMMaps(:,:,j,i),AlphaData=0.5)
        colorbar
        title(sprintf("%s (%s - %s)",classes(j),layers(i),layer_num(i)), ...
            Interpreter="none")
        colormap jet
        idx = idx + 1;
    end
end

%%

layers = ["res2a_branch2c","res2b_branch2c","res2c_branch2c","res3a_branch2c","res3a_branch1","res3b_branch2c","res3c_branch2c","res3d_branch2c","res4a_branch2c","res4b_branch2c","res4d_branch2c","res4c_branch2c","res4e_branch2c","res4f_branch2c","res5a_branch2c","res5b_branch2c","res5c_branch2c","aspp_Conv_1","aspp_Conv_2","aspp_Conv_3","aspp_Conv_4","dec_c1","dec_c2","dec_c3","dec_c4","scorer","dec_upsample2"];
layer_num=["12", "24", "34", "44", "45", "56", "66", "76", "86", "98", "108", "118", "128", "138", "148", "160", "170", "174", "177", "180", "183", "187", "192", "196", "199", "202", "203"];

numLayers = length(layers);

%classes = ["Car" "Road"];
numClasses = length(classes);

gradCAMMaps = [];
for i = 1:numLayers
    gradCAMMaps(:,:,:,i) = gradCAM(net,II,classes, ...
        ReductionLayer=reductionLayer, ...
        FeatureLayer=layers(i));
end

figure;
subplot(7,4,1)
imshow(II)
title('Original Tile')
idx = 2;
for i=1:numLayers
    for j=2
        subplot(7,4,idx)
        imshow(II)
        hold on
        imagesc(gradCAMMaps(:,:,j,i),AlphaData=0.5)
        colorbar
        title(sprintf("%s (%s)",layers(i),layer_num(i)), ...
            Interpreter="none")
        colormap turbo
        idx = idx + 1;
    end
end

%%

tile_orig=augm_origin_tile{34};
II = zeros(size(tile_orig),'uint8');
II(:,:,1) = tile_orig(:,:,1);
II(:,:,2) = tile_orig(:,:,2);
II(:,:,3) = tile_orig(:,:,3);
% Display the tile and label
%imshow(II);

act1 = activations(net,II,'res5a_branch2c');

sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);

I_act = imtile(mat2gray(act1),'GridSize',[8,8]);
imshow(I_act)

%%

tile_orig=augm_origin_tile{34};
II = zeros(size(tile_orig),'uint8');
II(:,:,1) = tile_orig(:,:,1);
II(:,:,2) = tile_orig(:,:,2);
II(:,:,3) = tile_orig(:,:,3);
% Display the tile and label
%imshow(II);

imgSize = size(II);
imgSize = imgSize(1:2);

act1ch22 = act1(:,:,:,22);
act1ch22 = mat2gray(act1ch22);
act1ch22 = imresize(act1ch22,imgSize);

II = imtile({II,act1ch22});
imshow(II)

%%

tile_orig=augm_origin_tile{34};
II = zeros(size(tile_orig),'uint8');
II(:,:,1) = tile_orig(:,:,1);
II(:,:,2) = tile_orig(:,:,2);
II(:,:,3) = tile_orig(:,:,3);
% Display the tile and label
%imshow(II);

[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);

II = imtile({II,act1chMax});
imshow(II)

%%

layer = 86;
name = net.Layers(layer).Name;

channels = 1:64;
I1 = deepDreamImage(net,name,channels, ...
    'PyramidLevels',1);
figure
I1 = imtile(I1,'ThumbnailSize',[64 64]);
imagesc(I1)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')

%%

layer = 196;
name = net.Layers(layer).Name;

channels = 1:36;
I1 = deepDreamImage(net,name,channels, ...
    'Verbose',false, ...
    "NumIterations",20, ...
    'PyramidLevels',3);
figure
I1 = imtile(I1,'ThumbnailSize',[250 250]);
imshow(I1)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')

%%

layers = ["res2a_branch2c","res2b_branch2c","res2c_branch2c","res3a_branch2c","res3a_branch1","res3b_branch2c","res3c_branch2c","res3d_branch2c","res4a_branch2c","res4b_branch2c","res4d_branch2c","res4c_branch2c","res4e_branch2c","res4f_branch2c","res5a_branch2c","res5b_branch2c","res5c_branch2c","aspp_Conv_1","aspp_Conv_2","aspp_Conv_3","aspp_Conv_4","dec_c1","dec_c2","dec_c3","dec_c4","scorer","dec_upsample2"];
layer_num=["12", "24", "34", "44", "45", "56", "66", "76", "86", "98", "108", "118", "128", "138", "148", "160", "170", "174", "177", "180", "183", "187", "192", "196", "199", "202", "203"];

numLayers = length(layers);

imgSize = size(II);
imgSize = imgSize(1:2);

tile_orig=augm_origin_tile{34};
II = zeros(size(tile_orig),'uint8');
II(:,:,1) = tile_orig(:,:,1);
II(:,:,2) = tile_orig(:,:,2);
II(:,:,3) = tile_orig(:,:,3);
% Display the tile and label
%imshow(II);

figure;
subplot(7,4,1)
imshow(II)
title('Original Tile')

ind_act=1;
for i=1:numLayers
    ind_act=ind_act+1;
    act1 = activations(net,II,layers(i));
    sz = size(act1);
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    [maxValue,maxValueIndex] = max(max(max(act1)));
    act1chMax = act1(:,:,:,maxValueIndex);
    act1chMax = mat2gray(act1chMax);
    act1chMax = imresize(act1chMax,imgSize);
    subplot(7,4,ind_act)
    imshow(act1chMax)
    title(sprintf("%s (%s)",layers(i),layer_num(i)), ...
            Interpreter="none")
    colormap gray
    colorbar
end

%%




