% Load the large image and its corresponding label
image_hok = Hok_raster_clipped_rect;
label_hok = double(Hok_LS_projected_rect);

% Determine the size of the image
[height, width, ~] = size(image_hok);
numrows=height*6;
numcols=width*6;

image_hok_res = imresize(image_hok,[numrows numcols],"bilinear");
label_hok_res = imresize(label_hok,[numrows numcols],"nearest");

% assume your matrix is called "A"
label_hok_res(label_hok_res~=0) = 1;

% Define the tile size
tilesize = [224 224];

% Calculate the number of tiles in each dimension
numtiles_rows_hok_res = ceil(size(image_hok_res,1)/tilesize(1));
numtiles_cols_hok_res = ceil(size(image_hok_res,2)/tilesize(2));

% Initialize cell arrays to store the tiles and corresponding labels
hok_tiles_cell_res = cell(numtiles_rows_hok_res,numtiles_cols_hok_res);
hok_labeltiles_cell_res = cell(numtiles_rows_hok_res,numtiles_cols_hok_res);

% Loop over the tiles and extract them from the image and label
for row = 1:numtiles_rows_hok_res
    for col = 1:numtiles_cols_hok_res
        % Calculate the indices for the current tile
        start_row = (row-1)*tilesize(1)+1;
        end_row = min(row*tilesize(1), size(image_hok_res,1));
        start_col = (col-1)*tilesize(2)+1;
        end_col = min(col*tilesize(2), size(image_hok_res,2));
        
        % Check if the current tile is the correct size
        tilesize_cur = [end_row-start_row+1, end_col-start_col+1];
        if all(tilesize_cur == tilesize)
            % Extract the current tile from the image and label
            tile = image_hok_res(start_row:end_row, start_col:end_col, :);
            labeltile = label_hok_res(start_row:end_row, start_col:end_col);
            
            % Store the original tile and label in the cell arrays
            hok_tiles_cell_res{row,col} = tile;
            hok_labeltiles_cell_res{row,col} = labeltile;
        end
    end
end

% Remove empty cells from the cell arrays
hok_tiles_cell_res = hok_tiles_cell_res(~cellfun('isempty',hok_tiles_cell_res));
hok_labeltiles_cell_res = hok_labeltiles_cell_res(~cellfun('isempty',hok_labeltiles_cell_res));

%%

% Display a few of the augmented image tiles and their labels

figure;
num_samples = 1;
for i = 1:num_samples
    % Select a random index
    idx = randi(size(hok_tiles_cell_res, 1));
    
    % Select a random augmented tile and label
    tile = hok_tiles_cell_res{idx};
    a = zeros(size(tile),'uint8');
    a(:,:,1) = tile(:,:,1);
    a(:,:,2) = tile(:,:,2);
    a(:,:,3) = tile(:,:,3);
    label = hok_labeltiles_cell_res{idx};
    
    % Display the tile and label
    subplot(num_samples, 2, 2*i - 1);
    imshow(a);
    title(sprintf('Augmented Tile %d', i));
    
    subplot(num_samples, 2, 2*i);
    imagesc(label);
    title(sprintf('Augmented Label %d', i));
end

%%

% Display the sizes of the resulting cells
fprintf('Hokkaido set size: %d tiles, %d labels\n', numel(hok_tiles_cell_res), numel(hok_labeltiles_cell_res));

%%

for k=1:3321
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Hokkaido_Tiles_res\hok_tiles_res',k);
    imwrite(hok_tiles_cell_res{k},filename, 'BitDepth', 16);
end

for k=1:3321
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Hokkaido_Labels_res\hok_labels_res',k);
    imwrite(hok_labeltiles_cell_res{k},filename, 'BitDepth', 1, 'Compression', 'none');
end

%%

tile = hok_tiles_res_1;
a = zeros(size(tile),'uint8');
a(:,:,1) = tile(:,:,1);
a(:,:,2) = tile(:,:,2);
a(:,:,3) = tile(:,:,3);
label = hok_labels_res_1;
subplot(1, 2, 1);
imshow(a);
subplot(1, 2, 2);
imagesc(label);

%%

imds_hok_res = imageDatastore('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Hokkaido_Tiles_res');

%%

classNames=["Non_LS" "LS"];
pixelLabelIDs=[0,1];

location_hok_res='C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Hokkaido_Labels_res';
pxds_hok_res = pixelLabelDatastore(location_hok_res,classNames,pixelLabelIDs);

%%

pximds_hok_res = pixelLabelImageDatastore(imds_hok_res,pxds_hok_res);

%%

I = readimage(imds_hok_res,760);
I = histeq(I);
figure;imshow(I)

%%

CC = readimage(pxds_hok_res,760);
%cmap = camvidColorMap;
%B = labeloverlay(I,CC,'Colormap','gray');
B = labeloverlay(I,CC);
imshow(B)
%pixelLabelColorbar(cmap,classes);

%%

tbl_hok_res = countEachLabel(pximds_hok_res);

%%

frequency_hok_res = tbl_hok_res.PixelCount/sum(tbl_hok_res.PixelCount);
inverseFrequency_hok_res = 1./frequency_hok_res;
bar(1:numel(classNames),frequency_hok_res)
xticks(1:numel(classNames)) 
xticklabels(tbl_hok_res.Name)
xtickangle(45)
ylabel('Frequency')

%%

id_number=2642;
I = readimage(imds_hok_res,id_number);
C = semanticseg(I, net);

subplot(1, 4, 1);
I = histeq(I);
imshow(I)

CC = readimage(pxds_hok_res,id_number);
B1 = labeloverlay(I,CC);
subplot(1, 4, 2);
imshow(B1)

B = labeloverlay(I,C,'Transparency',0.4);
subplot(1, 4, 3);
imshow(B)

expectedResult = readimage(pxds_hok_res,id_number);
actual = uint8(C);
expected = uint8(expectedResult);
subplot(1, 4, 4);
imshowpair(actual, expected)

%%

classes = ["Non_LS";"LS"];
iou_hok_res = jaccard(C,expectedResult);
table(classes,iou_hok_res)

%%

pxdsResults_hok_res = semanticseg(pximds_hok_res,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

%%

metrics_hok_res = evaluateSemanticSegmentation(pxdsResults_hok_res,pxds_hok_res,'Verbose',false);
metrics_hok_res.DataSetMetrics
metrics_hok_res.ClassMetrics

%%

analyzeNetwork(net)

%%

%featureLayer = "dec_relu4";
featureLayer = "res5a_branch2a";
reductionLayer = "softmax-out";

%%

gradCAMMap = gradCAM(net,I,classes, ...
    ReductionLayer=reductionLayer, ...
    FeatureLayer=featureLayer);

%%

figure
subplot(2,2,1)
imshow(I)
title("Test Image")
subplot(2,2,2)
B = labeloverlay(I,C,'Transparency',0.4);
imshow(B)
title("Semantic Segmentation")
subplot(2,2,3)
imshow(I)
hold on
imagesc(gradCAMMap(:,:,1),AlphaData=0.5)
title("Grad-CAM: " + classes(1))
colormap jet
subplot(2,2,4)
imshow(I)
hold on
imagesc(gradCAMMap(:,:,2),AlphaData=0.5)
title("Grad-CAM: " + classes(2))
colormap jet

%%

predictedLabels_Asakura = readall(pxdsResults_Asakura);

%%

asa_ind=0;
for j=1:81
    for i=1:41
        asa_ind=asa_ind+1;
        asa_ind
        I = hok_tiles_cell_res{asa_ind};
        L = hok_labeltiles_cell_res{asa_ind};
        [C,score,allScores] = semanticseg(I,net);
        inputs_hok_cell{i,j}=histeq(I);
        targets_hok_cell{i,j}=L;
        preds_hok_cell{i,j}=C;
        scores_hok_cell{i,j}=score;
        allScores1_hok_cell{i,j}=allScores(:,:,1);
        allScores2_hok_cell{i,j}=allScores(:,:,2);
    end
end

%%

% visualize the binary numerical matrix using montage
asa_ind=0;
for i=1:41
    for j=1:81
        asa_ind=asa_ind+1;
        subplot(41,81,asa_ind)
        imshow(inputs_hok_cell{i,j});
    end
end

%%

% create a 16-by-25 cell array of categorical labels
cell_array = preds_hok_cell;

% concatenate the categorical labels into a single large matrix
cat_labels = cell2mat(cellfun(@(x) double(x)-1, cell_array, 'UniformOutput', false));

% convert categorical labels to binary numerical matrix
%bin_labels = logical(cat_labels);

% visualize the binary numerical matrix using montage
montage(cat_labels);

%%

% create some sample data
data = allScores2_hok_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 41);
for i = 1:41
    rows{i} = cat(2, data{i,:});
end
result = cat(1, rows{:});

% visualize the result
figure;
imagesc(result);
axis on;
%colormap(gray);

%%

% create some sample data
data = inputs_hok_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 41);
for i = 1:41
    rows{i} = cat(2, data{i,:});
end
result = cat(1, rows{:});

% visualize the result
figure;
imshow(result);
%colormap(gray);

axis on;
[rows, columns, numberOfColorChannels] = size(result);
hold on;
for row = 1 : 224 : rows
  line([1, columns], [row, row], 'Color', 'w');
end
hold on
for col = 1 : 224 : columns
  line([col, col], [1, rows], 'Color', 'w');
end
hold off

%%

% create some sample data
data1 = targets_hok_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 41);
for i = 1:41
    rows{i} = cat(2, data1{i,:});
end
result1_label = cat(1, rows{:});

% visualize the result
figure;
imshow(result1_label);
%colormap(gray);

%%

figure;
B = labeloverlay(result,result1_label,'Colormap','hsv','Transparency',0);
imshow(B)
axis on;
[rows, columns, numberOfColorChannels] = size(result);
hold on;
for row = 1 : 224 : rows
  line([1, columns], [row, row], 'Color', 'w');
end
hold on
for col = 1 : 224 : columns
  line([col, col], [1, rows], 'Color', 'w');
end
hold off




