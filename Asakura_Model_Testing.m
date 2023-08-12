% Load the large image and its corresponding label
image_asakura = im2;
label_asakura = double(ref);

% assume your matrix is called "A"
label_asakura(label_asakura~=0) = 1;

% Determine the size of the image
[height, width, ~] = size(image_asakura);

% Define the tile size
tilesize = [224 224];

% Calculate the number of tiles in each dimension
numtiles_rows_asakura = ceil(size(image_asakura,1)/tilesize(1));
numtiles_cols_asakura = ceil(size(image_asakura,2)/tilesize(2));

% Initialize cell arrays to store the tiles and corresponding labels
asakura_tiles_cell = cell(numtiles_rows_asakura,numtiles_cols_asakura);
asakura_labeltiles_cell = cell(numtiles_rows_asakura,numtiles_cols_asakura);

% Loop over the tiles and extract them from the image and label
for row = 1:numtiles_rows_asakura
    for col = 1:numtiles_cols_asakura
        % Calculate the indices for the current tile
        start_row = (row-1)*tilesize(1)+1;
        end_row = min(row*tilesize(1), size(image_asakura,1));
        start_col = (col-1)*tilesize(2)+1;
        end_col = min(col*tilesize(2), size(image_asakura,2));
        
        % Check if the current tile is the correct size
        tilesize_cur = [end_row-start_row+1, end_col-start_col+1];
        if all(tilesize_cur == tilesize)
            % Extract the current tile from the image and label
            tile = image_asakura(start_row:end_row, start_col:end_col, :);
            labeltile = label_asakura(start_row:end_row, start_col:end_col);
            
            % Store the original tile and label in the cell arrays
            asakura_tiles_cell{row,col} = tile;
            asakura_labeltiles_cell{row,col} = labeltile;
        end
    end
end

% Remove empty cells from the cell arrays
asakura_tiles_cell = asakura_tiles_cell(~cellfun('isempty',asakura_tiles_cell));
asakura_labeltiles_cell = asakura_labeltiles_cell(~cellfun('isempty',asakura_labeltiles_cell));

%%

% Display a few of the augmented image tiles and their labels
figure;
num_samples = 1;
for i = 1:num_samples
    % Select a random index
    idx = randi(size(asakura_tiles_cell, 1));
    
    % Select a random augmented tile and label
    tile = asakura_tiles_cell{idx};
    a = zeros(size(tile),'uint8');
    a(:,:,1) = tile(:,:,1);
    a(:,:,2) = tile(:,:,2);
    a(:,:,3) = tile(:,:,3);
    label = asakura_labeltiles_cell{idx};
    
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
fprintf('Hokkaido set size: %d tiles, %d labels\n', numel(asakura_tiles_cell), numel(asakura_labeltiles_cell));

%%

for k=1:400
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Asakura_Tiles\Asakura_tiles',k);
    imwrite(asakura_tiles_cell{k},filename, 'BitDepth', 16);
end

for k=1:400
    filename = sprintf('%s_%d.png','C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Asakura_Labels\Asakura_labels',k);
    imwrite(asakura_labeltiles_cell{k},filename, 'BitDepth', 1, 'Compression', 'none');
end

%%

%tile = Asakura_tiles_2;
%a = zeros(size(tile),'uint8');
%a(:,:,1) = tile(:,:,1);
%a(:,:,2) = tile(:,:,2);
%a(:,:,3) = tile(:,:,3);
label = Asakura_labels_1;
subplot(1, 2, 1);
imshow(Asakura_tiles_1);
subplot(1, 2, 2);
imagesc(label);

%%

imds_Asakura = imageDatastore('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Asakura_Tiles');

%%

classNames=["Non_LS" "LS"];
pixelLabelIDs=[0,1];

location_Asakura='C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Hokkaido_DL_Transfer\02_Codes\MATLAB\Asakura_Labels';
pxds_Asakura = pixelLabelDatastore(location_Asakura,classNames,pixelLabelIDs);

%%

pximds_Asakura = pixelLabelImageDatastore(imds_Asakura,pxds_Asakura);

%%

I = readimage(imds_Asakura,60);
I = histeq(I);
figure;imshow(I)

%%

CC = readimage(pxds_Asakura,60);
%cmap = camvidColorMap;
%B = labeloverlay(I,CC,'Colormap','gray');
B = labeloverlay(I,CC);
imshow(B)
%pixelLabelColorbar(cmap,classes);

%%

tbl_hok = countEachLabel(pximds_hok);

%%

frequency_hok = tbl_hok.PixelCount/sum(tbl_hok.PixelCount);
inverseFrequency_hok = 1./frequency_hok;
bar(1:numel(classNames),frequency_hok)
xticks(1:numel(classNames)) 
xticklabels(tbl_hok.Name)
xtickangle(45)
ylabel('Frequency')

%%

id_number=1;
I = readimage(imds_Asakura,id_number);
C = semanticseg(I, net);

subplot(1, 4, 1);
I = histeq(I);
imshow(I)

CC = readimage(pxds_Asakura,id_number);
B1 = labeloverlay(I,CC);
subplot(1, 4, 2);
imshow(B1)

B = labeloverlay(I,C,'Transparency',0.4);
subplot(1, 4, 3);
imshow(B)

expectedResult = readimage(pxds_Asakura,id_number);
actual = uint8(C);
expected = uint8(expectedResult);
subplot(1, 4, 4);
%imshowpair(actual, expected,'ColorChannels', "red-cyan")
imshowpair(actual, expected)
%imshowpair(actual, expected, "blend")

%%

classes = ["Non_LS";"LS"];
iou_hok = jaccard(C,expectedResult);
table(classes,iou_hok)

%%

pxdsResults_Asakura = semanticseg(imds_Asakura,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

%%

metrics_Asakura = evaluateSemanticSegmentation(pxdsResults_Asakura,pxds_Asakura,'Verbose',false);
metrics_Asakura.DataSetMetrics
metrics_Asakura.ClassMetrics

%%

predictedLabels_Asakura = readall(pxdsResults_Asakura);

%%

inputs_Asakura = readall(imds_Asakura);
asa_ind=0;
for j=1:25
    for i=1:16
        asa_ind=asa_ind+1;
        asa_ind
        I = asakura_tiles_cell{asa_ind};
        L = asakura_labeltiles_cell{asa_ind};
        [C,score,allScores] = semanticseg(I,net);
        inputs_Asakura_cell{i,j}=I;
        targets_Asakura_cell{i,j}=L;
        preds_Asakura_cell{i,j}=C;
        scores_Asakura_cell{i,j}=score;
        allScores1_Asakura_cell{i,j}=allScores(:,:,1);
        allScores2_Asakura_cell{i,j}=allScores(:,:,2);
    end
end

%%

% visualize the binary numerical matrix using montage
asa_ind=0;
for i=1:16
    for j=1:25
        asa_ind=asa_ind+1;
        subplot(16,25,asa_ind)
        imshow(inputs_Asakura_cell{i,j});
    end
end

%%

% create a 16-by-25 cell array of categorical labels
cell_array = preds_Asakura_cell;

% concatenate the categorical labels into a single large matrix
cat_labels = cell2mat(cellfun(@(x) double(x)-1, cell_array, 'UniformOutput', false));

% convert categorical labels to binary numerical matrix
%bin_labels = logical(cat_labels);

% visualize the binary numerical matrix using montage
montage(cat_labels);

%%

% create some sample data
data = allScores2_Asakura_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 16);
for i = 1:16
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
data = inputs_Asakura_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 16);
for i = 1:16
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
data1 = targets_Asakura_cell;

% concatenate the matrices horizontally and vertically
rows = cell(1, 16);
for i = 1:16
    rows{i} = cat(2, data1{i,:});
end
result1_label = cat(1, rows{:});

% visualize the result
figure;
imshow(result1_label);
%colormap(gray);

%%

figure;
B = labeloverlay(result,result1_label,'Colormap','hsv','Transparency',0.1);
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


