%% Clean
clear
close all
clc

%% Loading the images as an image datastore
% Image datastore consents to label the images based on their folder names

CLASSIFICATION = 5;

if CLASSIFICATION == 0
    image_data = imageDatastore("data/images/selected/classification_2_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 1
        image_data = imageDatastore("data/images/selected/classification_4_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 2
        image_data = imageDatastore("data/images/noSelected/classification_2_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 3
        image_data = imageDatastore("data/images/noSelected/classification_4_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 4
        image_data = imageDatastore("data/images/1000_images/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 5
        image_data = imageDatastore("data/images/500_images/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

%% Splitting the images datastore into separate datastores for trainig, validation and testing
% 70 per training, 20 per validation, 10 per testing

[data_train, ~] = splitEachLabel(image_data, 0.7, 'randomized');
[data_validation, data_test] = splitEachLabel(image_data, 0.2, 'randomized');

class_number = numel(categories(data_train.Labels));

%% Alexnet

net = alexnet;
input_size = net.Layers(1).InputSize;

% All the layers except the last 3 will be extracted
% These layers must be fine-tuned for the new classification problems

alexnet_layers = net.Layers(1: end - 3);

% Three new layers will be added to the network structure
% 1) A fully connected layer that has as inputs the number of classes, with high weight and bias learning rate 
% 2) Softmax layer for classification
% 3) Classification layer for the final output

layers = [
	alexnet_layers
	fullyConnectedLayer(class_number, 'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
	softmaxLayer
	classificationLayer
];

pixelRange = [-30 30];
imgAugmenter = imageDataAugmenter( ...
	'RandXReflection', true, ...
	'RandXTranslation',pixelRange, ...
	'RandYTranslation',pixelRange ...
	);

augmented_img_data_train = augmentedImageDatastore(input_size(1:2), data_train, 'DataAugmentation', imgAugmenter);
augmented_img_data_validation = augmentedImageDatastore(input_size(1:2), data_validation);
augmented_img_data_test = augmentedImageDatastore(input_size(1:2), data_test);

options = trainingOptions(...
	'sgdm', ...
	'MiniBatchSize', 10, ...
	'MaxEpochs', 10, ...
	'InitialLearnRate', 1e-4, ...
	'Shuffle', 'every-epoch', ...
	'ValidationData',augmented_img_data_validation, ...
	'ValidationFrequency', 3, ...
	'Verbose', false, ...
	'Plots', 'training-progress' ... 
);

%% Training phase

cnn_network = trainNetwork(augmented_img_data_train, layers, options);
hold on		% maintain the same plot

hold on
[res_train, ~] = classify(cnn_network, augmented_img_data_train);
target_train = data_train.Labels;
%train_accuracy = mean(target_train==res_train);
plotconfusion(target_train, res_train);

%% Validation phase

[res_val, ~] = classify(cnn_network, augmented_img_data_validation);
target_val = data_validation.Labels;
validation_accuracy = mean(target_val == res_val);
plotconfusion(target_val, res_val);

%% Testing phase

[res_test, ~] = classify(cnn_network, augmented_img_data_test);
target_test = data_test.Labels;
test_accuracy = mean(target_test == res_test);
plotconfusion(target_test, res_test);