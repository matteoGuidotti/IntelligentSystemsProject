%% Clean
clear
close all
clc
format compact

%% Load the dataset
dataset = load('data/dataset.mat');
dataset = table2array(dataset.dataset);

%% Remove infinite values
is_inf = isinf(dataset);
[infinite_rows, ~] = find(is_inf == 1);
dataset(infinite_rows, :) = [];

%% OUTLIERS REMOVAL

% I ignore the first 2 lines beacuse they contain arousal and valence levels
dataset = dataset(:, 3:end);
[initial_rows, ~] = size(dataset);
% In rmoutliers I use the default method "median"
clean_dataset = rmoutliers(dataset);
[final_rows, ~] = size(clean_dataset);
fprintf("%i outliers have been removed\n", initial_rows - final_rows);

%% DATA BALANCING

% Instantiate some control variables
EXTRACT_VALENCE = 1;
EXTRACT_AROUSAL = 1;
BALANCING_DATA = 1;

% isolating leveles of arousal and valence
arousal_levels = clean_dataset(:, 1);
valence_levels = clean_dataset(:, 2);

% number of samples for arousal and valence
sample_arousal = groupcounts(arousal_levels);
sample_valence = groupcounts(valence_levels);

% plotting the graph
% plot the graph
figure("Name", "Sample for arousal before balancing");
bar(sample_arousal);
title("Sample for arousal before balancing");

fprintf("Data are unbalanced\n");

[~, min_arousal] = min(sample_arousal);
[~, max_arousal] = max(sample_arousal);

% plot the graph
figure("Name", "Sample for valence before balancing");
bar(sample_valence);
title("Sample for valence before balancing");

[~, min_valence] = min(sample_valence);
[~, max_valence] = max(sample_valence);

augmentation_factors = [0 0];

possible_values = unique(arousal_levels);

rep = 80;
row_to_check = final_rows;

if BALANCING_DATA == 1
	for k = 1:rep
		for i = 1:row_to_check
			if(clean_dataset(i,1)==possible_values(min_arousal) && clean_dataset(i,2)~=possible_values(max_valence)) || (clean_dataset(i,1)~=possible_values(max_arousal) && clean_dataset(i,2)==possible_values(min_valence))
				% Augmentation of the i-th row
				current_row = clean_dataset(i, :);
				row_to_add = current_row;
				% Random selection of the augmentation factor
				randomizer = 1 + (rand(1) - 0.5)/10;
				% Augmentation
				row_to_add(3: end) = current_row(3: end).*randomizer;
				% Adding the new sample to the dataset
				clean_dataset = [clean_dataset; row_to_add];
			end

			if((clean_dataset(i,1)==possible_values(max_arousal) && clean_dataset(i,2)~=possible_values(min_valence)) || (clean_dataset(i,2)==possible_values(max_valence) && clean_dataset(i,1)~=possible_values(min_arousal)))
				clean_dataset(i, :) = [];
			end
		end
		sample_arousal = groupcounts(clean_dataset(:,1));
		sample_valence = groupcounts(clean_dataset(:,2));

		[~, min_arousal] = min(sample_arousal);
		[~, max_arousal] = max(sample_arousal);

		[~, min_valence] = min(sample_valence);
		[~, max_valence] = max(sample_valence);
	end
	fprintf("Balancing process finished\n");

	sample_arousal = groupcounts(clean_dataset(:,1));
    sample_valence = groupcounts(clean_dataset(:,2));
    figure("Name", "Samples for arousal after balancing");
    bar(sample_arousal);
    title("Samples for arousal after balancing");
    fprintf("Arousal data balanced\n");
    figure("Name", "Samples for valence after balancing");
    bar(sample_valence);
    title("Samples for valence after balancing");
    fprintf("Valence data balanced\n");
end

%% FEATURES SELECTION

features_set = clean_dataset(:, 3:end);
target_arousal = clean_dataset(:, 1);
target_valence = clean_dataset(:, 2);

cv = cvpartition(target_arousal, 'holdout', 0.3);
idxTraining = training(cv);
idxTesting = test(cv);

x_train = features_set(idxTraining, :);
y_train_arousal = target_arousal(idxTraining, :);
y_train_valence = target_valence(idxTraining, :);

x_test = features_set(idxTesting, :);
y_test_arousal = target_arousal(idxTesting, :);
y_test_valence = target_valence(idxTesting, :);

sequentialfs_rep = 10;
nfeatures = 5;

%% AROUSAL FEATURES EXCTRACTION

if EXTRACT_AROUSAL == 1
	features_arousal = [zeros(1, 54); 1:54]';	% 54x2 matrix, first column 0, second 1:54
	for i = 1:sequentialfs_rep
		fprintf("Iteration number %i\n", i);
		c = cvpartition(y_train_arousal, 'k', 10);
		option = statset('display', 'iter', 'useParallel', true);
		% selected_features_arousal will contain 1 in the selected features indexes
		selected_features_arousal = sequentialfs(@myfun, x_train, y_train_arousal, 'cv', c, 'opt', option, 'nFeatures', nfeatures);
		% Count how many times a feature has been selected
		for j = 1:54
			if selected_features_arousal(j) == 1
				features_arousal(j, 1) = features_arousal(j, 1) + 1;
			end
		end
	end

	fprintf("\n");
	fprintf("*** AROUSAL: ");
	fprintf("\n");

	disp(features_arousal);
	fprintf("Sorting features:\n");
	features_arousal = sortrows(features_arousal, 1, 'descend');
	disp(features_arousal);

	% Isolating the top 10 selected arousal features
	best_arousal = features_arousal(1:10, 2);

	best_arousal_training.x_train = normalize(x_train(:, best_arousal));
	best_arousal_training.y_train = y_train_arousal';
	% Saving the obtained structure in the correct file
	save("data/training_arousal.mat", "best_arousal_training");

	best_arousal_testing.x_test = normalize(x_test(:, best_arousal));
    best_arousal_testing.y_test = y_test_arousal';
    save("data/testing_arousal.mat", "best_arousal_testing");
    fprintf("Arousal features saved\n");
end

%% VALENCE FEATURES EXTRACTION

if EXTRACT_VALENCE == 1
	features_valence = [zeros(1, 54); 1:54]';	% 54x2 matrix, first column 0, second 1:54
	for i = 1:sequentialfs_rep
		fprintf("Iteration number %i\n", i);
		c = cvpartition(y_train_valence, 'k', 10);
		option = statset('display', 'iter', 'useParallel', true);
		% selected_features_valence will contain 1 in the selected features indexes
		selected_features_valence = sequentialfs(@myfun, x_train, y_train_valence, 'cv', c, 'opt', option, 'nFeatures', nfeatures);
		% Count how many times a feature has been selected
		for j = 1:54
			if selected_features_valence(j) == 1
				features_valence(j, 1) = features_valence(j, 1) + 1;
			end
		end
	end

	fprintf("\n");
	fprintf("*** VALCENCE: ");
	fprintf("\n");

	disp(features_valence);
	fprintf("Sorting features:\n");
	features_valence = sortrows(features_valence, 1, 'descend');
	disp(features_valence);

	% Isolating the top 10 selected valence features
	best_valence = features_valence(1:10, 2);

	best_valence_training.x_train = normalize(x_train(:, best_valence));
	best_valence_training.y_train = y_train_valence';
	% Saving the obtained structure in the correct file
	save("data/training_valence.mat", "best_valence_training");

	best_valence_testing.x_test = normalize(x_test(:, best_valence));
    best_valence_testing.y_test = y_test_valence';
    save("data/testing_valence.mat", "best_valence_testing");
    fprintf("valence features saved\n");
end

%% Save best-3 features arousal dataset for task 3.3
arousal_possible_values = unique(target_arousal);
arousal_best3 = features_arousal(1:3, 2);
best3.x_train = normalize(x_train(:, arousal_best3));
best3.y_train = y_train_arousal';
best3.x_test = normalize(x_test(:, arousal_best3));
best3.y_test = y_test_arousal';
best3.best_features=arousal_best3;
best3.y_values= arousal_possible_values;
% Save struct in the correct file
save("data/best3.mat", "best3");
fprintf("Best-3 arousal features saved\n");

%% Function for sequentialfs
function err = myfun(x_train, t_train, x_test, t_test)
    net = fitnet(60);
    net.trainParam.showWindow=0;
    % net.trainParam.showCommandLine=1;
    xx = x_train';
    tt = t_train';
    net = train(net, xx, tt);
    y=net(x_test'); 
    err = perform(net,t_test',y);
end