%% Clean
clear
close all
clc

%% Loading the features

test_arousal = load('data/testing_arousal.mat'); 
train_arousal = load('data/training_arousal.mat');
x_train_arousal = train_arousal.best_arousal_training.x_train';
t_train_arousal = train_arousal.best_arousal_training.y_train'.';
x_test_arousal = test_arousal.best_arousal_testing.x_test';
t_test_arousal = test_arousal.best_arousal_testing.y_test'.';
fprintf("Arousal features loaded\n");

test_valence = load('data/testing_valence.mat');
train_valence = load('data/training_valence.mat');
x_train_valence = train_valence.best_valance_training.x_train';
t_train_valence = train_valence.best_valance_training.y_train'.';
x_test_valence = test_valence.best_valance_testing.x_test';
t_test_valence = test_valence.best_valance_testing.y_test'.';
fprintf("Valence features loaded\n");

MLP_AROUSAL = 0;
MLP_VALENCE = 0;
RBFN_AROUSAL = 1;
RBFN_VALENCE = 0;
TESTING_AROUSAL = 1;
TESTING_VALANCE = 0;

%% MLP OUTPUTTING AROUSAL LEVEL

if MLP_AROUSAL == 1
    % Optimal Neural Network Architecture found for arousal
	hidden_neurons = 45;
    mlp_net_arousal = fitnet(hidden_neurons);
    mlp_net_arousal.divideParam.trainRatio = 0.7;
    mlp_net_arousal.divideParam.testRatio = 0.1; 
    mlp_net_arousal.divideParam.valRatio = 0.2;
    mlp_net_arousal.trainParam.showWindow = 1;
    mlp_net_arousal.trainParam.showCommandLine = 1;
    mlp_net_arousal.trainParam.lr = 0.1; 
    mlp_net_arousal.trainParam.epochs = 100;
    mlp_net_arousal.trainParam.max_fail = 10;
    
    [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, t_train_arousal);
    view(mlp_net_arousal);
	plotperform(tr_arousal);

	y_test_arousal = mlp_net_arousal(x_test_arousal);
	plotregression(t_test_arousal, y_test_arousal, ['Final test arousal: ' + hidden_neurons + ' hidden neurons' ]);
    

% Traces of other experiments
elseif TESTING_AROUSAL == 1
    fprintf("Testing Arousal\n");
    max_neurons_1 = 120;
	best_R = 0;
	hiddenLayerSize_arousal = 0;

    for i=5:5:max_neurons_1    
        mlp_net_arousal = fitnet(i);
        mlp_net_arousal.divideParam.trainRatio = 0.7;
        mlp_net_arousal.divideParam.testRatio = 0.1; 
        mlp_net_arousal.divideParam.valRatio = 0.2;
        mlp_net_arousal.trainParam.showWindow = 0;
        mlp_net_arousal.trainParam.showCommandLine = 1;
        mlp_net_arousal.trainParam.lr = 0.1; 
        mlp_net_arousal.trainParam.epochs = 100;
        mlp_net_arousal.trainParam.max_fail = 15;
        [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, t_train_arousal);
        %view(mlp_net_arousal);
        y_test_arousal = mlp_net_arousal(x_test_arousal);
		v = figure;
		plotregression(t_test_arousal, y_test_arousal);

		str = v.Children(3).Title.String;
		ind = find(str=='=');
		current_R_string = str(ind + 1:end);
		current_R = str2double(current_R_string);
		if best_R < current_R
			best_R = current_R;
			hiddenLayerSize_arousal = i;
        end
		%close(v);
    end
	fprintf("Max R value saved: %d for hiddenLayerSize %d \n", best_R, hiddenLayerSize_arousal);
end

%% MLP OUTPUTTING VALENCE LEVELS

if MLP_VALENCE == 1
    % Optimal Neural Network Architecture found for valence
	hidden_neurons = 80;
    mlp_net_valence = fitnet(hidden_neurons);
    mlp_net_valence.divideParam.trainRatio = 0.7;
    mlp_net_valence.divideParam.testRatio = 0.1; 
    mlp_net_valence.divideParam.valRatio = 0.2;
    mlp_net_valence.trainParam.showWindow = 1;
    mlp_net_valence.trainParam.showCommandLine = 1;
    mlp_net_valence.trainParam.lr = 0.1; 
    mlp_net_valence.trainParam.epochs = 100;
    mlp_net_valence.trainParam.max_fail = 15;
    
    [mlp_net_valence, tr_valence] = train(mlp_net_valence, x_train_valence, t_train_valence);
    view(mlp_net_valence);
	plotperform(tr_valence);

	y_test_valence = mlp_net_valence(x_test_valence);
	plotregression(t_test_valence, y_test_valence, ['Final test valence: ' + hidden_neurons + ' hidden neurons']);
    

% Traces of other experiments
elseif TESTING_valence == 1
    fprintf("Testing valence\n");
    max_neurons_1 = 120;
	best_R = 0;
	hiddenLayerSize_valence = 0;

    for i=5:5:max_neurons_1    
        mlp_net_valence = fitnet(i);
        mlp_net_valence.divideParam.trainRatio = 0.7;
        mlp_net_valence.divideParam.testRatio = 0.1; 
        mlp_net_valence.divideParam.valRatio = 0.2;
        mlp_net_valence.trainParam.showWindow = 0;
        mlp_net_valence.trainParam.showCommandLine = 1;
        mlp_net_valence.trainParam.lr = 0.1; 
        mlp_net_valence.trainParam.epochs = 100;
        mlp_net_valence.trainParam.max_fail = 15;
        [mlp_net_valence, tr_valence] = train(mlp_net_valence, x_train_valence, t_train_valence);
        %view(mlp_net_valence);
        y_test_valence = mlp_net_valence(x_test_valence);
		v = figure;
		plotregression(t_test_valence, y_test_valence);

		str = v.Children(3).Title.String;
		ind = find(str=='=');
		current_R_string = str(ind + 1:end);
		current_R = str2double(current_R_string);
		if best_R < current_R
			best_R = current_R;
			hiddenLayerSize_valence = i;
        end
        %close(v);
    end
	fprintf("Max R value saved: %d for hiddenLayerSize %d \n", best_R, hiddenLayerSize_valence);
end

%% RBF PART

%% RBF TRAINING FOR AROUSAL

% RBFN outputting arousal levels

if RBFN_AROUSAL == 1
    %Creation of RBFN
    goal_ar = 0;
    spread_ar = 1.07;
    K_ar = 500;
    Ki_ar = 50;

    rbf_arousal = newrb(x_train_arousal,y_train_arousal,goal_ar,spread_ar,K_ar,Ki_ar);
    view (rbf_arousal);
    %Test
    test_output_arousal_rbf = rbf_arousal(x_test_arousal);
    plotregression(y_test_arousal, test_output_arousal_rbf, 'Final test arousal with RBF');
end

%% RBFN outputting valence levels

if RBFN_VALENCE == 1
    %Creation of RBFN
    goal_va = 0;
    spread_va = 0.7;
    K_va = 500;
    Ki_va = 50;
    
    rbf_valence = newrb(x_train_valence,y_train_valence,goal_va,spread_va, K_va, Ki_va);
    view (rbf_valence);
    %Test
    test_output_valence_rbf = rbf_valence(x_test_valence);
    plotregression(y_test_valence, test_output_valence_rbf, 'Final test valence with RBF');
end