clc;
clear all;
close all;

%% load data set:
load('moondataX.mat');
moonX = moonX';
load('moondataY.mat');
moonY = double(moonY);
trainX = moonX(:, 1:70);
trainY = moonY(:, 1:70);
%% plot data:
figure()
gscatter(moonX(1,1:70), moonX(2,1:70), moonY(:,1:70), 'rgb');

%% initiate the network here:
% choose number of layers and neurons at each layer
% be carefull! select simplest model you can.
%  train on "trainX" and "trainY"
N_neuron = 4;
W1 = randn(N_neuron,2); %
b1 = randn(N_neuron,1); %
W2 = randn(1,N_neuron); %
b2 = randn(1,1); %
%------ add youre code here------ %
N_epoch = 100;
P_matrix = trainX;
T_matrix = trainY;
N_sample = size(P_matrix,2);
alpha = 5;
epsilon = 10^-3;
error_vec = zeros(N_epoch,1);
for epoch_index = 1 : N_epoch
    for index = 1 : N_sample
        % input and target
        sample = P_matrix(:,index);
        target = T_matrix(:,index);
        a0 = sample;
        
        % forward propagation
        n1 = W1 * a0 + b1;
        a1 = logsig(n1);
        n2 = W2 * a1 + b2;
        a2 = logsig(n2);
        
        % output and error
        output = a2;
        error = target - output;
        
        % backward propagation
        s2 = -2 * (a2 .* (1 - a2)) * error;
        % s2 = -2 * error;
        s1 = diag(a1 .* (ones(N_neuron,1) - a1)) * transpose(W2) * s2;
        
        % weight and bias update
        W2 = W2 - alpha * s2 * transpose(a1);
        b2 = b2 - alpha * s2;
        W1 = W1 - alpha * s1 * transpose(a0);
        b1 = b1 - alpha * s1;
    end
    A1 = logsig(W1 * P_matrix + repmat(b1,1,N_sample));
    A2 = logsig(W2 * A1 + repmat(b2,1,N_sample));
    all_error = A2 - T_matrix;
    % all_error = heaviside(A2 - 0.5) - T_matrix;
    error_vec(epoch_index,1) = norm(all_error,2);
    if (norm(all_error,2) < epsilon)
        break;
    end 
end
%%  predict output:
%------- add your code here------ %
testX = moonX(:, 71:100);
testY = moonY(:, 71:100);
y = zeros(1, 30);
for i = 1:30
    a0 = testX(:,i);
    a1 = logsig(W1 * a0 + b1);
    a2 = logsig(W2 * a1 + b2);
    y(:,i)  = a2;
end
y = heaviside(y - 0.5);
%% Evaluate by Confusion matrix:
confusionmat(moonY(71:end), y)
figure()
plot(error_vec)

%% plot output:
figure()
plotpv(moonX(:,71:end), moonY(:,71:end), [-1, 1, -1, 1])
hold on
gscatter(moonX(1,71:end), moonX(2,71:end), y, 'rgb');
title('predicted')

