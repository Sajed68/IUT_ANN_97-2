clc
clear
close all

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
%------ add your code here------ %

%%  predict output:
y = randi(2,size(moonY(1,71:end)))-1;
%------- add your code here------ %

%% Evaluate by Confusion matrix:
confusionmat(moonY(71:end), y)

%% plot output:
figure()
plotpv(moonX(:,71:end), moonY(:,71:end), [-1, 1, -1, 1])
hold on
gscatter(moonX(1,71:end), moonX(2,71:end), y, 'rgb');
title('predicted')
