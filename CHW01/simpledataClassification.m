clc
clear
close all

%% load data set:
load('testdataX.mat');
testX = testX';;
testY = zeros(3, 100);
testY(1, find(testX(1,:) < 0 & testX(2,:) < 0)) = 1;
testY(1, find(testX(1,:) > 0 & testX(2,:) < 0)) = 1;
testY(2, find(testX(1,:) < 0 & testX(2,:) > 0)) = 1;
testY(3, find(testX(1,:) > 0 & testX(2,:) > 0)) = 1;
%% plot data:
figure()
[~, label] = max(testY);
gscatter(testX(1,1:70), testX(2,1:70), label(:,1:70), 'rgb');

%% initiate the network:
W = randn(3,2);
b = randn(3,1);
%% train W and b on "trainX" and "trainY" data
%------- write your code here:---------%

%%  predict output:
y = zeros(3, 31);
for i = 1:30
    y(:,i)  = hardlim( W*(testX(:,70+i)) + b);
end
%% Evaluate by Confusion matrix:
[~, P] = max(y);
confusionmat(label(70:end), P)

%% plot output:
figure()
plotpv(testX(:,70:end), testY(1,70:end), [-1, 1, -1, 1])
hold on
gscatter(testX(1,70:end), testX(2,70:end), P, 'rgb');
title('predicted')
plotpc(W(1,:), b(1))
plotpc(W(2,:), b(2))
plotpc(W(3,:), b(3))