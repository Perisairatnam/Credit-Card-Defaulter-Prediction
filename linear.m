clear all; close all;
%Reading xlx file 
trainData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\Balancedtraining.csv");
X = trainData(:,1:80);
Y = trainData(:,81);

LiMdl = fitglm(X, 'ResponseVar', Y);
testData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\Balancedtesting.csv");
XTest = testData(:,1:80);
YTest = testData(:,81);

[label,score] = predict(LiMdl,XTest);
