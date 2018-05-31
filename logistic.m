
%%%%
clear all; close all;
%Reading xlx file 
trainData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\trainBal.csv");
X = trainData(:,1:80);
Y = trainData(:,81);
LoMdl = fitglm(X, Y, 'distr', 'binomial', 'link', 'logit')  %Logistic

[b fitinfo] = lassoglm(X, Y,  'binomial','NumLambda',25, 'Alpha', 1);
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');

lassoPlot(b, fitinfo, 'PlotType', 'CV');

trainData(:,[17,18,20,23,24,27,28,30,31,34,35,38,40,41,45,46,48,51,56,57,58,61,64,65,67,68,69,72,73,74]) =[];

[b1 fitinfo1] = lassoglm(trainData(:,1:50), trainData(:,51),  'binomial','NumLambda',25,'CV',10, 'Alpha', 1);
lassoPlot(b1,fitinfo1,'PlotType','Lambda','XScale','log');


indx = fitinfo.Index1SE;
B0 = b(:,indx);
nonzeros = sum(B0 ~= 0)

p = LoMdl.Fitted.Response;
Z = LoMdl.Fitted.LinearPredictor;
figure, gscatter(Z, p, Y); grid on


testData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\Balancedtesting.csv");
XTest = testData(:,1:80);
YTest = testData(:,81);
[label,score] = predict(LoMdl,XTest);


accuracy = sum(diag(cfm))/sum(cfm(:));
prcsn0 = cfm(1,1)/sum(cfm(:,1));
recall0 = cfm(1,1)/sum(cfm(1,:));
F1_0 = (2*prcsn0*recall0)/(recall0 + prcsn0);
prcsn1 = cfm(2,2)/sum(cfm(:,2));
recall1 = cfm(2,2)/sum(cfm(2,:)); % calculate recall for class 1 
F1_1 = (2*prcsn1*recall1)/(recall1 + prcsn1);

figure, 
subplot(4, 1, 1), plot(trainedModelLogistic.GeneralizedLinearModel)
subplot(4, 1, 2), plotDiagnostics(trainedModelLogistic.GeneralizedLinearModel)
subplot(4, 1, 3), plotDiagnostics(trainedModelLogistic.GeneralizedLinearModel,'cookd')
subplot(4, 1, 4), plotResiduals(trainedModelLogistic.GeneralizedLinearModel)

