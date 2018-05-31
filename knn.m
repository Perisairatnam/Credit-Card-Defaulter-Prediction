clear all; close all;
%Reading xlx file 
trainData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\RtrainData.csv");
X = trainData(:,1:50);
Y = trainData(:,51);
Ones = find(Y); % This returns the index for all '1's
Zeros = find(~Y); % This returns the index for all '0's
size(Ones)
size(Zeros)

KnnMdl = fitcknn(X,Y,'NumNeighbors',10,'Standardize',1);
KnnMdl.ClassNames
KnnMdl.Prior

testData = csvread("P:\New folder (3)\Course Material UST\Sem 4-Fall2017\Monday == Machine Learning 763\Project\Dataset\RtestData.csv");
%testData(:,[17,18,20,23,24,27,28,30,31,34,35,38,40,41,45,46,48,51,56,57,58,61,64,65,67,68,69,72,73,74]) =[];

XTest = testData(:,1:50);
YTest = testData(:,51);
[labell,scorel] = predict(KnnMdl,XTest); %test data prediction


%%%%
cfm10 = confusionmat(YTest, labell);

accuracy10 = sum(diag(cfm10))/sum(cfm10(:));
prcsn0 = cfm(1,1)/sum(cfm(:,1));
recall0 = cfm(1,1)/sum(cfm(1,:));
F1_0 = (2*prcsn0*recall0)/(recall0 + prcsn0);
prcsn1 = cfm(2,2)/sum(cfm(:,2));
recall1 = cfm(2,2)/sum(cfm(2,:)); % calculate recall for class 1 
F1_1 = (2*prcsn1*recall1)/(recall1 + prcsn1);

%ROC class 0
[xknn0, yknn0, ~, AUCknn0] = perfcurve(YTest, score(:, 1), 0); % For the first class. 
figure, plot(xknn0, yknn0); % Plot graph 
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate'); % insert X and Y labels 
title('\bf ROC for class 0 by KNN'); % insert main title 
legend('AUC 0');
%ROC class 1
[xknn1, yknn1, ~, AUCknn1] = perfcurve(YTest, score(:, 2), 1); % For the second class. 
figure, plot(xknn1, yknn1);
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by KNN');
legend('AUC 1');


% Convert the integer label vector to a class-identifier matrix.
[n,p] = size(XTest)
isLabels = unique(YTest);
nLabels = numel(isLabels)
[~,grpOOF] = ismember(label,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(YTest,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [num2cell(isLabels); {''}];
h.YTickLabel = [num2cell(isLabels); {''}];


