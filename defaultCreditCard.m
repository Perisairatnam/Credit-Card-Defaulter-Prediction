testData = defaultofcreditcardclients;
testData.SEPTREPAYSTS = testData.SEPTREPAYSTS + 2;
testData.APRREPAYSTS = testData.APRREPAYSTS + 2;
testData.AUGREPAYSTS = testData.AUGREPAYSTS + 2;
testData.JULYREPAYSTS = testData.JULYREPAYSTS + 2;
testData.JUNEREPAYSTS = testData.JUNEREPAYSTS + 2;
testData.MAYREPAYSTS = testData.MAYREPAYSTS + 2;
testData.ID = [];

X = testData(:,1:23);
Y = [testData.default]; % Dependent variable

svm_mdl = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Crossval', 'on', 'Standardize', true);
[label, score] = predict(svm_mdl.Trained{8,1}, X);
cfm = confusionmat(Y, label);
accuracy = sum(diag(cfm))/sum(cfm(:)); % Calculate the accuracy of the confusion matrix
prcsn0 = cfm(1,1)/sum(cfm(:,1));
recall0 = cfm(1,1)/sum(cfm(1,:)); % calculate recall for class 0
F1_0 = (2*prcsn0*recall0)/(recall0 + prcsn0);
prcsn1 = cfm(2,2)/sum(cfm(:,2));
recall1 = cfm(2,2)/sum(cfm(2,:)); % calculate recall for class 1
F1_1 = (2*prcsn1*recall1)/(recall1 + prcsn1);



testData = table2array(predictors);
Y = testData(:,81);
X = testData(:,1:80);
[b, fitinfo] = lasso(X,Y,'CV',10,'Alpha',1);
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');
allData =[X(:,1),X(:,2),X(:,5),X(:,6),X(:,10),X(:,11),X(:,12),X(:,13),X(:,14),X(:,22),X(:,23),X(:,25),X(:,26),X(:,27),X(:,29),X(:,32),X(:,33),X(:,39),X(:,42),X(:,44),X(:,49),X(:,52),X(:,53),X(:,57),X(:,60),X(:,62),X(:,63),X(:,75),X(:,76),Y];
[training, ~, testing] = dividerand(allData',0.7,0.0,0.3); % Training (60%), Testing (20%) and Validation (20%)
training = training';
% valInd = valInd';
testing = testing';
Xtrain = training(:,1:29);
Ytrain = training(:,30);
Xtest = testing(:,1:29);
Ytest = testing(:,30);

Xnew =[X(:,1),X(:,2),X(:,5),X(:,6),X(:,10),X(:,11),X(:,12),X(:,13),X(:,14),X(:,22),X(:,23),X(:,25),X(:,26),X(:,27),X(:,29),X(:,32),X(:,33),X(:,39),X(:,42),X(:,44),X(:,49),X(:,52),X(:,53),X(:,57),X(:,60),X(:,62),X(:,63),X(:,75),X(:,76)];
svm_mdl = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'Crossval', 'on');
[label, score] = predict(svm_mdl.Trained{10,1}, Xtest);

cfm = confusionmat(Ytest, label);
accuracy = sum(diag(cfm))/sum(cfm(:)); % Calculate the accuracy of the confusion matrix
prcsn0 = cfm(1,1)/sum(cfm(:,1));
recall0 = cfm(1,1)/sum(cfm(1,:)); 
F1_0 = (2*prcsn0*recall0)/(recall0 + prcsn0);
prcsn1 = cfm(2,2)/sum(cfm(:,2));
recall1 = cfm(2,2)/sum(cfm(2,:)); % calculate recall for class 1
F1_1 = (2*prcsn1*recall1)/(recall1 + prcsn1);



[xpos, ypos, ~, AUC0] = perfcurve(Ytest, score(:, 1), 0); % For the first class.
figure, plot(xpos, ypos); % Plot graph
% xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate'); % insert X and Y labels
title('\bf ROC for class 0 by SVM'); % insert main title
legend('AUC = 0.8022');


[xpos, ypos, ~, AUC1] = perfcurve(Ytest, score(:, 2), 1); % For the second class.
figure, plot(xpos, ypos);
% xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by SVM');
legend('AUC = 0.7979');
