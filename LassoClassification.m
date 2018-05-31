% Combine training, testing and validation data
training = table2array(Balancedtraining);
testing = table2array(Balancedtesting);
validation = table2array(Balancedvalidation);

testData =[training;testing;validation];

X = testData(:,1:80);
Y = testData(:,81);

% Run lasso on balanced
[b, fitinfo] = lasso(X,Y,'CV',10,'Alpha',1);
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');

% using index 68, 
allData =[X(:,1),X(:,2),X(:,5),X(:,6),X(:,8),X(:,10),X(:,11),X(:,12),X(:,13),X(:,14),X(:,22),X(:,25),X(:,29),X(:,32),X(:,33),X(:,36),X(:,39),X(:,42),X(:,43),X(:,44),X(:,52),X(:,57),X(:,60),X(:,62),X(:,63),X(:,70),X(:,75),X(:,76),X(:,77),X(:,78),X(:,79),Y];
% Apply svm
[training, ~, testing] = dividerand(allData',0.7,0.0,0.3); % Training (60%), Testing (20%) and Validation (20%)
training = training';
% valInd = valInd';
testing = testing';
Xtrain = training(:,1:31);
Ytrain = training(:,32);
Xtest = testing(:,1:31);
Ytest = testing(:,32);
svm_mdl = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'Crossval', 'on');
[label, score] = predict(svm_mdl.Trained{10,1}, Xtest);
cfm = confusionmat(Ytest, label);
accuracy = sum(diag(cfm))/sum(cfm(:));
prcsn0 = cfm(1,1)/sum(cfm(:,1));
recall0 = cfm(1,1)/sum(cfm(1,:)); 
F1_0 = (2*prcsn0*recall0)/(recall0 + prcsn0);
prcsn1 = cfm(2,2)/sum(cfm(:,2));
recall1 = cfm(2,2)/sum(cfm(2,:)); % calculate recall for class 1
F1_1 = (2*prcsn1*recall1)/(recall1 + prcsn1);

[xpos, ypos, ~, AUC0] = perfcurve(Ytest, score(:, 1), 0); % For the first class.
figure, plot(xpos, ypos); % Plot graph
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate'); % insert X and Y labels
title('\bf ROC for class 0 by SVM'); % insert main title
legend('AUC = 0.7305');
 

[xpos, ypos, ~, AUC1] = perfcurve(Ytest, score(:, 2), 1); % For the second class.
figure, plot(xpos, ypos);
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by SVM');
legend('AUC = 0.7305');