training = table2array(RtrainData);
testing = table2array(RtestData);
Xtrain = training(:,1:50);
Ytrain = training(:,51);
Xtest = testing(:,1:50);
Ytest = testing(:,51);
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
legend('AUC = 0.7272');
 

[xpos, ypos, ~, AUC1] = perfcurve(Ytest, score(:, 2), 1); % For the second class.
figure, plot(xpos, ypos);
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by SVM');
legend('AUC = 0.7272');




trn = table2array(Balancedtraining);
tst = table2array(Balancedtesting);
vald = table2array(Balancedvalidation);
testData =[trn;tst;vald];
testNew =[testData(:,1:16),testData(:,19),testData(:,21:22),testData(:,25:26),testData(:,29),testData(:,32:33),testData(:,36:37),testData(:,39),testData(:,42:44),testData(:,47),testData(:,49:50),testData(:,52:55),testData(:,59:60),testData(:,62:63),testData(:,70:71),testData(:,75:81)];
[training, ~, testing] = dividerand(testNew', 0.7, 0.0, 0.3); % Training (70%) and Testing (30%)
training = training';
testing = testing';
