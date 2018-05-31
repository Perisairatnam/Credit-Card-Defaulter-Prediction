%Auto generated Models: Prediction
%Logistic:
[labelL,scoreL] = trainedModelLogistic.predictFcn(XTest)  ; %auto generated model - prediction

cfmLog = confusionmat(labelL, YTest); %confusion matrix: Logistic

accuracyLog = sum(diag(cfmLog))/sum(cfmLog(:));
prcsnLog0 = cfmLog(1,1)/sum(cfmLog(:,1));
recallLog0 = cfmLog(1,1)/sum(cfmLog(1,:));
F1Log_0 = (2*prcsnLog0*recallLog0)/(recallLog0 + prcsnLog0);
prcsnLog1 = cfmLog(2,2)/sum(cfmLog(:,2));
recallLog1 = cfmLog(2,2)/sum(cfmLog(2,:)); % calculate recall for class 1 
F1Log_1 = (2*prcsnLog1*recallLog1)/(recallLog1 + prcsnLog1);

%ROC class 0
[xlog0, ylog0, ~, AUClog0] = perfcurve(YTest, scoreL(:, 1), 0); % For the first class. 
figure, plot(xlog0, ylog0); % Plot graph 
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate'); % insert X and Y labels 
title('\bf ROC for class 0 by Logistic'); % insert main title 
legend('AUC 0');
%ROC class 1
[xlog1, ylog1, ~, AUClog1] = perfcurve(YTest, scoreL(:, 2), 1); % For the second class. 
figure, plot(xlog1, ylog1);
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by Logistic');
legend('AUC 1');


% Convert the integer label vector to a class-identifier matrix.
[n,p] = size(XTest)
isLabels = unique(YTest);
nLabels = numel(isLabels)
[~,grpOOF] = ismember(yfitLog,isLabels); 
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


figure,
for i=1:trainedModelLogistic.GeneralizedLinearModel.NumCoefficients
    subplot(trainedModelLogistic.GeneralizedLinearModel.NumCoefficients, 1, 1)
    plotSlice(trainedModelLogistic.GeneralizedLinearModel, trainedModelLogistic.GeneralizedLinearModel.CoefficientNames{i});
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auto Generated model: Weighted KNN
[labelK,scoreK,costK] = trainedModelKNN.predictFcn(XTest)  ; %auto generated model - prediction

cfmKnn = confusionmat(labelK, YTest); %confusion matrix: Logistic

accuracyKnn = sum(diag(cfmKnn))/sum(cfmKnn(:));
prcsnKnn0 = cfmKnn(1,1)/sum(cfmKnn(:,1));
recallKnn0 = cfmKnn(1,1)/sum(cfmKnn(1,:));
F1Knn_0 = (2*prcsnKnn0*recallKnn0)/(recallKnn0 + prcsnKnn0);
prcsnKnn1 = cfmKnn(2,2)/sum(cfmKnn(:,2));
recallKnn1 = cfmKnn(2,2)/sum(cfmKnn(2,:)); % calculate recall for class 1 
F1Knn_1 = (2*prcsnKnn1*recallKnn1)/(recallKnn1 + prcsnKnn1);

%ROC class 0
[xpos, ypos, ~, AUC0] = perfcurve(YTest, scoreK(:, 1), 0); % For the first class. 
figure, plot(xpos, ypos); % Plot graph 
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate'); % insert X and Y labels 
title('\bf ROC for class 0 by KNN(Auto)'); % insert main title 
legend('AUC 0');
%ROC class 1
[xpos, ypos, ~, AUC1] = perfcurve(YTest, scoreK(:, 2), 1); % For the second class. 
figure, plot(xpos, ypos);
xlim([-0.05 1.05]), ylim([-0.05 1.05]);
xlabel('\bfFPrate'), ylabel('\bfTPrate');
title('\bf ROC for class 1 by KNN(Auto)');
legend('AUC 1');


% Convert the integer label vector to a class-identifier matrix.
[n,p] = size(XTest)
isLabels = unique(YTest);
nLabels = numel(isLabels)
[~,grpOOF] = ismember(yfitLog,isLabels); 
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

%%%%%%%%%%%%%%%%%%%
[labelSG,scoreSG,costSG] = trainedModelSVMG.predictFcn(XTest)  ; %auto generated model - prediction

cfmSVM = confusionmat(labelSG, YTest); %confusion matrix: Logistic
accuracySVM = sum(diag(cfmSVM))/sum(cfmSVM(:));
prcsnSVM0 = cfmSVM(1,1)/sum(cfmSVM(:,1));
recallSVM0 = cfmSVM(1,1)/sum(cfmSVM(1,:));
F1SVM_0 = (2*prcsnSVM0*recallSVM0)/(recallSVM0 + prcsnSVM0);
prcsnSVM1 = cfmSVM(2,2)/sum(cfmSVM(:,2));
recallSVM1 = cfmSVM(2,2)/sum(cfmSVM(2,:)); % calculate recall for class 1 
F1SVM_1 = (2*prcsnSVM1*recallSVM1)/(recallSVM1 + prcsnSVM1);
