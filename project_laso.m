mockData = defaultofcreditcardclients;
mockData.ID = [];
mockData.SEPTREPAYSTS = mockData.SEPTREPAYSTS + 3;
mockData.APRREPAYSTS = mockData.APRREPAYSTS + 3;
mockData.AUGREPAYSTS = mockData.AUGREPAYSTS + 3;
mockData.JULYREPAYSTS = mockData.JULYREPAYSTS + 3;
mockData.JUNEREPAYSTS = mockData.JUNEREPAYSTS + 3;
mockData.MAYREPAYSTS = mockData.MAYREPAYSTS + 3;
tmp = mockData.EDUCATION; % Education has variables that are not in the data description and hence these values are moved to 'Other' (4)
tmp((tmp == 0 | tmp == 5 | tmp == 6)) = 4; 
mockData.EDUCATION = tmp;

marital = mockData.MARRIAGE;
marital(marital == 0) = 3;
mockData.MARRIAGE = marital;

sex = dummyvar(mockData.SEX);
education = dummyvar(mockData.EDUCATION);
marriage = dummyvar(mockData.MARRIAGE);
septRepSts = dummyvar(mockData.SEPTREPAYSTS);
augRepSts = dummyvar(mockData.AUGREPAYSTS);
julRepSts = dummyvar(mockData.JULYREPAYSTS);
junRepSts = dummyvar(mockData.JUNEREPAYSTS);
mayRepSts = dummyvar(mockData.MAYREPAYSTS);
aprRepSts = dummyvar(mockData.APRREPAYSTS);

Xnum = [mockData.LIMIT_BAL,mockData.AGE, mockData.SEPTSTMT, mockData.AUGSTMT, mockData.JULYSTMT, mockData.JUNESTMT, mockData.MAYSTMT, mockData.APRSTMT,mockData.SEPTAMT,mockData.AUGAMT,mockData.JULYAMT,mockData.JUNEAMT,mockData.MAYAMT,mockData.APRAMT];
Xstnd = zscore(Xnum);

X = [Xstnd(:,1),sex(:,2),education(:,2:4),marriage(:,2:3),Xstnd(:,2),septRepSts(:,2:11),augRepSts(:,2:11),julRepSts(:,2:11),junRepSts(:,2:11),mayRepSts(:,2:11),aprRepSts(:,2:11),Xstnd(:,3:14)];
Y = [mockData.default];

% Divide data into training, testing and validation ratios.
testData = table2array(predictors)'; % Predictos is 30000x81 data with the last column being the response variable
[training, validation, testing] = dividerand(testData,0.6,0.2,0.2); % Training (60%), Testing (20%) and Validation (20%)
training = training';
validation = validation';
testing = testing';
dlmwrite('training.csv',training);
dlmwrite('testing.csv',testing);
dlmwrite('validation.csv',validation);

% Handle imbalance data problem
testData = table2array(predictors); % Original data
Y = testData(:,81); % Response variables
Ones = find(Y); % This returns the index for all '1's
Zeros = find(~Y); % This returns the index for all '0's
A = testData(Ones,:); % Get all rows with '1'  response variable
B = testData(Zeros,:); % Get all rows with '0'  response variable
[newB, ~, ~] = dividerand(B',0.3,0.5,0.2);
newB = newB';

testNew =[A;newB];
[training, validation, testing] = dividerand(testNew',0.6,0.2,0.2); % Training (60%), Testing (20%) and Validation (20%)

training = training';
validation = validation';
testing = testing';

dlmwrite('Balancedtraining.csv',training);
dlmwrite('Balancedtesting.csv',testing);
dlmwrite('Balancedvalidation.csv',validation);