%% Plot the distribution of digitTobeClassified and the rest
figure
logicalIdForDigitTobeClassifiedTrainingData = (training_data(1,:) == digitTobeClassified);
digitTobeClassifiedTrainingData = training_data(:,logicalIdForDigitTobeClassifiedTrainingData);
RestTrainingData = training_data(:,~logicalIdForDigitTobeClassifiedTrainingData);
plot(digitTobeClassifiedTrainingData(2,:),digitTobeClassifiedTrainingData(3,:),'b*');
hold on;
plot(RestTrainingData(2,:),RestTrainingData(3,:),'r.');
hold off;
xlabel('Intensity');
ylabel('Symmetry');