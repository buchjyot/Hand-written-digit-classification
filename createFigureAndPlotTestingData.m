figure;
logicalIdForDigitTobeClassifiedTestingData = (testing_data(1,:) == digitTobeClassified);
digitTobeClassifiedTestingData = testing_data(:,logicalIdForDigitTobeClassifiedTestingData);
RestTestingData = testing_data(:,~logicalIdForDigitTobeClassifiedTestingData);
plot(digitTobeClassifiedTestingData(2,:),digitTobeClassifiedTestingData(3,:),'b*');
hold on;
plot(RestTestingData(2,:),RestTestingData(3,:),'r.');
xlabel('Intensity')
ylabel('Symmetry')