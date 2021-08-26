function printConfusion(bTest,bTrue)
% Find out quality measures for the classifier
truePositive = sum((bTest == 1) & (bTrue == 1))%#ok
falsePositive = sum((bTest == 1) & (bTrue == -1))%#ok
falseNegative = sum((bTest == -1) & (bTrue == 1))%#ok
trueNegative = sum((bTest == -1) & (bTrue == -1))%#ok
truePositiveRate = truePositive/(truePositive+falseNegative)%#ok Sensitivity
trueNegativeRate = trueNegative/(trueNegative+falsePositive)%#ok Specificity

% Correctly classified labels
accuracy = (truePositive+trueNegative)/...
    (truePositive+falsePositive+falseNegative+trueNegative)%#ok
end