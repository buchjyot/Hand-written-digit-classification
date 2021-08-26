%% Setup Workspace

% Classification of digit 1 from the rest using Linear Regression
close all;
clear;
clc;
digitTobeClassified = 1;

%% Read the training data from txt file
fileID = fopen('features_train.txt','r');% open file
formatSpec = '%f %f %f';% specifying the reading format
sizeA=[3 inf];% specifying the size of the data matrix
training_data = fscanf(fileID,formatSpec,sizeA);% reading the data matrix
fclose(fileID);

% Getting the size of the matrix data
training_data_length=length(training_data);

% Generate the data matrix A
A = [training_data(2,:)' training_data(3,:)' ones(length(training_data),1)];

% Generate the label matrix b
b = -ones(length(training_data),1);
b(training_data(1,:) == digitTobeClassified) = 1;

%% Read the testing data from txt file
fileID = fopen('features_test.txt','r');% open file
formatSpec = '%f %f %f';% specifying the reading format
sizeA=[3 inf];% specifying the size of the data matrix
testing_data = fscanf(fileID,formatSpec,sizeA);% reading the data matrix
fclose(fileID);

% Getting the size of the data matrix
testing_data_length = length(testing_data);

% Generate the data matrix Av for validation
Av = [testing_data(2,:)' testing_data(3,:)' ones(testing_data_length,1)];

% Generate the label matrix bv for validation
bTrue = -ones(testing_data_length,1);
bTrue(testing_data(1,:) == digitTobeClassified) = 1;

%% Least Square optimal solution - Only for Reference

% fprintf('***********************************************************\n');
% fprintf('### Least Square Optimal Solution using Normal Equation: \n');
% fprintf('***********************************************************\n');
xStarLeastSqr = (A'*A)\A'*b; 

%% Steepest descent with Armijo stepsize rule for linear regression

fprintf('*************************************************************\n');
fprintf('### Starting Steepest descent with Armijo step size rule:\n');
fprintf('*************************************************************\n');

% Initial guess for the algorithm
x0 = [1;-2;1];

% Objective Function
f = @(x) (0.5*norm(A*x-b));

% Gradient Function
g = @(x) (A'*(A*x-b));

% Steepest Descent with Armijo stepsize rule
x = x0;
sigma = 0.00001;
beta = 0.1;
s = 1;
epsilon = 1e-3;

nFeval = 1;
r = 1;
MAX_ITER = 10000;
obj = f(x);
gradient = g(x);

objForPlotting = zeros(1,MAX_ITER);
objForPlotting(r) = obj;
GradientNormForPlotting = zeros(1,MAX_ITER);
GradientNormForPlotting(r) = norm(gradient);
stateForPlotting = zeros(3,MAX_ITER);
stateForPlotting(:,r) = x0;

while norm(gradient) > epsilon && r < MAX_ITER
    % Steepest descent direction i.e. -grad
    direction = -gradient;
    
    % Start with stepsize = s
    alpha = s;
    newobj = f(x + alpha*direction);
    nFeval = nFeval+1;
    
    % Armijo stepsize rule check i.e. do we have sufficient descent?
    while (newobj-obj) > alpha*sigma*gradient'*direction
        alpha = alpha*beta;
        newobj = f(x + alpha*direction);
        nFeval = nFeval+1;
    end
    
    % Update the next state
    x = x + alpha*direction;
    
    % Print Status every 45 iterations
    if(mod(r,45)==1)
        fprintf('Iter:%5.0f \n',r);
        fprintf('Feval:%5.0f\n',nFeval);
        fprintf('OldObj:%5.5e\n',obj);
        fprintf('NewObj:%5.5e\n',newobj);
        fprintf('ReductionInObj:%5.5e\n',obj-newobj);
        fprintf('GradientNorm:%5.2f\n',norm(gradient));
        fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',x);
        fprintf('-----------------------------------------------------\n');
    end
    obj = newobj;
    gradient = g(x);
    r = r+1;
    stateForPlotting(:,r) = x;
    objForPlotting(r) = obj;
    GradientNormForPlotting(r) = norm(gradient);
end

% Print the final iteration
fprintf('Iter:%5.0f \n',r);
fprintf('Feval:%5.0f\n',nFeval);
fprintf('OldObj:%5.5e\n',obj);
fprintf('NewObj:%5.5e\n',newobj);
fprintf('ReductionInObj:%5.5e\n',obj-newobj);
fprintf('GradientNorm:%5.2f\n',norm(gradient));
fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',x);
fprintf('-----------------------------------------------------\n');

% Check MAX_ITER
if r == MAX_ITER
    fprintf('Maximum iteration limit reached.\n');
end

% Display optimal solution
xStarGradientDescent = x %#ok

% Plot the reduction in gradient norm and objective reduction
figure(1)
plot(1:r,objForPlotting(1:r),'LineWidth',2);
xlabel('Iterations');ylabel('Objective Value');grid on;
legend('Gradient Descent with Armijo stepsize rule');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

figure(2)
plot(1:r,GradientNormForPlotting(1:r),'LineWidth',2);
xlabel('Iterations');ylabel('Norm of the Gradient');grid on;
legend('Gradient Descent with Armijo stepsize rule');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

% Plot states for gradient descent
figure
plot(1:r,stateForPlotting(:,1:r),'LineWidth',2)
xlabel('Iterations');ylabel('States');grid on;
legend('x(1)','x(2)','x(3)');
title('Gradient Descent Performance')

% Plot the saperating boundry
createFigureAndPlotTrainingData
hold on;
equationOflineGD = @(a1,a2) (xStarGradientDescent(1)*a1 +...
    xStarGradientDescent(2)*a2 + xStarGradientDescent(3));
h = fimplicit(equationOflineGD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Training Data: Classification boundry with Gradient Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

% Visulize how the line changes as algorithm progresses
createFigureAndPlotTrainingData
hold on;
for i = 1:r
    if(mod(i,20)==1 || i==1)
        eqOflineGD = @(a1,a2) (stateForPlotting(1,i)*a1 +...
            stateForPlotting(2,i)*a2 + stateForPlotting(3,i));
        h = fimplicit(eqOflineGD,[get(gca,'XLim'),get(gca,'YLim')]);
        set(h,'LineWidth',2,'Color','green','LineStyle','--');
        hold on;
    end
end
h = fimplicit(equationOflineGD,[get(gca,'XLim'),get(gca,'YLim')]);
set(h,'LineWidth',3,'Color','magenta');
title('Training Data: Change in classification boundry: Gradient Descent');
grid on;
hold off;

% Use the optimal model derived from gradient descent to classify a digit
% in the test/validation data Compute Av*xStarGradientDescent
bClassifierTest = Av*xStarGradientDescent;
bTest = sign(bClassifierTest);

% Print Confusion Matrices
printConfusion(bTest,bTrue);

% Plot the results in figure
createFigureAndPlotTestingData
h = fimplicit(equationOflineGD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Testing Data: Classification boundry with Gradient Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

%% Coordinate descent for linear regression

fprintf('*************************************************************\n');
fprintf('### Starting coordinate descent: \n');
fprintf('*************************************************************\n');

% Initial guess for the algorithm
x0 = [1;-2;1];

% Objective Function
f = @(x) (0.5*norm(A*x-b));

% Gradient Function
g = @(x) (A'*(A*x-b));

% Coordinate Descent
r = 1;
iter = r;
MAX_ITER = 100000;
x = zeros(3,MAX_ITER);
epsilon = sqrt(eps);

x(:,r) = x0;
obj = f(x(:,r));
gradient = g(x(:,r));

objForPlotting = zeros(1,MAX_ITER);
objForPlotting(r) = obj;
GradientNormForPlotting = zeros(1,MAX_ITER);
GradientNormForPlotting(r) = norm(gradient);
stateForPlotting = zeros(3,MAX_ITER);
stateForPlotting(:,r) = x(:,r);
prevState = stateForPlotting(:,r);
ReductionInObj = 1;

% Stopping Criteria for the algorithm
while norm(ReductionInObj) > epsilon
    for i = 1:3
        for j = 1:3
            if ~isequal(i,j)
                % If i~=j then just copy the values.
                x(j,r+1) = x(j,r);
            else
                % That means i==j
                % Remove the ith/jth column and call it Aj
                Aj = A;
                Aj(:,i)=[];
                xj = x(:,r);
                xj(i) = [];
                
                % Update the x(i,r+1)
                x(i,r+1) = A(:,i)'*A(:,i)\A(:,i)'*(b-Aj*xj);
            end
        end
        % Increment the r
        r = r + 1;
        
        % If we reach maximum limit then break the loop
        if r == MAX_ITER
            break;
        end
    end
    
    % Increment the iteration count and save the plotting state after
    % iteration
    prevState = stateForPlotting(:,iter);
    iter = iter + 1;
    stateForPlotting(:,iter) = x(:,r);
    GradientNormForPlotting(iter) = norm(g(x(:,r)));
    updatedState = stateForPlotting(:,iter);
    OldObj = f(prevState);
    NewObj = f(updatedState);
    ReductionInObj = OldObj-NewObj;
    objForPlotting(iter) = NewObj;
    
    % Print Status
    if(mod(iter,10)==1)
        fprintf('Iter:%5.0f \n',iter);
        fprintf('OldObj:%5.5e\n',OldObj);
        fprintf('NewObj:%5.5e\n',NewObj);
        fprintf('ReductionInObj:%5.5e\n',ReductionInObj);
        fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',updatedState);
        fprintf('-----------------------------------------------------\n');
    end
    
    % Check MAX_ITER break the outer loop
    if r == MAX_ITER
        fprintf('Maximum iteration limit reached.\n');
        break;
    end
end

% Print Optimal Solution
xStarCoordinateDescent = stateForPlotting(:,iter) %#ok

% Plot the boundry
createFigureAndPlotTrainingData
hold on;
equationOflineCD = @(a1,a2) (xStarCoordinateDescent(1)*a1 +...
    xStarCoordinateDescent(2)*a2 + xStarCoordinateDescent(3));
h = fimplicit(equationOflineCD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Training Data: Classification boundry with Coordinate Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

% Visulize how the line changes as algorithm progresses
createFigureAndPlotTrainingData
for i = 1:iter
    if(mod(i,10)==1 || i==1)
        hold on;
        eqOflineCD = @(a1,a2) (stateForPlotting(1,i)*a1 +...
            stateForPlotting(2,i)*a2 + stateForPlotting(3,i));
        h = fimplicit(eqOflineCD,[get(gca,'XLim'),get(gca,'YLim')]);
        set(h,'LineWidth',2,'Color','green','LineStyle','--');
    end
end
h = fimplicit(equationOflineCD,[get(gca,'XLim'),get(gca,'YLim')]);
set(h,'LineWidth',3,'Color','magenta');
title('Training Data: Change in classification boundry: Coordinate Descent');
grid on;
hold off;

% Plot the reduction in gradient norm and objective reduction
figure(1)
hold on;
plot(1:iter,objForPlotting(1:iter),'LineWidth',2);
hold off;
legend('Gradient Descent with Armijo stepsize rule','Coordinate Descent');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

figure(2)
hold on;
plot(1:iter,GradientNormForPlotting(1:iter),'LineWidth',2);
hold off;
legend('Gradient Descent with Armijo stepsize rule','Coordinate Descent');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

% Plot coordinate descent performance
figure
plot(1:iter,stateForPlotting(:,1:iter),'LineWidth',2)
xlabel('Iterations');ylabel('States');grid on;
legend('x(1)','x(2)','x(3)');
title('Coordinate Descent Performance')

% Use the optimal model derived from coordinate descent to classify a digit
% in the test/validation data
% Compute Av*xStarGradientDescent
bClassifierTest = Av*xStarCoordinateDescent;
bTest = sign(bClassifierTest);

% Print Confusion Matrices
printConfusion(bTest,bTrue);

% Plot the results in figure
createFigureAndPlotTestingData
hold on;
h = fimplicit(equationOflineCD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Testing Data: Classification boundry with Coordinate Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

%% Support Vector Machine using coordinate descent

fprintf('**************************************************************\n');
fprintf('### Starting Support Vector Machine using Coordinate Descent: \n');
fprintf('*************************************************************\n');

% Fix c initially
c = 100;

% Start with r = 1
r = 1;

% Number of iterations are 1 initially
iter = r;

% Maximum iterations
MAX_ITER = 10000;

% Allocate memory for the primal variable
x = zeros(3,MAX_ITER);

% Allocate memory for fullgradient
fullgradient = zeros(training_data_length,MAX_ITER);

% Allocate memory for dual varialbe
lambda = zeros(training_data_length,MAX_ITER);

% Stopping critera
epsilon = 1e-2;

% Generate the random vector lambda for initial state
% 0 <= lambda <= c
lambda(:,r) = zeros(training_data_length,1);

% Dual objective to be maximized
dualObj = @(xt,lambda) sum(lambda) - 0.5*norm(xt)^2;

% Projection on to 0 <= x <= c
proj = @(x) max(min(x,c),0);

% ith Gradient Function
ithgrad = @(i,calulatedPrimal) 1-(b(i)*A(i,:)*calulatedPrimal);

% Full gradient
fullgrad = @(x) 1 - b.*A*x;

% Dual to primal variable update equation
primalClaculate = @(lambdaIn) sum((lambdaIn.*b).*A);

% Initial x0 is given by sum of all lambda(i)*b(i)*A(i,:)
% i.e. Update the initial guess for the primal variable
x(:,r) = primalClaculate(lambda(:,r));
fullgradient(:,r) = fullgrad(x(:,r));

% Plotting varialbes
dualobjForPlotting = zeros(1,MAX_ITER);
dualobjForPlotting(r) = dualObj(x(:,r),lambda(:,r));

while true
    xt = x(:,iter);
    lambdaold = lambda(:,iter);
    OldObj = dualObj(xt,lambdaold);
    lambdanew = lambdaold;
    
    for i = 1:training_data_length
        % Compute ith gradient
        grad = ithgrad(i,xt);
        
        % Update single coordinate at a time
        % If i==j then update the lambda vector i.e. ith coordinate
        lambdanew(i) = proj(lambdaold(i)+(b(i)^2*norm(A(i,:)')^2)\(grad));
        
        % Update (r+1)th Primal based on lambda
        xt = xt + (lambdanew(i)-lambdaold(i))*b(i)*A(i,:)';
        
        % Update fullgradient
        fullgradient (i,r) = grad;
        
        % Increment the r
        r = r + 1;
        
        % If we reach maximum limit then break the loop
        if r == MAX_ITER
            break;
        end
    end
    
    % Store the primal solution after one iteration
    x(:,iter+1) = xt;
    deltax = x(:,iter) - x(:,iter+1);
    lambda(:,iter+1) = lambdanew;
    
    % Dual Objective will be increesing
    NewObj = dualObj(xt,lambdanew);
    
    % save the plotting state after iteration
    prevState = x(:,iter);
    newState = x(:,iter+1);
    
    % Dual objective should be increasing
    increaseInObj = NewObj-OldObj;
    dualobjForPlotting(iter+1) = NewObj;
    
    % Print Status
    fprintf('Iter:%5.0f \n',iter);
    fprintf('OldObj:%5.5e\n',OldObj);
    fprintf('NewObj:%5.5e\n',NewObj);
    fprintf('IncreeseInObj:%5.5e\n',increaseInObj);
    fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',newState);
    fprintf('-----------------------------------------------------\n')
    
    % Increment the iteration count
    iter = iter + 1;
    
    % Stopping creteria
    if norm(deltax) < epsilon
        if norm(proj(lambda(:,iter) - fullgradient(:,iter))-lambda(:,iter))...
                < epsilon % Stopping creteria
            break;
        end
    end
    
    % Check MAX_ITER break the outer loop
    if iter == MAX_ITER
        fprintf('Maximum iteration limit reached.\n');
        break;
    end
end

% Print the optimal solution with SVM
xStarSVMCD = xt %#ok<NOPTS>

% Plot the SVM boundry
createFigureAndPlotTrainingData
hold on;
equationOflineSVMCD = @(a1,a2) (xStarSVMCD(1)*a1 + ...
    xStarSVMCD(2)*a2 + xStarSVMCD(3));
equationOflineSVMCDBound1 = @(a1,a2) (xStarSVMCD(1)*a1 + ...
    xStarSVMCD(2)*(a2+1) + xStarSVMCD(3));
equationOflineSVMCDBound2 = @(a1,a2) (xStarSVMCD(1)*a1 + ...
    xStarSVMCD(2)*(a2-1) + xStarSVMCD(3));
h = fimplicit(equationOflineSVMCD,[get(gca,'XLim'),get(gca,'YLim')]);
h1 = fimplicit(equationOflineSVMCDBound1,[get(gca,'XLim'),get(gca,'YLim')]);
h2 = fimplicit(equationOflineSVMCDBound2,[get(gca,'XLim'),get(gca,'YLim')]);
title('Training Data: Classification boundry for SVM using Coordinate Descent');
set(h,'LineWidth',2,'Color','magenta');
set(h1,'LineWidth',2,'Color','cyan');
set(h2,'LineWidth',2,'Color','cyan');
grid on;
hold off;

% Plot how dual objective increeses as iteration progresses
figure
plot(1:iter,dualobjForPlotting(1:iter),'LineWidth',2);
ylabel('Objective Value')
xlabel('Iteration')
title('Increese in the dual objective value for Support Vector Machine');

% Use the optimal model derived from SVM to classify a digit in the
% test/validation data
% Compute Av*xStarGradientDescent
bClassifierTest = Av*xStarSVMCD;
bTest = sign(bClassifierTest);

% Print Confusion Matrices
printConfusion(bTest,bTrue);

% Plot the results in figure
createFigureAndPlotTestingData
h = fimplicit(equationOflineSVMCD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Testing Data: Classification boundry for SVM using Coordinate Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

%% Newton's method with constant stepsize rule
fprintf('*************************************************************\n');
fprintf(['### Starting Newton method with constant stepsize',...
    ' rule for linear regression problem: \n']);
fprintf('*************************************************************\n');

% Initial guess for the algorithm
x0 = [1;-2;1];

% Objective Function
f = @(x) (0.5*norm(A*x-b));

% Gradient Function
g = @(x) (A'*(A*x-b));

% Hessian Computation
h = A'*A;

% Newton's method with constant stepsize rule
x = x0;
% Stepsize is inverse of the maximum eigen value of hessian, algorithm
% stops at 828493 iteration
alpha =  1/max(eig(h)); %#ok<NASGU>
% Selecting higher stepsize makes algorithm to converge faster
alpha = 0.01;

epsilon = sqrt(eps);
r = 1;
MAX_ITER = 10000;
obj = f(x);
gradient = g(x);

objForPlotting = zeros(1,MAX_ITER);
objForPlotting(r) = obj;
GradientNormForPlotting = zeros(1,MAX_ITER);
GradientNormForPlotting(r) = norm(gradient);
stateForPlotting = zeros(3,MAX_ITER);
stateForPlotting(:,r) = x0;
ReductionInObj = 1;

while abs(ReductionInObj) > epsilon
    % Steepest descent direction i.e. -grad
    direction = -h\gradient;
    
    % Update the next state, Newton's Iteration
    x = x + alpha*direction;
    
    % Compute New Objective value
    newobj = f(x);
    ReductionInObj = obj-newobj;
    
    % Print Status every 45 iterations
    if(mod(r,45)==1)
        fprintf('Iter:%5.0f \n',r);
        fprintf('OldObj:%5.5e\n',obj);
        fprintf('NewObj:%5.5e\n',newobj);
        fprintf('ReductionInObj:%5.5e\n',ReductionInObj);
        fprintf('GradientNorm:%5.2f\n',norm(gradient));
        fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',x);
        fprintf('-----------------------------------------------------\n')
    end
    
    % For the next iteration
    obj = newobj;
    gradient = g(x);
    r = r+1;
    
    % For plotting
    stateForPlotting(:,r) = x;
    objForPlotting(r) = obj;
    GradientNormForPlotting(r) = norm(gradient);
end

% Print the final iteration
fprintf('Iter:%5.0f \n',r);
fprintf('OldObj:%5.5e\n',obj);
fprintf('NewObj:%5.5e\n',newobj);
fprintf('ReductionInObj:%5.5e\n',ReductionInObj);
fprintf('GradientNorm:%5.2f\n',norm(gradient));
fprintf('x(1):%.4d | x(2):%.4d | x(3):%.4d\n',x);
fprintf('-----------------------------------------------------\n')

% Check MAX_ITER
if r == MAX_ITER
    fprintf('Maximum iteration limit reached.\n');
end

% Optimal Solution using Newton's method
xStarNewton = x %#ok<NOPTS>

% Plot the reduction in gradient norm and objective reduction
figure(1)
hold on;
plot(1:r,objForPlotting(1:r),'LineWidth',2);
legend('Gradient Descent with Armijo stepsize rule',...
    'Coordinate Descent','Newton Method with Constant stepsize');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

figure(2)
hold on;
plot(1:r,GradientNormForPlotting(1:r),'LineWidth',2);
legend('Gradient Descent with Armijo stepsize rule',...
    'Coordinate Descent','Newton Method with Constant stepsize');
title('Algorithm Performance');
set(gca,'XLim',[0 500]);

figure
plot(1:r,stateForPlotting(:,1:r),'LineWidth',2)
xlabel('Iterations');ylabel('States');grid on;
legend('x(1)','x(2)','x(3)');
title('Newton Method Performance')

% Plot the boundry
createFigureAndPlotTrainingData
hold on;
equationOflineNewton = @(a1,a2) (xStarNewton(1)*a1 + ...
    xStarNewton(2)*a2 + xStarNewton(3));
h = fimplicit(equationOflineNewton,[get(gca,'XLim'),get(gca,'YLim')]);
title(sprintf('Training Data: Classification boundry with Newton Method'));
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;

% Visulize how the line changes as algorithm progresses
createFigureAndPlotTrainingData
for i = 1:r
    if(mod(i,20)==1 || i==1)
        hold on;
        eqOflineNewton = @(a1,a2) (stateForPlotting(1,i)*a1 +...
            stateForPlotting(2,i)*a2 + stateForPlotting(3,i));
        h = fimplicit(eqOflineNewton,[get(gca,'XLim'),get(gca,'YLim')]);
        set(h,'LineWidth',2,'Color','green','LineStyle','--');
    end
end
h = fimplicit(equationOflineGD,[get(gca,'XLim'),get(gca,'YLim')]);
set(h,'LineWidth',3,'Color','magenta');
title('Training Data: Change in classification boundry with Newton Method');
grid on;
hold off;

% Use the optimal model derived from SVM to classify a digit in the
% test/validation data
% Compute Av*xStarGradientDescent
bClassifierTest = Av*xStarNewton;
bTest = sign(bClassifierTest);

% Print Confusion Matrices
printConfusion(bTest,bTrue);

% Plot the results in figure
createFigureAndPlotTestingData
h = fimplicit(equationOflineSVMCD,[get(gca,'XLim'),get(gca,'YLim')]);
title('Testing Data: Classification boundry for SVM using Coordinate Descent');
set(h,'LineWidth',2,'Color','magenta');
grid on;
hold off;