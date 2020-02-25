  %Author: Michael Busa
%Homework #2: Q1 part 2
%Date: 2/24/2020
%Purpose: Import Data from the saved .mat file, evaulate the gaussian pdfs 
%         and find the ROC curve for true and false positive probs


%Load in saved dataset
clear
clc
N=10000;
load('d_validate_10k.mat');
%Separate actual data from true class labels for each point
samples = data_set(1:2,:);
true_class_labels = data_set(3,:);

%Prepare to separate data
class0_data = zeros(2,N);
class1_data = zeros(2,N);
class0_sample_num = 0;
class1_sample_num = 0;

%Class priors, gaussian parameters
class0_prior = .9;
class1_prior = .1;

mu0 = [-2;0];
mu1 = [2;0];

sig0=[1 -.9;-.9 2];
sig1=[2 .9;.9 1];

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(:,n)=samples(:,n);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(:,n)=samples(:,n);
        class0_sample_num=class0_sample_num+1;
    end
end

%Calculate class pdfs using evalGaussian function
pdf0 = evalGaussian(samples,mu0,sig0);
pdf1 = evalGaussian(samples,mu1,sig1);
discriminant = log(pdf1)-log(pdf0);

%Decision rule with a varying Threshold
increments = 10000;
probabilities = zeros(increments,2);
threshold_values = zeros(increments,1);
row_counter=1;
min_pError = 1;
for i=increments:-1:1
    lambda = [1 1-(i/increments); (i/increments) 1]; %By changing the lambdas, the threshold gradually increases
    gamma = ((lambda(1,1)-lambda(2,1))/(lambda(2,2)-lambda(1,2)))*(class0_prior/class1_prior);
    threshold_values(row_counter) = gamma;
    dec = (discriminant>=log(gamma));
    false_pos = find(dec==1 & true_class_labels==0); p10 = length(false_pos)/class0_sample_num; % probability of false positive
    true_pos = find(dec==1 & true_class_labels==1); p11 = length(true_pos)/class1_sample_num; % probability of true positive
    probabilities(row_counter,:) = [p10,p11];
    row_counter = row_counter+1;
    pError = p10 + (1-p11); %(1-p(true pos) + p(false pos))
    if pError<min_pError
        min_pError = pError;
        min_probs = [p10,p11];
    end
end



min_pError = (min_probs(1)+(1-min_probs(2)))/2;

%Plot the ROC Curve with the minimum p(error) point marked
figure(1)
scatter(probabilities(:,1),probabilities(:,2),'b.') %plot of false positive vs true positive
hold on
plot(min_probs(1),min_probs(2), '-p','Markersize', 15, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red')
txt = strcat('  \leftarrow Min P(error) at [',num2str(min_probs(1)));
txt = strcat(txt,',');
txt = strcat(txt,num2str(min_probs(2)));
txt = strcat(txt,']');
text(min_probs(1),min_probs(2),txt);
title('ROC Curve of Classifier')
xlabel('p(False Positive)'),ylabel('p(True Positive)')
legend('ROC', 'Min P(error)', 'Location', 'Northeast')
axis equal

%Plot of the data set
figure(2),
scatter(class0_data(1,:),class0_data(2,:),'or')
hold on
scatter(class1_data(1,:),class1_data(2,:),'b+')
title('Saved Dataset from Given Gaussians')
legend('Class 0', 'Class 1', 'Location', 'North')
xlabel('x1'),ylabel('x2')
hold off
%Print estimate of min pError
fprintf('The estimate of the minimum p(Error) is %.4f',min_pError);

%lambda and gamma values
lam=[0 1;1 0]; %loss values 0-1 loss
gam=class0_prior/class1_prior; %with 0-1 loss, gamma is just division of the class priors ((1-0/1-0)*(p0/p1))

%Decide on a class
dec = (discriminant>=log(gam));
true_neg = find(dec==0 & true_class_labels==0); 
false_pos = find(dec==1 & true_class_labels==0); 
false_neg = find(dec==0 & true_class_labels==1); 
true_pos = find(dec==1 & true_class_labels==1); 

figure(3)
hold on
plot(samples(1,true_neg),samples(2,true_neg),'oc')
plot(samples(1,false_pos),samples(2,false_pos),'*m')
plot(samples(1,false_neg),samples(2,false_neg),'xm')
plot(samples(1,true_pos),samples(2,true_pos),'+c')
axis equal,

% Prepare figure for boundary
horizontalGrid = linspace(floor(min(samples(1,:))),ceil(max(samples(1,:))),101);
verticalGrid = linspace(floor(min(samples(2,:))),ceil(max(samples(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
GridValues = log(evalGaussian([h(:)';v(:)'],mu1,sig1))-log(evalGaussian([h(:)';v(:)'],mu0,sig0)) - log(gam);
min_vals = min(GridValues);max_vals = max(GridValues);
discriminantGrid = reshape(GridValues,91,101);

%Plot the boundary
contour(horizontalGrid,verticalGrid,discriminantGrid,[min_vals*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*max_vals]); % plot equilevel contours of the discriminant function 
legend('Class 0 Correct','Class 0 Incorrect','Class 1 Incorrect','Class 1 Correct', 'Decision Boundary', 'Location', 'Southeast')
title('Original Data, Classifier Decisions'),
xlabel('x1'), ylabel('x2')

%% Train with 10 samples
%clear all
N=10;
load('d_train_10.mat');
samples = data_set(1:2,:)';
true_class_labels = data_set(3,:)';

%Prepare to separate data
class0_data = zeros(N,2);
class1_data = zeros(N,2);
class0_sample_num = 0;
class1_sample_num = 0;

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(n,:)=samples(n,:);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(n,:)=samples(n,:);
        class0_sample_num=class0_sample_num+1;
    end
end

%From Suggestion from professor, do it linear first to initialize first
%theta values
z_lin = [ones(N,1) samples];
t_init = zeros(3,1);
t_lin = GD(z_lin,N,true_class_labels,t_init,1,1000);
%Then use these linear values to calculate the quadratic values
z_quad = [ones(N,1) samples(:,1) samples(:,2) (samples(:,1)).^2 samples(:,1).*samples(:,2) (samples(:,2)).^2];
t_init_quad = [t_lin;zeros(3,1)];
t = GD(z_quad,N,true_class_labels,t_init_quad,1,1000);

%Plot of the data set
figure(4),
scatter(class0_data(1:class0_sample_num,1),class0_data(1:class0_sample_num,2),'or')
hold on
scatter(class1_data(class0_sample_num+1:N,1),class1_data(class0_sample_num+1:N,2),'b+')
title('Classifier using 10 points for Training')

% Prepare figure for boundary
horizontalGrid = linspace(floor(min(samples(:,1))-1),ceil(max(samples(:,1))+1),101);
verticalGrid = linspace(floor(min(samples(:,2))-4),ceil(max(samples(:,2))+2),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
z_test = [ones((91*101),1) h(:) v(:) h(:).^2 h(:).*v(:) v(:).^2];
sigmoid = 1./(1+exp(-z_test*t));
min_vals = min(sigmoid);max_vals = max(sigmoid);
discriminantGrid = reshape(sigmoid,91,101);

%Plot the boundary
contour(horizontalGrid,verticalGrid,discriminantGrid,1); % plot equilevel contours of the discriminant function 
xlabel('x1'), ylabel('x2')
legend('Class 1', 'Class 2', 'GD Decision Boundary')
%% Train with 100 samples
%clear all
N=100;
load('d_train_100.mat');
samples = data_set(1:2,:)';
true_class_labels = data_set(3,:)';

%Prepare to separate data
class0_data = zeros(N,2);
class1_data = zeros(N,2);
class0_sample_num = 0;
class1_sample_num = 0;

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(n,:)=samples(n,:);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(n,:)=samples(n,:);
        class0_sample_num=class0_sample_num+1;
    end
end

%From Suggestion from professor, do it linear first to initialize first
%theta values
z_lin = [ones(N,1) samples];
t_init = zeros(3,1);
t_lin = GD(z_lin,N,true_class_labels,t_init,1,1000);
%Then use these linear values to calculate the quadratic values
z_quad = [ones(N,1) samples(:,1) samples(:,2) (samples(:,1)).^2 samples(:,1).*samples(:,2) (samples(:,2)).^2];
t_init_quad = [t_lin;zeros(3,1)];
t = GD(z_quad,N,true_class_labels,t_init_quad,1,1000);

%Plot of the data set
figure(5),
scatter(class0_data(1:class0_sample_num,1),class0_data(1:class0_sample_num,2),'or')
hold on
scatter(class1_data(class0_sample_num+1:N,1),class1_data(class0_sample_num+1:N,2),'b+')
title('Classifier using 100 points for Training')

% Prepare figure for boundary
horizontalGrid = linspace(floor(min(samples(:,1))),ceil(max(samples(:,1))),101);
verticalGrid = linspace(floor(min(samples(:,2))-3),ceil(max(samples(:,2))+3),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
z_test = [ones((91*101),1) h(:) v(:) h(:).^2 h(:).*v(:) v(:).^2];
sigmoid = 1./(1+exp(-z_test*t));
min_vals = min(sigmoid);max_vals = max(sigmoid);
discriminantGrid = reshape(sigmoid,91,101);

%Plot the boundary
contour(horizontalGrid,verticalGrid,discriminantGrid,1); % plot equilevel contours of the discriminant function 
xlabel('x1'), ylabel('x2')
legend('Class 1', 'Class 2', 'GD Decision Boundary')

%% Train with 1000 samples
%clear all
N=1000;
load('d_train_1000.mat');
samples = data_set(1:2,:)';
true_class_labels = data_set(3,:)';

%Prepare to separate data
class0_data = zeros(N,2);
class1_data = zeros(N,2);
class0_sample_num = 0;
class1_sample_num = 0;

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(n,:)=samples(n,:);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(n,:)=samples(n,:);
        class0_sample_num=class0_sample_num+1;
    end
end

%From Suggestion from professor, do it linear first to initialize first
%theta values
z_lin = [ones(N,1) samples];
t_init = zeros(3,1);
t_lin = GD(z_lin,N,true_class_labels,t_init,1,1000);
%Then use these linear values to calculate the quadratic values
z_quad = [ones(N,1) samples(:,1) samples(:,2) (samples(:,1)).^2 samples(:,1).*samples(:,2) (samples(:,2)).^2];
t_init_quad = [t_lin;zeros(3,1)];
t = GD(z_quad,N,true_class_labels,t_init_quad,1,1000);

%Plot of the data set
figure(6),
scatter(class0_data(1:class0_sample_num,1),class0_data(1:class0_sample_num,2),'or')
hold on
scatter(class1_data(class0_sample_num+1:N,1),class1_data(class0_sample_num+1:N,2),'b+')
title('Classifier using 1000 points for Training')

% Prepare figure for boundary
horizontalGrid = linspace(floor(min(samples(:,1))-1),ceil(max(samples(:,1))+1),101);
verticalGrid = linspace(floor(min(samples(:,2))-2),ceil(max(samples(:,2))+2),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
z_test = [ones((91*101),1) h(:) v(:) h(:).^2 h(:).*v(:) v(:).^2];
sigmoid = 1./(1+exp(-z_test*t));
min_vals = min(sigmoid);max_vals = max(sigmoid);
discriminantGrid = reshape(sigmoid,91,101);

%Plot the boundary
contour(horizontalGrid,verticalGrid,discriminantGrid,1); % plot equilevel contours of the discriminant function 
xlabel('x1'), ylabel('x2')
legend('Class 1', 'Class 2', 'GD Decision Boundary')

%% Test classifier on validation set
% load in validation set
N=10000;
class0_prior = .9;
class1_prior = .1;
clear data_set;
clear samples;
clear true_class_labels;
load('d_validate_10k.mat');
samples = data_set(1:2,:);
true_class_labels = data_set(3,:);

%Prepare to separate data
class0_data = zeros(2,N);
class1_data = zeros(2,N);
class0_sample_num = 0;
class1_sample_num = 0;

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(:,n)=samples(:,n);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(:,n)=samples(:,n);
        class0_sample_num=class0_sample_num+1;
    end
end

% Decide based on which side of the line each point is on
z_valid = [ones(1,N); samples(1,:); samples(2,:); samples(1,:).^2; samples(1,:).*samples(2,:); samples(2,:).^2]';
dec = (1./(1+exp(-z_valid*t))>.5)';

true_neg = find(dec==0 & true_class_labels==0);
false_pos = find(dec==1 & true_class_labels==0);
false_neg = find(dec==0 & true_class_labels==1);
true_pos = find(dec==1 & true_class_labels==1);

p00 = length(true_neg)/class0_sample_num; % probability of true negative
p10 = length(false_pos)/class0_sample_num; % probability of false positive
p01 = length(false_neg)/class1_sample_num; % probability of false negative
p11 = length(true_pos)/class1_sample_num; % probability of true positive

err_percent = (p10*class0_prior + p01*class1_prior)*100;

% Plot decisions and decision boundary
figure(7)
hold on
plot(samples(1,true_neg),samples(2,true_neg),'og')
plot(samples(1,false_pos),samples(2,false_pos),'*r')
plot(samples(1,false_neg),samples(2,false_neg),'xr')
plot(samples(1,true_pos),samples(2,true_pos),'+g')
axis equal,

%Plot the boundary

contour(horizontalGrid,verticalGrid,discriminantGrid,1); % plot contour of decision rule
xlabel('x1'), ylabel('x2')
legend('Class 0 Correct','Class 0 Incorrect','Class 1 Incorrect','Class 1 Correct', 'Decision Boundary', 'Location', 'Southwest')

%legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier','Location', 'Northwest');
title('Test Data Classification using Gradient Descent');
fprintf('\nTotal error using Gradient Descent: %.2f%%\n',err_percent);

axis equal,
%% Functions
function t = GD(z, N, true_class_labels, t, alpha, runs)
    c = zeros(runs, 1);
    for i = 1:runs % while norm(cost_gradient) > threshold
        h = 1./(1+exp(-z*t));	% Sigmoid function   
        c(i) = (-1/N)*((sum(true_class_labels' * log(h)))+(sum((1-true_class_labels)' * log(1-h))));
        c_gradient = (1/N)*(z' * (h - true_class_labels));
        t = t - (alpha.*c_gradient); % Update theta
    end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end


