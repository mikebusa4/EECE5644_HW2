%Author: Michael Busa
%Homework #2: Q3
%Date: 2/20/2020
%Purpose: Generates N samples from a GMM with 4 Gaussian components, then
%         uses EM algorithm to estimate the parameters of a GMM that has
%         the same number of components as the true GMM

%% Generate True GMM Samples
clear all
clf
delta = 1e-2; %tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates

%Gaussian component parameters
alpha_true = [.3, .27, .23, .2]; %Probabilites that a sample comes from each component
mu_true = [0 5 8 10; 0 3 4 6];
sigma_true(:,:,1) = [5 -1;-1 2];
sigma_true(:,:,2) = [1 1;1 10]*.5;
sigma_true(:,:,3) = [3 -2;-2 8]*.1;
sigma_true(:,:,4) = [4 -.1;-.1 7]*.25;

N_samples = [10 100 1000];
choice =3;%input('1(10) 2(100) or 3(1000): ');
ideal_size = 120;%input('Ideal size: ');
%Generate data in the GMM
for j = choice:choice
    data_set = zeros(3,N_samples(j));
    data_set = generate_GMM_samples(alpha_true, mu_true, sigma_true, N_samples(j), data_set);

    data_set = data_set';
    data_set = sortrows(data_set,3);
    data_set = data_set';
    %Sort the data and labels
    samples = data_set(1:2,1:N_samples(j));
    labels = data_set(3,1:N_samples(j));
    sample_count = [0 0 0 0];
    for i=1:4
        sample_count(i) = length(find(labels==i));
    end
    
    
    figure(choice)
    plot(samples(1,1:sample_count(1)),samples(2,1:sample_count(1)),'b.')
    hold on
    plot(samples(1,(sample_count(1)+1):sample_count(1)+sample_count(2)),samples(2,sample_count(1)+1:sample_count(1)+sample_count(2)),'rx')
    plot(samples(1,sample_count(1)+sample_count(2)+1:sample_count(1)+sample_count(2)+sample_count(3)),samples(2,sample_count(1)+sample_count(2)+1:sample_count(1)+sample_count(2)+sample_count(3)),'g*')
    plot(samples(1,sample_count(1)+sample_count(2)+sample_count(3)+1:sample_count(1)+sample_count(2)+sample_count(3)+sample_count(4)),samples(2,sample_count(1)+sample_count(2)+sample_count(3)+1:sample_count(1)+sample_count(2)+sample_count(3)+sample_count(4)),'ko')
    legend('Gaussian 1', 'Gaussian 2', 'Gaussian 3', 'Gaussian 4','Location', 'Northwest')
    title('True 4-Component GMM')
    xlabel('x1'), ylabel('y2')
end

%% EM
for Runs = 1:5
    %120 ideal size for 1000 samples
    % 58 ideal size for 100 samples
    % 40 ideal size for 10 samples
    N = ideal_size; %Number of samples in the train/validate sets
    N_real = N_samples(choice); %Number of samples in the real GMM data set
    for X = 1:100
        Runs
        X
        for B = 1:10
            clearvars -except choice Runs kill ideal_size gmm_choice B X bestGMM N_samples N N_real samples labels alpha_true mu_true sigma_true delta regWeight sample_count data_set
            alpha_total = zeros(6,6);
            mu_total = zeros(2,6,6);
            Sigma_total = zeros(2,2,6,6);
            for M = 1:6
                M;
                dtrain = zeros(2,N);
                dvalidate = zeros(2,N);
                for i=1:N
                    dtrain(:,i) = samples(:,randi(N_real,1,1));
                    dvalidate(:,i) = samples(:,randi(N_real,1,1));
                end

                % Initialize the GMM to randomly selected samples
                alpha = ones(1,M)/M;
                shuffledIndices = randperm(N);
                mu = dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
                [~,assignedCentroidLabels] = min(pdist2(mu',dtrain'),[],1); % assign each sample to the nearest mean
                for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
                    Sigma(:,:,m) = cov(dtrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(2,2);
                    if(isnan(Sigma(:,:,m)))
                        Sigma(:,:,m) = eye(2,2);
                    end
                end
                t = 0; %displayProgress(t,x,alpha,mu,Sigma);

                Converged = 0; % Not converged at the beginning
                %Dalpha_past=zeros(1,1);
                %Dmu_past=0;
                %DSigma_past=zeros(1,1,M);
                while ~Converged
                    for l = 1:M
                        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(dtrain,mu(:,l),Sigma(:,:,l));
                    end
                    plgivenx = temp./sum(temp,1);
                    alphaNew = mean(plgivenx,2);
                    w = plgivenx./repmat(sum(plgivenx,2),1,N);
                    muNew = dtrain*w';
                    for l = 1:M
                        v = dtrain-repmat(muNew(:,l),1,N);
                        u = repmat(w(l,:),2,1).*v;
                        SigmaNew(:,:,l) = u*v' + regWeight*eye(2,2); % adding a small regularization term
                    end
                    Dalpha = sum(abs(alphaNew-alpha));
                    Dmu = sum(sum(abs(muNew-mu)));
                    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                    alpha = alphaNew; 
                    mu = muNew; 
                    Sigma = SigmaNew;
                    t = t+1;
                    %displayProgress(t,dtrain,alpha,mu,Sigma,M);
                end
                %pause(1)
                alpha_total(M,1:M) = alpha';
                mu_total(:,M,1:M) = mu;
                Sigma_total(:,:,M,1:M) = Sigma;
               
            end

            logLikelihood = zeros(1,6);
            for M = 1:6
               logLikelihood(M) = sum(log(evalGMM(dvalidate,alpha_total(M,1:M),mu_total(:,M,1:M),Sigma_total(:,:,M,1:M)))); 
            end


            [~,bestGMM(B)] = min(abs(logLikelihood));
            gmm_choice(X) = mode(bestGMM);
        end
    end
    kill;
    figure(Runs)
    C = categorical(gmm_choice,[1 2 3 4 5 6],{'1','2','3', '4', '5', '6'});
    histogram(C,'BarWidth', .5)
    yticks([0:5:100])
    xlabel('Num of Gaussian Components')
    ylabel('Bootstrap Sets Chosen')
    
end
%% Functions
function data_set = generate_GMM_samples(alpha, mu, sigma, N, data_set)
    for n=1:N
        k=rand();
        if k<alpha(1)
            data_set(1:2,n) = randGaussian(1,mu(:,1),sigma(:,:,1));
            data_set(3,n) = 1;
        elseif k<alpha(1)+alpha(2)
            data_set(1:2,n) = randGaussian(1,mu(:,2),sigma(:,:,2));
            data_set(3,n) = 2;
        elseif k<alpha(1)+alpha(2)+alpha(3)
            data_set(1:2,n) = randGaussian(1,mu(:,3),sigma(:,:,3));
            data_set(3,n) = 3;
        else
            data_set(1:2,n) = randGaussian(1,mu(:,4),sigma(:,:,4));
            data_set(3,n) = 4;
        end
    end
end

function displayProgress(t,x,alpha,mu,Sigma,M)
figure(M),
if size(x,1)==2
    cla
    plot(x(1,:),x(2,:),'k.'); 
    txt = strcat('Data and Estimated GMM Contours:  ', int2str(M));
    xlabel('x_1'), ylabel('x_2'), title(txt),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
end
%logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
%xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow;
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

