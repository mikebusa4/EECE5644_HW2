%Author: Michael Busa
%Homework #2: Q2
%Date: 2/24/2020
%Purpose: Generate samples according to y=ax^3+bx^2+cx+d+v, find MAP
%         estimate for w by varying gamma, analyze results using L2
%         distance


clear
clc
clf

iterations=100; %Number of gamma values to be tested
Runs = 100; %Number of times the experiment is run
lspace_min = -3; 
lspace_max = 3;
gamma_data = zeros(Runs,iterations);
for j=1:Runs
    N=10; %Number of data points generated
    w = [1 (1/2) -(11/16) -(3/16)]; %True coefficients with roots in [-1,1]
    x = zeros(1,N);
    y = zeros(1,N);
    %Parameters of the gaussian for V
    v_sigma = .1;
    v_mu=0;
    v = randGaussian(N,v_mu,v_sigma);
    x_real = [-1:.01:1];
    y_real = w(1).*x_real.^3+w(2).*x_real.^2+w(3).*x_real+w(4);
    for i=1:10
        x(i)=(rand()-.5)*2;
        y(i) = w(1)*x(i)^3+w(2)*x(i)^2+w(3)*x(i)+w(4)+v(i);
    end
   
    index =1;
    w_map=zeros(4,iterations);
    x_vals = linspace(-1,1,200);
    x_plots = [x_vals.^3; x_vals.^2; x_vals; ones(1,200)];
    y_vals = zeros(1,200);
    for gamma = logspace(lspace_min,lspace_max,iterations)
        %Equation for w_map detailed in the PDF
        w_map(:,index) = -((w*w'+(v_sigma/gamma)^2)*eye(size(w,2)))^-1*sum(repmat(y,size(w,2),1).*w',2);
        index=index+1;
        %L2 distances: square_root(sum(estimation - real)^2))
        distances = sqrt(sum((w_map-w').^2));
    end
    gamma_data(j,:) = distances;
    best_w = find_best_w(w_map,w',iterations);
    y_best = best_w(1).*x_real.^3+best_w(2).*x_real.^2+best_w(3).*x_real+best_w(4);

   
    figure(2)
    plot(x,y,'b.', 'MarkerSize', 15)
    hold on
    plot(x_real,y_real,'r-')
%    plot(x_real,y_best,'g-')
    axis([-1, 1, min(y)-1, max(y)+1])
    xlabel('x'),ylabel('y')
    title('Real function y and random x data with noise')
    legend('Data with Noise', 'True Function')
    hold off
end
%%
data = zeros(5,Runs);
for col = 1:iterations
    test_col = gamma_data(:,col);
    %Gather data for each gamma value, find min,25,50,75,& max
    data(:,col) = quantile(test_col,[0 .25 .5 .75 1]);
end

x_plotting = logspace(lspace_min,lspace_max,Runs);
y_plotting = data;

figure(4)
hold on
plot(x_plotting, y_plotting(1,:),'b.-')
plot(x_plotting, y_plotting(2,:),'r.-')
plot(x_plotting, y_plotting(3,:),'g.-')
plot(x_plotting, y_plotting(4,:),'c.-')
plot(x_plotting, y_plotting(5,:),'k.-')
hold off
title('0, 25 ,50 ,75, and 100% Gamma Values per experiment')
axis([0 .2 (min(y_plotting,[],'all')-.5) (max(y_plotting,[],'all')+.5)])
legend('Minimum', '25%', 'Median', '75%', 'Max')
xlabel('Gamma')
ylabel('L2 Distace')
%%
function [best_w] = find_best_w(w_map,w_real,n)
    min_dist = 10e5;
    for i=1:n
        temp_dist = abs(w_map(:,i)-w_real);
        total_dist = sum(temp_dist,2);
        if total_dist<min_dist
            min_dist = total_dist;
            best_w=w_map(:,i);
        end
    end
end

