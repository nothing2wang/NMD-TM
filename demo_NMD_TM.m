clear all
close all
clc

rep=2;  %number of runs for each algorithm---> it must be r>1

%Choose the approximation rank
r=40;  %32

%Choose between synthetic data or MNIST dataset
number=2;
switch number
    case 1 %noiseless synthetic data
        m=4000; n=2000; %dimension of the problem
        W1=randn(m,r); H1=randn(r,n);  X=max(0,W1*H1); 
    case 2 %MNIST dataset
        Y=load('mnist_all.mat');
        %w1=1:5:3000; %Number of images for each digit
        w1=1:5000;
        X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
           Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
        X=double(X);
        [m,n]=size(X);
end

%Parameters setting
param.maxit=300000; param.tol=1.e-4; param.tolerr = 0; param.time=20;

%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param.time,c);

for k=1:rep

    %Random initialization
    %alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    %param.W0=alpha*randn(n,r); param.H0=(randn(r,m));
    %param.Theta=param.W0*param.H0
    %Nuclear norm initialization
    Theta1=randn(m,n);
    [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
    [ua,sa,va] = svds(Theta2,r); 
    svalues = diag(sa);
    param.W0 = ua; 
    param.H0 = sa*va';
    param.Theta0=param.W0*param.H0;
    
    % NMD-TM
    param.beta1=0.95; param.beta2=-0.05; param.lambda=0.0001;
    [T_TM,err_TM,it_TM,t_TM]=NMD_TM(X,r,param);
    err_TM_k(k)=err_TM(end);
    p_TM(:,k)=spline(t_TM,err_TM,time);

end

%Find best solution
opt=min(err_TM_k);

%Compute the average error
mean_TM=sum(p_TM')/rep;

%Subtract best solution
p_TM_mean=mean_TM-opt; 

%Plot the results
figure
set(gca,'Fontsize',18)
semilogy(time,p_TM_mean,'r-','LineWidth',1.9);
grid on
xlabel('Time','FontSize',22,'FontName','times'); ylabel('err(t)','FontSize',20,'FontName','times');
legend({'NMD-TM'},'FontSize',22,'FontName','times')
