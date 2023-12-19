function [Theta,err,i,time]=NMD_TM(X,r,param)

% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,W,H} ||Z-WH||_F^2  s.t. rank(WH)<=r, max(0,Z)=X 
% 
% using a three blocks alternating minimization; 
%
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   W0      : m-by-r matrix
%   H0      : r-by-n matrix
%   param   : structure, containing the parameter of the model
%       .W0 = initialization of the variable W (default: randn)
%       .H0 = initialization of the variable H (default: randn)
%       .maxit  = maximum number of iterations (default: 1000) 
%       .tol    = tolerance on the relative error (default: 1.e-4)
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-5)
%       .time   = time limit (default: 20)
%       .beta1  = fixed momentum parameter (default: 0.7)
%       .display= if set to 1, it diplayes error along iterations (default: 1)
% 
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of    
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)=r.
%   err     : vector containing evolution of relative error along
%             iterations ||X-max(0,Theta)||_F / || X ||_F
%   i       : number of iterations  
%   time    : vector containing time counter along iterations
%%
% this code taken and partially adapted from  https://gitlab.com/ngillis/ReLU-NMD

[m,n]=size(X); 
 if nargin < 3
    param = [];
 end
 if ~isfield(param,'W0') || ~isfield(param,'H0')
     param.W0=randn(m,r); param.H0=randn(r,n);  
 end
if ~isfield(param,'maxit')
    param.maxit = 1000; 
end
if ~isfield(param,'lambda')
    param.lambda = 0.0001; 
end
if ~isfield(param,'tol')
    param.tol = 1.e-4; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-5;
end
if ~isfield(param,'time')
    param.time = 20; 
end
if ~isfield(param,'beta1')
    param.beta1 = 0.95; 
end
if ~isfield(param,'beta2')
    param.beta2 = -0.05; 
end
if ~isfield(param,'display')
    param.display = 1;
end

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n'; 
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2])); 
    X(X<0) = 0; 
end

%Detecting positive components of X
normX=norm(X,'fro');
idx=(X==0);
idxp=(X>0);

%Initialize the latent variable
Z0 = zeros(m,n);   
Z0(idxp) = nonzeros(X);

%Create istances for the variables
Z=Z0; W_old=param.W0; H_old=param.H0; Theta=W_old*H_old; Z_old=Z;  Theta_old=Theta;

%Initialize error and time counter
err(1)=norm(max(0,Theta)-X,'fro')/normX;
time(1)=0;

if param.display == 1
    disp('Running 3B-NMD-TM, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.2; lambda=param.lambda;
for i=1:param.maxit
    tic
    %update of Z ----> Z=min(0,Theta)
    Z=min(0,Theta.*idx);
    Z=Z+X.*idxp;

    %Momentum on Z
    Z=(1+param.beta1)*Z-param.beta1*Z_old;

    %update of W
    
    W=(lambda*eye(r)+H_old*H_old')\(H_old*Z'); 
    W=W';
    W= (1+param.beta2)*W-param.beta2*W_old;
    
    
    %update of H
    H=(lambda*eye(r)+W'*W)\(W'*Z);
    H= (1+param.beta2)*H-param.beta2*H_old;
    
    %Compute approximation matrix
    Theta=W*H; 
    Theta_return=Theta; %keep trace of the value for the time limit stopping criteria

    %Error computation
    err(i+1)=norm(max(0,Theta)-X,'fro')/normX;
    if err(i+1)<param.tol 
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
         if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: ||X-max(0,WH)||/||X|| < %2.0d\n',param.tol);
        end
        break
    end
    
    if i >= 11  &&  abs(err(i+1) - err(i-10)) < param.tolerr
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < %2.0d\n',param.tolerr);
        end
        break
        break
    end
    %Momentum step on Theta, not in the last iteration otherwise Theta has not rank r
    if i<param.maxit-1 
        Theta=(1+param.beta1)*Theta-param.beta1*Theta_old;
    end

    %Update of old variables
    Z_old=Z; Theta_old=Theta; W_old=W; H_old=H; 

    %Time check
    time(i+1)=time(i)+toc;
    if time(i+1)>param.time
        Theta=Theta_return;  %Needed for Theta in order to be of rank r
        break
    end
    
    if param.display == 1 && time(i+1) >= cntdis 
        disp_time = min(60,disp_time*1.5);
        fprintf('[%2.2d : %2.2f] - ',i,100*err(i+1)); 
        cntdis = time(i+1)+disp_time; % display every disp_time
        numdis = numdis+1; 
        if mod(numdis,5) == 0
            fprintf('\n');
        end
    end
    
end
if param.display == 1
    fprintf('Final relative error: %2.2f%%, after %2.2d iterations. \n',100*err(i+1),i); 
end