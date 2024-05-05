function [A,B,Z] = FSSMSC_bound(Xp, U, L_M, L_C,nl,para)

nv = length(Xp);
n = size(Xp{1},2);
m = size(U{1},2);
% preprocessing
lambda_Z = para.lambda_Z;
lambda_M = para.lambda_M;
lambda = para.lambda;
lambda0 = para.lambda0;
eta = 1/lambda;
beta = para.beta;
epsilon = para.epsilon;
mZ = para.mZ;
epsilon_q = para.epsilon_q;
C_q = para.C_q;

k = para.k;
maxiter = para.maxiter;
tol = para.tol;
I = eye(m);

%%%%%%Initialization
LHS = lambda .* I;
RHS = zeros(m,n);
for v = 1:nv
    LHS = LHS + U{v}'*U{v};
    RHS = RHS + U{v}'*Xp{v};
end
B = LHS\RHS;
Z = max(0,B-lambda_Z/lambda);
Z = min(mZ,Z); 
Z = unit_row(Z')'; 

q = (Z * Z') * ones(m,1);
q = max(epsilon_q,q);
q = min(C_q,q);

A = zeros(k,m);
Lambda = zeros(m,n);

iter = 0;
while iter < maxiter
    iter = iter + 1;        
    %update A
    LU = epsilon/k .* I + Z(:,1:nl)*L_C*Z(:,1:nl)';
    W = Z * Z';   
    D_inv = diag(1./sqrt(q));  
    
    LL = lambda_M * Z(:,1:nl)*L_M*Z(:,1:nl)' + I - D_inv * W * D_inv;
    LL = 0.5 * (LL + LL');
    LU = 0.5 * (LU + LU');
    [A,~] = v1_trace_ratio_optim(LU, LL,k,20);   
    Ak = A';
    A = normr(A);   
    A = A';
    
    
    %udpate Z
    Zk = Z;   
    tau1 = trace((A*LL*A'));
    tau2 = trace((A*Z(:,1:nl)*L_C*Z(:,1:nl)'*A'))+epsilon; 
    D_inv = diag(1./sqrt(q));  
    
    T =  - 2 * beta./tau2 * D_inv *(A'*A) * D_inv;
    L_pc = beta/(tau2.^2)*(2*tau2*lambda_M*L_M - 2*tau1 * L_C);
    AtA = A'*A;
    temp = AtA * Z(:,1:nl) * L_pc;
    temp = [temp,zeros(m,n-nl)];
    nabla_Z = temp + T * Z;
    nabla_Z = nabla_Z - Lambda + lambda .* (Z-B) + lambda_Z * ones(m,n);
    ZZq = W * ones(m,1)-q;
    temp = ZZq * (ones(1,m) * Z) + ones(m,1) * (ZZq'*Z);
    nabla_Z = nabla_Z + lambda0 * temp;
    
    Z = Z - eta .* nabla_Z;
    Z = max(0,Z);
    Z = min(mZ,Z); 
    Z = unit_row(Z')';   
    
    % update q
    qk = q;
    LU = Z(:,1:nl)*L_C*Z(:,1:nl)';
    tau2 = trace((A*LU*A')) + epsilon;  
    W = Z * Z';
    AtA = A'*A;
    
    q_inv = 1./q;
    
    temp1 = W * diag(sqrt(q_inv))*AtA * diag(q_inv.*sqrt(q_inv));
    nabla_q = beta./tau2 .* (diag(temp1)) ...
             + lambda0*(q-W*ones(m,1));
    q = q - eta * nabla_q;
    q = max(epsilon_q,q);
    q = min(C_q,q);
    
    % update B
    Bk = B; 
    LHS = lambda .* I;
    RHS = lambda .* Z - Lambda;
    for v = 1:nv
        LHS = LHS + U{v}'*U{v};
        RHS = RHS + U{v}'*Xp{v};
    end
    B = LHS\RHS;
    
    % update Lambda
    Lambda = Lambda + lambda.*(B-Z);
    
    % update lambda and eta
    eta = 1/lambda;
    
    % display
    diffZ = max(max(abs(Z-Zk)));
    diffB = max(max(abs(B-Bk))); 
    stopC = max([diffZ,diffB]);
    if (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['iter ' num2str(iter)  ',nnzZ=' num2str(nnz(Z))  ,...
            ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    % early stop
    if stopC < tol 
        fprintf('convergence after iteration %d\n',iter);
        break;   
    end
    
end


