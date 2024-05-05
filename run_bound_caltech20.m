warning off
addpath([pwd, '/measure']);

%%% parameters
para.maxiter = 30;
para.tol = 1e-3;
para.lambda = 100;
para.lambda0 = 100;
para.epsilon = 1e-5;
para.epsilon_q = 1e-5;
para.mZ = 1.0;
para.lambda_Zs =0.2; % [0.01,0.02,0.05,0.07,0.1,0.2,0.5];  
para.lambda_Ms = 1e4; %[1e1,1e2,1e3,1e4,1e5]; 
para.betas = 1e-4; %[1e-5,1e-4,1e-3,1e-2,1e-1];

kmeans_repeat = 1;
num_neighbors = 2;
rank_neighbor = 2;
m = 500;

%%%% load data
view_idx = -1;
load ./dataset/Caltech101-20.mat
if size(X,1) > 1.5
    X = X';    
end
% X = X(view_idx);
k = length(unique(Y));
para.k = k;
view_num = length(X);
n = length(Y);
para.C_q = n;
totalrun = 20;
% preprocessing
for t = 1:view_num
    X{t} = normr(full(X{t}));
end

fs = floor(n*[0.05]);

loop = 0;
for f_iter = 1:length(fs)
    f = fs(f_iter);
    for outerouter_iter = 1:length(para.lambda_Zs)
        para.lambda_Z = para.lambda_Zs(outerouter_iter);
        for outer_iter = 1:length(para.lambda_Ms)
            para.lambda_M = para.lambda_Ms(outer_iter);
            for middle_iter = 1:length(para.betas)
                para.beta = para.betas(middle_iter);
                for runid = 1:totalrun
                    rng(1000 * runid)
                    tic
                    %%%% landmarks selection
                    conX = cell2mat(X);
                    nCols = cell2mat(cellfun(@(x) size(x, 2), X, 'UniformOutput', 0));
                    [~,U]=litekmeans(conX,m,'MaxIter',100,'Replicates',kmeans_repeat);
                    U = mat2cell(U, m, nCols);

                    %%%%encode supervision
                    random_sampler = randperm(length(Y));
                    label_ind = random_sampler(1:f);
                    label_ind = sort(label_ind,'ascend')';
                    fidelity = [label_ind,Y(label_ind)];
                    [W_mustlink,W_cannotlink] = encode_link(fidelity,k);
                    num_mustlink = nnz(W_mustlink);
                    num_cannotlink = nnz(W_cannotlink);
                    L_M = diag(sum(W_mustlink,2))-W_mustlink;
                    L_C = diag(sum(W_cannotlink,2))-W_cannotlink;
                    L_M = L_M./num_mustlink;
                    L_C = L_C./num_cannotlink;
                    % permute data index
                    test_ind = setdiff([1:n]',label_ind);
                    perm_ind = [label_ind;test_ind];
                    for v = 1:view_num
                        Xp{v} = X{v}(perm_ind,:)';
                        U{v} = U{v}';
                    end
                    Yp = Y(perm_ind);

                    [A,B,Z] = FSSMSC_bound(Xp, U, L_M, L_C,f,para);

                    % use A
                    A = normr(A')';
                    F = A * Z;
                    F = normr(F');

                    timer1(runid) = toc;

                    % Obtain Clustering labels
                    C = F(1:f,:);
                    CF_dist = pdist2(F,C);
                    [~,idx_cla] = min(CF_dist,[],2);
                    idx_cla = fidelity(idx_cla,2);

                    timer2(runid) = toc - timer1(runid);

                    % Obtain Clustering labels by kmeans 
                    [idx_clu, ~] = kmeans(F, k,'maxiter',1000,'replicates',10,'EmptyAction','singleton');

                    timer3(runid) = toc - timer2(runid) - timer1(runid);

                    totaltimer(runid) = toc;

                    [result_clu(runid,:)] = Clustering8Measure(Yp((f+1):n), idx_clu((f+1):n));
                    [result_cla(runid,:)] = Clustering8Measure(Yp((f+1):n), idx_cla((f+1):n));

                    fprintf('clu result: %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',result_clu(runid,1),result_clu(runid,2),result_clu(runid,3),result_clu(runid,4),totaltimer(runid));
                    fprintf('cla result: %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',result_cla(runid,1),result_cla(runid,2),result_cla(runid,3),result_cla(runid,4),totaltimer(runid));
                end
                mean_clu_res = mean(result_clu,1);
                std_clu_res = std(result_clu,1);
                mean_cla_res = mean(result_cla,1);
                std_cla_res = std(result_cla,1);


                fprintf('mean_clu_result: %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',mean_clu_res(1),mean_clu_res(2),mean_clu_res(3),mean_clu_res(4),mean(totaltimer));
                fprintf('mean_cla_result: %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',mean_cla_res(1),mean_cla_res(2),mean_cla_res(3),mean_cla_res(4),mean(totaltimer));

                output1 = [',f=' num2str(f,'%.1f'), ',totalrun=' num2str(totalrun),',view_idx=' num2str(view_idx),...
                    ',nlandmarks=' num2str(m),  ',nCluster=' num2str(k), ',kmeans_repeat=' num2str(kmeans_repeat), ',lambda=' num2str(para.lambda), ...
                         ',lambda_Z=' num2str(para.lambda_Z,'%.4f'), ',beta=' num2str(para.beta,'%2.1e'),...
                          ',lambda_M=' num2str(para.lambda_M,'%.4f'),',tol=' num2str(para.tol,'%.4f'), ',maxiter=' num2str(para.maxiter,'%.1f'),...
                          ',lambda0=' num2str(para.lambda0),',epsilon_q=' num2str(para.epsilon_q),',C_q=' num2str(para.C_q),',mZ=' num2str(para.mZ),',epsilon=' num2str(para.epsilon)
                          ];
                output2 = ['measure: ACC nmi Purity Fscore Precision Recall AR Entropy'];
                output3 = ['clu mean:',...
                          num2str(mean_clu_res(1),'%.4f'),' ', num2str(mean_clu_res(2),'%.4f'),' ',num2str(mean_clu_res(3),'%.4f'),' ',num2str(mean_clu_res(4),'%.4f'),' ', ...
                          num2str(mean_clu_res(5),'%.4f'),' ',num2str(mean_clu_res(6),'%.4f'),' ',num2str(mean_clu_res(7),'%.4f'),' ',num2str(mean_clu_res(8),'%.4f')
                         ];
                output4 = ['clu std:',...
                          num2str(std_clu_res(1),'%.4f'),' ',num2str(std_clu_res(2),'%.4f'),' ',num2str(std_clu_res(3),'%.4f'),' ',num2str(std_clu_res(4),'%.4f'),' ', ...
                          num2str(std_clu_res(5),'%.4f'),' ',num2str(std_clu_res(6),'%.4f'),' ',num2str(std_clu_res(7),'%.4f'),' ',num2str(std_clu_res(8),'%.4f')
                         ];
                output5 = ['cla mean:',...
                          num2str(mean_cla_res(1),'%.4f'),' ', num2str(mean_cla_res(2),'%.4f'),' ',num2str(mean_cla_res(3),'%.4f'),' ',num2str(mean_cla_res(4),'%.4f'),' ', ...
                          num2str(mean_cla_res(5),'%.4f'),' ',num2str(mean_cla_res(6),'%.4f'),' ',num2str(mean_cla_res(7),'%.4f'),' ',num2str(mean_cla_res(8),'%.4f')
                         ];
                output6 = ['cla std:',...
                          num2str(std_cla_res(1),'%.4f'),' ',num2str(std_cla_res(2),'%.4f'),' ',num2str(std_cla_res(3),'%.4f'),' ',num2str(std_cla_res(4),'%.4f'),' ', ...
                          num2str(std_cla_res(5),'%.4f'),' ',num2str(std_cla_res(6),'%.4f'),' ',num2str(std_cla_res(7),'%.4f'),' ',num2str(std_cla_res(8),'%.4f')
                         ];
                output7 = ['totaltime of clu:',',avgrate=' num2str(mean(timer1+timer3),'%.4f'),',std=' num2str(std(timer1+timer3),'%.4f'),...
                            'totaltime of cla:',',avgrate=' num2str(mean(timer1+timer2),'%.4f'),',std=' num2str(std(timer1+timer2),'%.4f') 
                           ];  
                fid = fopen('./results/Caltech20_bound_results.txt','a');
                fprintf(fid, '%s\n', output1);
                fprintf(fid, '%s\n', output2);
                fprintf(fid, '%s\n', output3);
                fprintf(fid, '%s\n', output4);
                fprintf(fid, '%s\n', output5);
                fprintf(fid, '%s\n', output6);
                fprintf(fid, '%s\n', output7);
                fclose(fid);
            end
        end
    end
end
