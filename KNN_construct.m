function y=KNN_construct(X, U, num_neighbors,rank_neighbor)
% X: size n * d, n is the number of samples
% num_neighbors: total number of neighbors for each sample
% rank_neighbor: the rank_neighbor-th nearest distance as sigma

[n,~]=size(X);
[m,~] = size(U);
num_s =num_neighbors;
dist = pdist2(X,U,'euclidean');

[dist,idx] = sort(dist,2); % sort each row of dist in ascending order and return the index idx
dist = dist(:,1:num_neighbors);
idx = idx(:,1:num_neighbors);

sigma=sparse(1:n,1:n, 1./dist(:,rank_neighbor),n,n);  

id_row=repmat([1:n]',1,num_s);
id_col=double(idx);
w=exp(-(sigma * dist).^2);
y=sparse(id_row,id_col,w,n,m);

    