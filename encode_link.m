function [W_mustlink,W_cannotlink] = encode_link(fidelity,k)
n = size(fidelity,1);
W_mustlink = zeros(n,n);
W_cannotlink = zeros(n,n);
for j = 1:k
    index1 = find(fidelity(:,2)==j); 
    for t = 1: k
        index2 = find(fidelity(:,2)==t);   
        if j==t
            W_mustlink(index1,index2) = ones(length(index1),length(index2));
        else
            W_cannotlink(index1,index2) = ones(length(index1),length(index2));
        end
    end
end
W_mustlink = (W_mustlink+W_mustlink')*0.5;
W_cannotlink = (W_cannotlink+W_cannotlink')*0.5;
end