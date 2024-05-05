function [X] = unit_row(Z)
    sumZ = sum(Z,2) + eps ;
    temp = repmat(sumZ,1,size(Z,2));
    X = Z./temp;
end