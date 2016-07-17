function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i=1:K
    cc = idx==i;
    anssx = 0;
    anssy = 0;
    opop = 0;
    dist = zeros(n, 1);
    for j = 1:m
        if( cc(j)==1 )
            for kk = 1:n
              dist(kk) = dist(kk) + X(j, kk);
            end;  
            opop = opop + 1;
        end;
    end;
    dist = dist/opop;
    centroids(i, :) = dist';
end;    

% =============================================================


end

