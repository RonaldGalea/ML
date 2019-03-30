function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
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

% i will implement it in a vectorized way, using the following trick:
% in order to take the lines i1, i2, ... ik from a matrix, multiply by a vector with ones on those
% positions, and 0s in rest
% to compute the mean of centroid j, we need the mean of lines i1, i2, ... ik from X,
% where i1, i2, ... ik are the examples closest to j

for i = 1:K
  
  v = (idx == i); % vector with 1s on the positions i in idx where idx(i) = k
  nr = nnz(v); % number of nonzero elements = number of examples of j
  centroids(i, :) = (v' * X) / nr;
  
endfor






% =============================================================


end

