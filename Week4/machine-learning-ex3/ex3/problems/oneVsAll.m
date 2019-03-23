function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


initial_theta = zeros(n + 1, 1); % a vector of zeros, the initial guess for theta that will be opimised by fmincg


options = optimset('GradObj', 'on', 'MaxIter', 50); % options for fmincg, saying we provide the gradient and setting #iterations 

for c = 1:num_labels  % apply logistic regression for each class
  
  groundTruth = (y==c);  % sets the current class, y is simply a vector from 1 ... 10, after y == c we will have a 1
                         % at the cth position, this will be needed for the current class' cost function
  
  [theta] = fmincg (@(t)(lrCostFunction(t, X, groundTruth, lambda)), initial_theta, options);
  % having obtained the optimal theta for this class, we will add it to the results matrix
  % theta is an n+1 dimensional column, since this needs to be a row in all_theta, we will take its transpose
  
  all_theta(c,:) = theta;  % the cth row in all_theta becomes the newly found optimal theta
  
endfor










% =========================================================================


end