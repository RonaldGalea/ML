function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    
    
    % we need theta = theta - alpha*(dJ/dtheta0, dJ/dtheta1,..., dj/dthetan)
    % the main problem is obtaining the gradient vector
    
    gradVector = zeros(size(theta),1);

for iter = 1:num_iters
    
    for i=1:1:size(gradVector)
      gradVector(i)=(((X*theta-y)') * X(:,i))/m; 
    endfor
    
    % vectorized version: gradientVector = X'*(X*theta - y)/m
    
    % each partial derivative of the cost function, simply multiply with the respective features data column
    % X multiplied by theta gives a vector with predictions for each training example
    % each of X's columns is the training data with respect to a feature
    
    theta = theta - (alpha)*gradVector;
    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
