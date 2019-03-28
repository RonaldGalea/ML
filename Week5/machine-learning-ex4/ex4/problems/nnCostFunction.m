function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% For these computations, it is easier to view X as having a training example on each column

X = X';

% Add a row of ones to the X data matrix
X = [(ones(size(X,2), 1))'; X]; % a column of ones, transposed


X1 = sigmoid(Theta1 * X); % hypothesis * example, this is why i transposed X
                 % if in X element (i;j) meant jth example of feature i, in X1 it means jth example of the new feature i
                 % so X1 has the exact same structure as X, only that is holds the data for the new features
                 % those in the current layer
                 % so we can continue the computations in the same manner, the general formula being Xi = Thetai * Xi-1
				 % the way the new features are learned is by selecting proper weights for the initial ones!
  
% Add a row of ones to the X1 data matrix
X1 = [(ones(size(X1,2), 1))'; X1];  
                
% X2 is our predictions matrix, each column represents the prediction for the initial data column j, a num_labels size column with each entry a number from 0 to 1
X2 = sigmoid(Theta2 * X1); 

% we would now like to have instead of simple 1-num_labels labels, num_labels dimensional vectors with 1 on
% the correct possition and 0's in rest

labels = zeros(num_labels, m); % each column will be such a vector
one_numlables = zeros(num_labels,1);

for i = 1:num_labels
  one_numlables(i) = i;
endfor

for i = 1:m
  groundTruth = (one_numlables == y(i)); % groundTruth is now a column with 1 on the y(i)th position
  labels(:,i) = groundTruth;
endfor

% then a vectorized cost function can be written as:
% remember: each column in labels is the actual value, each column in X2 is the prediction, both are size m x num_labels
% we need 2xsum to take the sum of all elemets of a matrix
% element wise multiplication compares each predicted value with the actual value

J = (-1/m) * sum ( sum( (labels .* log(X2) + (1-labels) .* log(1-X2) ) ) );

% we should not regularize bias terms, so we remove the bias columns
Theta1Reg = Theta1(:,2:end);
Theta2Reg = Theta2(:,2:end);

J = J + lambda/(2*m) * ( sum( sum( Theta1Reg.^2 ) )  + sum ( sum (Theta2Reg.^2) ) );

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta3 = zeros(num_labels,1); % errors column vector, one for each label
delta2 = zeros(size(Theta1,1) + 1,1); % errors for layer 2, size(Theta1,1) = # nr of activation nodes, and I added 1 for the bias 

for t = 1:m
  
  A1 = X(:,t); % jth data example: 401x1
  
  Z2 = Theta1 * A1;
  
  A2 = sigmoid(Z2); % second layer's feature column without the bias term: 25x401 * 401x1 = 25x1
  
  A2 = [1; A2]; % added bias term: 26x1
  
  Z3 = Theta2*A2;
  
  A3 = sigmoid(Z3); % 10x26 * 26x1 = 10x1 - the prediction for jth example
  
  delta3 = A3 - labels(:,t); % we use the previously comuted labels matrix, which tells what class each example is in, in each of its columns

  delta2 = (Theta2' * delta3)(2:end).*(sigmoidGradient(Z2)); % removed bias 
  
  % this examples contribution to the gradient
  Theta1_grad = Theta1_grad + delta2 * A1';
  Theta2_grad = Theta2_grad + delta3 * A2';  
  
endfor

 % take the average of all contributions
 Theta1_grad = Theta1_grad/m;
 Theta2_grad = Theta2_grad/m;
 
 % add regularization partials to gradients, without the first column which are the bias weights
 Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
 Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
