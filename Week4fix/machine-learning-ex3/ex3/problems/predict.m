function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% For these computations, it is easier to view X as having a training example on each column

X = X';

% Add a row of ones to the X data matrix
X = [(ones(size(X,2), 1))'; X];


X1 = sigmoid(Theta1 * X); % hypothesis * example, this is why i transposed X
                 % if in X element (i;j) meant jth example of feature i, in X1 it means jth example of the new feature i
                 % so X1 has the exact same structure as X, only that is holds the data for the new features
                 % those in the current layer
                 % so we can continue the computations in the same manner, the general formula being Xi = Thetai * Xi-1
				 % the way the new features are learned is by selecting proper weights for the initial ones!
  
% Add a row of ones to the X1 data matrix
X1 = [(ones(size(X1,2), 1))'; X1];  
                
X2 = sigmoid(Theta2 * X1); 

% now each column contains a prediction, so we only need to look at the max element on each column

max_elem = p;
[max_elem, p] = max(X2,[],1); % same as predictOneVsAll
p = mod(p,10);

% =========================================================================


end
