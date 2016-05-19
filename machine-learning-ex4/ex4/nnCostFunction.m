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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Part 1. Cost function
% --------------------------

% transform the labels vector in a m x num_labels matrix of 1's and 0
y_matrix = [];
for i=1:m,
	y_matrix = [ y_matrix; [1:1:num_labels] == y(i) ];
end;

% Add ones to the X data matrix
X = [ones(m, 1) X];

hidden_layer_features = sigmoid(X*Theta1'); % (m x n+1) * (n+1 x hidden_layer_size) = m x hidden_layer_size

hidden_layer_features = [ones(m, 1) hidden_layer_features]; % m x hidden_layer_size+1

% output layer hypothesis
h_of_x_over_k = sigmoid(hidden_layer_features * Theta2'); % (m x hidden_layer_size+1) * (hidden_layer_size+1 x num_labels) = m x num_labels


inner_term = -y_matrix.*log(h_of_x_over_k) - (1-y_matrix).*log(1-h_of_x_over_k);

% sum(A,2) sums elements in every row and produces a column vector with the results
J = (1/m)*sum(sum(inner_term,2));



% Regularized cost term

Theta1_nofirst_col_sq = Theta1(:, 2:size(Theta1,2)).^2; % remove col1 from Theta1 and Theta2 and square the elements
Theta2_nofirst_col_sq = Theta2(:, 2:size(Theta2,2)).^2;

% Add to J the sum of the elements of the two matrices multiplied by lambda/2*m
J = J + lambda/(2*m) * (sum(Theta1_nofirst_col_sq(:)) + sum(Theta2_nofirst_col_sq(:)));



% Part 2
% ---------------

% Theta1 size = hidden_layer_size x input_layer_size+1
% Theta2 size = num_labels x hidden_layer_size+1

for i=1:m,

	y_vector = y_matrix(i,:); % (1 x num_labels)

	a_1 = X(i,:); % (1 x input_layer_size+1)
	z_2 = a_1*Theta1'; % (1 x hidden_layer_size)
	a_2 = [1 sigmoid(z_2)]; % (1 x hidden_layer_size+1)
	z_3 = a_2 * Theta2'; % (1 x num_labels)
	a_3 = sigmoid(z_3); % (1 x num_labels)
		
	delta3 = a_3 - y_vector; % (1 x num_labels)
	delta2 = (delta3 * Theta2).*sigmoidGradient([1 z_2]); % (1 x num_labels) * (num_labels x hidden_layer_size+1) = 1 x hidden_layer_size+1
	delta2 = delta2(2:end); % remove the first element -> 1 x hidden_layer_size
	
	
	Theta1_grad = Theta1_grad + delta2' * a_1; % (hidden_layer_size x 1) * (1 x input_layer_size+1) = hidden_layer_size x input_layer_size+1
	Theta2_grad = Theta2_grad + delta3' * a_2; %  (num_labels x 1) * (1 x hidden_layer_size+1) = num_labels x hidden_layer_size+1
	
end;


Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularization

Theta1_zeroed_first_col = Theta1;
Theta1_zeroed_first_col(:,1) = zeros(hidden_layer_size,1); % hidden_layer_size x input_layer_size+1
Theta2_zeroed_first_col = Theta2;
Theta2_zeroed_first_col(:,1) = zeros(num_labels,1); % num_labels x hidden_layer_size+1

Theta1_grad = Theta1_grad + (lambda/m * Theta1_zeroed_first_col);
Theta2_grad = Theta2_grad + (lambda/m * Theta2_zeroed_first_col);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
