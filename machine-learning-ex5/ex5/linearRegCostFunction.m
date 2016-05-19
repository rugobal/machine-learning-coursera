function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Cost function
h_of_x = X*theta; % (m x n+1) * (n+1 x 1) = m x 1

J = 1/(2*m) * sum((h_of_x - y).^2);


reg_term = lambda/(2*m) * sum( theta(2:end).^2 );

J = J + reg_term;


% Gradient

error = h_of_x - y; % m x 1

% size(theta) = n+1 x 1
% size(y) = m x 1
%
% size(X) = m x n+1  --> x_10 x_11 x_12
%                        x_20 x_12 x_13
%                        x_30 x_22 x_33

% size(error' * X) -> (1 x m) * (m x n+1) = 1 x n + 1 

grad = 1/m * error' * X;

theta_zeroed_elem0 = zeros(1,size(theta));
theta_zeroed_elem0(2:end) = theta(2:end); 

grad = grad + (lambda/m * theta_zeroed_elem0);

% =========================================================================

grad = grad(:);

end
