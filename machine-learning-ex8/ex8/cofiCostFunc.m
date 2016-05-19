function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); % (num_movies x n)
Theta_grad = zeros(size(Theta)); % (num_users x n)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% X*Theta' (num_movies x n) * (n x num_users) = (num_movies x num_users)

cost_matrix = ( (X*Theta') - Y ).*R;

J = sum(sum(cost_matrix.^2))/2;

% Gradients
X_grad = (cost_matrix) * Theta; % (num_movies x num_users) * (num_users x n) = (num_movies x n); 

Theta_grad = (cost_matrix)' * X; % (num_users x num_movies) * (num_movies x n) = (num_users x n); 

%for i = 1:num_movies
%	idx = find(R(i,:) == 1);
%	theta_tmp = Theta(idx,:);
%	y_tmp = Y(i,idx);
%	X_grad(i,:) = ((X(i,:) * theta_tmp') - y_tmp) * theta_tmp;
%end
%
%for j = 1:num_users
%	idx = find(R(:,j) == 1);
%	% X is a movies x features matrix (i x k)
%	x_tmp = X(idx,:);
%	% Y is a movies x users matrix (i x j)
%	y_tmp = Y(idx,j);
%	Theta_grad(j,:) = ((Theta(j,:) * x_tmp') - y_tmp') * x_tmp;
%end

% Regularized cost
J = J + lambda/2 * ( sum(sum(Theta.^2)) + sum(sum(X.^2)) );

% Regularized gradients
X_grad = X_grad + (lambda*X);

Theta_grad = Theta_grad + (lambda*Theta);



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
