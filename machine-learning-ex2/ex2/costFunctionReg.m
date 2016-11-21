function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%% Cost

J_logistic = 1/m * sum( (-y .* log( sigmoid(X*theta) ) ) - ( (1 - y) .* log( 1 - sigmoid(X*theta) ) ) );

% sum from j=1 to n using matrix multiplication
ones_1_n = [0 ones(1,length(theta)-1)];
J_regularize = lambda/(2*m) * (ones_1_n * (theta .^ 2));

J = J_logistic + J_regularize;

%% Gradient
grad_logistic = 1/m * transpose(sum( (sigmoid(X*theta) - y) .* X ));
grad_regularize = (lambda/m) * (transpose(ones_1_n) .* theta);
grad = grad_logistic + grad_regularize;

% =============================================================

end
