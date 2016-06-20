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
opop = size(theta);
aa = y' * log( sigmoid( ( theta' * X' )' ) );
bb = (1 - y') * log( 1 - sigmoid( ( theta' * X' )' ) );
anss = 0;
for i = 2:opop
    anss = anss + ( theta( i ) * theta( i ) );

J = ( aa + bb ) / m;
J = -J;
J = J + ( lambda * anss ) / ( 2 * m );   

grad = (X' * (sigmoid(X * theta) - y)) * (1/m) ;
for lo = 2:opop
       grad(lo) = grad(lo) + (  ( lambda * theta(lo) ) / m );
end;

% =============================================================

end