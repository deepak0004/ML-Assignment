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


htheta = X*theta;
htheta = htheta - y;
htheta = htheta.*htheta;
htheta = sum( htheta );
htheta = htheta/(2*m);

thetaa = theta(2:end);
thetaa = thetaa.*thetaa;
thetaa =  sum( thetaa );
thetaa = thetaa * lambda;
thetaa = thetaa/(2*m);
J = htheta + thetaa;

firtheta = X*theta - y;  
for j = 1:size(X, 2)
    firtheta = X*theta - y;
    firtheta = firtheta .* X(:, j);
    su = sum( firtheta ) / m;
    if( j ~= 1 )
      su = su + ( theta(j) * lambda ) / m;
    end;  
    grad(j) = su;
end;    

% =========================================================================

grad = grad(:);

end