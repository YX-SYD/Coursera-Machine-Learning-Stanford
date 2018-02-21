function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
par_list = [0.01,0.03,0.1,0.3,1,3,10,30];
par_number = length(par_list)
x1 = [1 2 1]; x2 = [0 4 -1];
result = zeros(par_number^2,3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i=1:par_number
  C = par_list(i);
  for j=1:par_number
    sigma = par_list(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model,Xval);
    cost = mean(double(predictions ~= yval));
    if ((i==1)&&(j==1))
      cost_min = cost;
    else
      if cost < cost_min
        cost_min = cost;
        C_select = C;
        sigma_select = sigma;
      end
    end
  end
end

C = C_select;
sigma = sigma_select;
    
   






% =========================================================================

end
