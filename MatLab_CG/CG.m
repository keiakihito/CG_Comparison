%Display up to 1e-8 point
format long g

% Main script starts here
% Set up 3 x 3 SPD matrix hardcoded
fprintf('~~3 x 3 SPD matrix~~ \n');
A = single([1.5004, 1.3293, 0.8439; 1.3293, 1.2436, 0.6936; 0.8439, 0.6936, 1.2935]);
disp(A)

% Set given vector b = [1, 1, 1]
b = single([1; 1; 1]);

% Set up initial guess x_0 = [0, 0, 0]
x_ans = single([0; 0; 0]);
x_0 = single([0; 0; 0]);

% Set epsilon
eps = single(1e-6);

% Max number of iterations
maxItr = 27;

% Solve Ax = b with manual CG implementation
fprintf('Solve Ax = b with my_pcg()\n');
x_myPcg = my_pcg(A, b, eps, maxItr, x_0);
fprintf("\n\nx_sol = \n");
disp(x_myPcg);


% Answer key
% Solve Ax = b with pcg
fprintf('\n~~Answer Key~~\n');
fprintf('Solve Ax = b with pcg()\n');
x_ans = pcg(A, b, eps, maxItr);
disp(x_ans);

% Compare answer and my solution
validateSol(x_ans, x_myPcg);
















