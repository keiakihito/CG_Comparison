
% Manual Conjugeta Gradient
function[x_sol] = my_pcg(A, b, tol, maxitr, x_0);
    %Set up return value, x_sol
    x_sol = x_0;

    Ax = A * x_0;
%    disp(Ax);

    % r<- b-Ax
    r = b - Ax;
    %disp(r);

    % d <- r
    d = r;
%    disp(d);

    %delta_new <- r^{T}*r
    delta_new = r' * r;
%    disp(delta_new);

    %Set counter
    wkr = 1;

    while wkr < maxitr && delta_new > tol * tol
        fprintf("\n\n= = = =  Iteartion %d = = = = \n", wkr);

        % q <- Ad
        q = A * d;
        fprintf("\nq = \n");
        disp(q);

        %Set dot <- (d^{T}*q)
        dot = (d' *q);
%        disp(dot);

        %alpha <- delta_{new} /
        alpha = delta_new / dot;
        fprintf("\nalpha =  %f\n", alpha);
%        disp(alpha);

        %x_{i+1} <- x_{i} + alpha * d
        x_sol = x_sol + alpha * d;
        fprintf("\nx_sol = \n");
        disp(x_sol)

        if (mod(wkr, 50) == 0 )
            % r <- b - Ax
            r = b - A * x_sol; % Recompute residual
        else
            % r <- r - alpha*q
            r = r - alpha * q;
            fprintf("\nr = \n");
            disp(r);
        end % end of if

        %delta_{old} <- delta_{new}
        delta_old = delta_new;
        fprintf("\ndelta_old =  %f\n", delta_old);
%        disp(delta_old);

        %delta_{new} <- r^{T} * r
        delta_new = r' * r;
        fprintf("\ndelta_new =  %f\n", delta_new);
%        disp(delta_new);

        %beta <- delta_{new} / delta_{old}
        beta = delta_new / delta_old;
        fprintf("\nbeta =  %f\n", beta);
%        disp(beta);


        % d_{i+1} <- r_{i+1} + beta * d_{i}
        d = r + beta * d;
        fprintf("\nd = \n");
        disp(d);

        % Increment counter
        wkr = wkr + 1;
    end % end of while

    if(wkr < maxitr)
        fprintf("\n\nmy_pcg() converged at iteration %d\n", wkr);
    end %end of if


end % end of function

%Error checking
function[] = validate(x_ans, x_myPcg)
    % Initialize maximum error
    max_error = 0;

    % Calculate the absolute error
    error_vector = abs(x_ans - x_myPcg);

    % Check and display if error exceeds previous maximum
    for i = 1:length(error_vector)
        if error_vector(i) > max_error
            max_error = error_vector(i);
            fprintf('Current max error at index %d: %e\n', i, max_error);
        end % end of if
    end % end of for

    % Display the final maximum error
    fprintf('Max error: %e\n', max_error);
end % end of validate





%Main function

%% ~~Test 1~~~
%%Set up 3 x 3 SPD matrix hardcoded
fprintf('~~3 x 3 SPD matrix~~ \n');
A = [1.5004 1.3293 0.8439; 1.3293 1.2436 0.6936; 0.8439 0.6936 1.2935];
disp(A)

% Set given vector b = [1, 1, 1]
b = [1; 1; 1];

% Set up initial guess x_0 = [0, 0, 0]
x_ans = [0; 0; 0];
x_0 = [0; 0; 0];




%% ~~Test 2~~~
%%Let N is row and column number of matrix
%N = 10000;
%S = rand(N, N);
%%disp('Matrix S');
%%disp(S);
%
%% Create SPD matrix with A^{T}*A
%A = S' * S;
%%disp('SPD: A');
%%disp(A);
%
%b = ones(N, 1);
%x_ans = zeros(N,1);
%x_0 = zeros(N,1);



% Set epsilon
eps = 1e-6;

%Maxnumber of iteration
maxItr = 10000;


%Solve Ax = b with manual CG implemenation
fprintf('Slove Ax = b with my_pcg()\n');
x_myPcg = my_pcg(A, b, eps, maxItr, x_0);
fprintf("\nx_sol = \n");
disp(x_myPcg)

%Answer key
%Solve Ax = b with pcg
fprintf('\n~~Answer Key~~~\n');
fprintf('Slove Ax = b with pcg()\n');
x_ans = pcg(A, b, eps, maxItr);
disp(x_ans);


%Compare answer and my solution
%validate(x_ans, x_myPcg);

%{
Sample Run
~~3 x 3 SPD matrix~~
    1.5004    1.3293    0.8439
    1.3293    1.2436    0.6936
    0.8439    0.6936    1.2935

Slove Ax = b with my_pcg()


= = = =  Iteartion 1 = = = =

q =
    3.6736
    3.2665
    2.8310


alpha =  0.307028

x_sol =
    0.3070
    0.3070
    0.3070


r =
   -0.1279
   -0.0029
    0.1308


delta_old =  3.000000

delta_new =  0.033476

beta =  0.011159

d =
   -0.1167
    0.0083
    0.1420



= = = =  Iteartion 2 = = = =

q =
   -0.0444
   -0.0465
    0.0908


alpha =  1.892012

x_sol =
    0.0862
    0.3226
    0.5756


r =
   -0.0439
    0.0850
   -0.0411


delta_old =  0.033476

delta_new =  0.010837

beta =  0.323739

d =
   -0.0817
    0.0877
    0.0049



= = = =  Iteartion 3 = = = =

q =
   -0.0020
    0.0038
   -0.0018


alpha =  22.483763

x_sol =
   -1.7511
    2.2935
    0.6858


r =
   1.0e-13 *

    0.3909
    0.3482
    0.2869


delta_old =  0.010837

delta_new =  0.000000

beta =  0.000000

d =
   1.0e-13 *

    0.3909
    0.3482
    0.2869



my_pcg() converged at iteration 4

x_sol =
   -1.7511
    2.2935
    0.6858


~~Answer Key~~~
Slove Ax = b with pcg()
pcg converged at iteration 3 to a solution with relative residual 3.4e-14.
   -1.7511
    2.2935
    0.6858


Process finished with exit code 0

%}%


