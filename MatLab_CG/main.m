
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
    disp(delta_new);

    %Set counter
    wkr = 0;

    while wkr < maxitr && delta_new > tol * tol

        % q <- Ad
        q = A * d;
%        disp(q);

        %alpha <- delta_{new} / (d^{T}*q)
        alpha = delta_new / (d' * q);
%        disp(alpha);

        %x_{i+1} <- x_{i} + alpha * d
        x_sol = x_sol + alpha * d;
%        disp(x_sol)

        if (mod(wkr, 50) == 0 && wkr ~= 0)
            % r <- b - Ax
            r = b - A * x_sol; % Recompute residual
        else
            % r <- r - alpha*q
            r = r - alpha * q;
%            disp(r);
        end % end of if

        %delta_{old} <- delta_{new}
        delta_old = delta_new;

        %delta_{new} <- r^{T} * r
        delta_new = r' * r;
    %    disp(delta_new);

        %beta <- delta_{new} / delta_{old}
        beta = delta_new / delta_old;
    %    disp(beta);

        % d_{i+1} <- r_{i+1} + beta * d_{i}
        d = r + beta * d;
%        disp(d);

        % Increment counter
        wkr = wkr + 1;
    end % end of while

    if(wkr < maxitr)
        fprintf("my_pcg() converged at iteration %d\n", wkr);
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
%disp(A)

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
maxItr = 100000;

%%Answer key
%%Solve Ax = b with pcg
%fprintf('Slove Ax = b with pcg()\n');
%x_ans = pcg(A, b, eps, maxItr);
%disp(x_ans);

%Solve Ax = b with manual CG implemenation
fprintf('Slove Ax = b with my_pcg()\n');
x_myPcg = my_pcg(A, b, eps, maxItr, x_0);
%disp(x_myPcg)

%Compare answer and my solution
%validate(x_ans, x_myPcg);



