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
validate(x_ans, x_myPcg);







%% Define the my_pcg function
function[x_sol] = my_pcg(A, b, tol, maxitr, x_0)
    % Set up return value, x_sol
    x_sol = x_0;

    Ax = A * x_0;
    % disp(Ax);

    % r <- b - Ax
    r = b - Ax;
    % disp(r);

    % d <- r
    d = r;
    % disp(d);

    % delta_new <- r^{T} * r
    delta_new = r' * r;
    disp(delta_new);

    %Save it the calculation for relative residual
    initial_delta = delta_new;

    % Set counter
    wkr = 1;

    while wkr < maxitr && delta_new > tol * tol
         fprintf("\n\n= = = =  Iteration %d = = = = \n", wkr);

        % q <- Ad
        q = A * d;
        fprintf("\nq = \n");
        disp(q);

        % Set dot <- (d^{T} * q)
        dot = (d' * q);
        % disp(dot);

        % alpha <- delta_{new} / dot
        alpha = delta_new / dot;
        fprintf("\nalpha =  %f\n", alpha);

        % x_{i+1} <- x_{i} + alpha * d
        x_sol = x_sol + alpha * d;
        fprintf("\nx_sol = \n");
        disp(x_sol);

        if (mod(wkr, 50) == 0)
            % r <- b - Ax
            r = b - A * x_sol; % Recompute residual
        else
            fprintf("\n\n~~Before r <- r - alpha * q ~~\n");
            fprintf("\nr = \n");
            disp(r);
            fprintf("\nalpha = %f", alpha);
            fprintf("\n\nq = \n");
            disp(q);
            % r <- r - alpha * q
            r = r - alpha * q;

            fprintf("\n\n~~After r <- r - alpha * q ~~\n");
            fprintf("\nr = \n");
            disp(r);
            fprintf("\nalpha = %f\n", alpha);
            fprintf("\n\nq = \n");
            disp(q);
        end % end of if

        % delta_{old} <- delta_{new}
        delta_old = delta_new;
        fprintf("\ndelta_old =  %f\n", delta_old);

        % delta_{new} <- r^{T} * r
        delta_new = r' * r;
        fprintf("\ndelta_new =  %f\n", delta_new);

        % beta <- delta_{new} / delta_{old}
        beta = delta_new / delta_old;
        fprintf("\nbeta =  %f\n", beta);

        % d_{i+1} <- r_{i+1} + beta * d_{i}
        d = r + beta * d;
        fprintf("\nd = \n");
        disp(d);

        % Calculate relateve residual
        relative_residual = sqrt(delta_new) / sqrt(initial_delta);
        fprintf("\n\nRelative residual = %f\n", relative_residual);


        % Increment counter
        wkr = wkr + 1;
    end % end of while

    if(wkr < maxitr)
        fprintf("\n\nmy_pcg() converged at iteration %d\n", wkr-1);
    end % end of if

    itr = wkr - 1;
    fprintf("\n\nRelative residual = %f", relative_residual);
end % end of function


%
%% Define the validate function
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






