
% Manual Conjugeta Gradient
function[x_sol] = my_pcg(A, b, tol, maxitr, x_0);
    %Set up return value, x_sol
    x_sol = x_0;

    Ax = A * x_0;
%    disp(Ax);

    % r<- b-Ax
    r = b - Ax;
%    disp(r);

    % d <- r
    d = r;
%    disp(d);

    %delta_new <- r^{T}*r
    delta_new = r' * r;
    disp(delta_new);

    %Set counter
    wkr = 1;

    while wkr < maxitr && delta_new > tol * tol
%        fprintf("\n\n= = = =  Iteartion %d = = = = \n", wkr);

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

        %x_{i+1} <- x_{i} + alpha * d
        x_sol = x_sol + alpha * d;
        fprintf("\nx_sol = \n");
        disp(x_sol);

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
        disp(delta_old);

        %delta_{new} <- r^{T} * r
        delta_new = r' * r;
        fprintf("\ndelta_new =  %f\n", delta_new);
        disp(delta_new);

        %beta <- delta_{new} / delta_{old}
        beta = delta_new / delta_old;
        fprintf("\nbeta =  %f\n", beta);
        disp(beta);


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

    itr =wkr - 1;
    fprintf("\n Iteration %d", itr);
    fprintf("\n Relative Error: delta_new = %f\n", delta_new);


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

%%% ~~Test 1~~~
%%%Set up 3 x 3 SPD matrix hardcoded
%fprintf('~~3 x 3 SPD matrix~~ \n');
%A = [1.5004 1.3293 0.8439; 1.3293 1.2436 0.6936; 0.8439 0.6936 1.2935];
%disp(A)
%
%% Set given vector b = [1, 1, 1]
%b = [1; 1; 1];
%
%% Set up initial guess x_0 = [0, 0, 0]
%x_ans = [0; 0; 0];
%x_0 = [0; 0; 0];




% ~~Test 2~~~
%Let N is row and column number of matrix
%N = 19;
%S = rand(N, N);
%%disp('Matrix S');
%%disp(S);
%
%% Create SPD matrix with A^{T}*A
%format long g;
%A = S' * S;
%disp('SPD: A');
%disp(A);
%
%b = ones(N, 1);
%x_ans = zeros(N,1);
%x_0 = zeros(N,1);



%Debug for CUDA 19 by 19
A = [
    9.3918, 6.6007, 6.3940, 6.2324, 6.0555, 4.9198, 5.9791, 4.3268, 5.6480, 5.2860, 7.4877, 4.5666, 4.9256, 6.4601, 6.5433, 6.0055, 6.4163, 4.9032, 5.5738;
    6.6007, 6.9943, 5.2786, 5.2829, 5.1832, 3.1949, 5.2337, 4.1912, 4.8611, 4.9841, 5.9790, 4.1163, 4.3596, 4.9439, 5.3257, 5.0629, 5.1506, 4.6618, 5.3536;
    6.3940, 5.2786, 6.6018, 4.6547, 5.5515, 4.2225, 4.7058, 3.7232, 4.6784, 4.0535, 5.7858, 3.8962, 4.4437, 4.9393, 5.1749, 4.5644, 5.7526, 5.1598, 4.5692;
    6.2324, 5.2829, 4.6547, 6.4837, 5.1888, 4.0699, 5.3413, 3.9317, 4.7237, 4.8594, 6.1045, 4.0298, 3.5701, 4.8748, 5.6800, 4.0319, 5.3133, 4.5944, 4.7613;
    6.0555, 5.1832, 5.5515, 5.1888, 6.4653, 4.3358, 4.9567, 3.8437, 4.2128, 4.5556, 5.7876, 4.0070, 4.4707, 4.5421, 5.7059, 4.2554, 5.9133, 5.2349, 5.0687;
    4.9198, 3.1949, 4.2225, 4.0699, 4.3358, 4.7261, 3.3840, 2.2092, 3.7733, 3.3743, 4.9615, 2.9836, 3.2188, 3.8934, 4.3613, 3.3114, 4.5301, 3.8181, 3.6113;
    5.9791, 5.2337, 4.7058, 5.3413, 4.9567, 3.3840, 7.0722, 4.0417, 4.5106, 4.7280, 5.7858, 4.1902, 3.9915, 4.9707, 5.7849, 5.0159, 4.7386, 4.3284, 5.2583;
    4.3268, 4.1912, 3.7232, 3.9317, 3.8437, 2.2092, 4.0417, 3.5335, 3.1209, 3.0387, 3.8865, 2.7753, 2.7530, 3.7246, 3.9503, 3.3912, 3.9507, 3.2583, 3.8240;
    5.6480, 4.8611, 4.6784, 4.7237, 4.2128, 3.7733, 4.5106, 3.1209, 6.4609, 4.1038, 6.0113, 4.2932, 4.1595, 5.5648, 4.8813, 4.1930, 4.1077, 4.0631, 4.3061;
    5.2860, 4.9841, 4.0535, 4.8594, 4.5556, 3.3743, 4.7280, 3.0387, 4.1038, 5.0346, 5.2025, 3.6426, 3.6059, 3.7162, 4.6563, 3.8483, 3.9120, 3.6490, 4.8429;
    7.4877, 5.9790, 5.7858, 6.1045, 5.7876, 4.9615, 5.7858, 3.8865, 6.0113, 5.2025, 7.9688, 5.4157, 4.6194, 6.4850, 6.7675, 5.3269, 6.1117, 5.3438, 5.7159;
    4.5666, 4.1163, 3.8962, 4.0298, 4.0070, 2.9836, 4.1902, 2.7753, 4.2932, 3.6426, 5.4157, 4.9784, 3.4655, 5.0211, 4.5979, 3.8621, 3.6243, 3.8400, 4.4969;
    4.9256, 4.3596, 4.4437, 3.5701, 4.4707, 3.2188, 3.9915, 2.7530, 4.1595, 3.6059, 4.6194, 3.4655, 4.5696, 4.1065, 3.8278, 3.7711, 3.8671, 3.8748, 4.1143;
    6.4601, 4.9439, 4.9393, 4.8748, 4.5421, 3.8934, 4.9707, 3.7246, 5.5648, 3.7162, 6.4850, 5.0211, 4.1065, 6.8418, 5.6289, 5.1061, 4.9464, 4.1055, 5.0408;
    6.5433, 5.3257, 5.1749, 5.6800, 5.7059, 4.3613, 5.7849, 3.9503, 4.8813, 4.6563, 6.7675, 4.5979, 3.8278, 5.6289, 7.0175, 4.3347, 5.5875, 4.9894, 5.4625;
    6.0055, 5.0629, 4.5644, 4.0319, 4.2554, 3.3114, 5.0159, 3.3912, 4.1930, 3.8483, 5.3269, 3.8621, 3.7711, 5.1061, 4.3347, 6.0332, 4.6887, 3.1360, 5.1251;
    6.4163, 5.1506, 5.7526, 5.3133, 5.9133, 4.5301, 4.7386, 3.9507, 4.1077, 3.9120, 6.1117, 3.6243, 3.8671, 4.9464, 5.5875, 4.6887, 6.8489, 5.0721, 4.6645;
    4.9032, 4.6618, 5.1598, 4.5944, 5.2349, 3.8181, 4.3284, 3.2583, 4.0631, 3.6490, 5.3438, 3.8400, 3.8748, 4.1055, 4.9894, 3.1360, 5.0721, 5.6506, 3.6533;
    5.5738, 5.3536, 4.5692, 4.7613, 5.0687, 3.6113, 5.2583, 3.8240, 4.3061, 4.8429, 5.7159, 4.4969, 4.1143, 5.0408, 5.4625, 5.1251, 4.6645, 3.6533, 6.4536
];

N = 19;
b = ones(N, 1);
x_ans = zeros(N,1);
x_0 = zeros(N,1);



% Set epsilon
eps = 1e-6;

%Maxnumber of iteration
maxItr = 27;


%Solve Ax = b with manual CG implemenation
fprintf('Slove Ax = b with my_pcg()\n');
x_myPcg = my_pcg(A, b, eps, maxItr, x_0);
fprintf("\nx_sol = \n");
disp(x_myPcg);

%Answer key
%Solve Ax = b with pcg
fprintf('\n~~Answer Key~~~\n');
fprintf('Slove Ax = b with pcg()\n');
x_ans = pcg(A, b, eps, maxItr);
%disp(x_ans);


%Compare answer and my solution
validate(x_ans, x_myPcg);

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

%}


