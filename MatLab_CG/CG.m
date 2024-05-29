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


%{

~~3 x 3 SPD matrix~~
          1.5004          1.3293          0.8439
          1.3293          1.2436          0.6936
          0.8439          0.6936          1.2935

Solve Ax = b with my_pcg()
     3



💫💫💫= = = =  Iteration 1 = = = = 💫💫💫

q =
          3.6736
          3.2665
           2.831


alpha =  0.307028

x_sol =
       0.3070279
       0.3070279
       0.3070279



~~Before r <- r - alpha * q ~~

r =
     1
     1
     1


alpha = 0.307028

ngtAlpha = -0.307028

q =
          3.6736
          3.2665
           2.831



~~After r <- r - alpha * q ~~

r =
      -0.1278976
    -0.002906561
       0.1308041


alpha = 0.307028


q =
          3.6736
          3.2665
           2.831


delta_old =  3.000000

delta_new =  0.033476

beta =  0.011159

d =
       -0.116739
     0.008252095
       0.1419628



Relative residual = 0.105635


💫💫💫= = = =  Iteration 2 = = = = 💫💫💫

q =
     -0.04438324
     -0.04645342
       0.0908365


alpha =  1.892012

x_sol =
      0.08615638
       0.3226409
       0.5756232



~~Before r <- r - alpha * q ~~

r =
      -0.1278976
    -0.002906561
       0.1308041


alpha = 1.892012

ngtAlpha = -1.892012

q =
     -0.04438324
     -0.04645342
       0.0908365



~~After r <- r - alpha * q ~~

r =
     -0.04392401
      0.08498386
     -0.04105961


alpha = 1.892012


q =
     -0.04438324
     -0.04645342
       0.0908365


delta_old =  0.033476

delta_new =  0.010837

beta =  0.323739

d =
     -0.08171692
      0.08765538
     0.004899234



Relative residual = 0.060104


💫💫💫= = = =  Iteration 3 = = = = 💫💫💫

q =
    -0.001953309
     0.003780025
    -0.001825981


alpha =  22.483807

x_sol =
       -1.751151
        2.293468
       0.6857766



~~Before r <- r - alpha * q ~~

r =
     -0.04392401
      0.08498386
     -0.04105961


alpha = 22.483807

ngtAlpha = -22.483807

q =
    -0.001953309
     0.003780025
    -0.001825981



~~After r <- r - alpha * q ~~

r =
   -6.180257e-06
   -5.498528e-06
   -4.611909e-06


alpha = 22.483807


q =
    -0.001953309
     0.003780025
    -0.001825981


delta_old =  0.010837

delta_new =  0.000000

beta =  0.000000

d =
   -6.180933e-06
   -5.497803e-06
   -4.611869e-06



Relative residual = 0.000005


💫💫💫= = = =  Iteration 4 = = = = 💫💫💫

q =
   -2.047406e-05
   -1.825217e-05
   -1.499482e-05


alpha =  0.302987

x_sol =
       -1.751153
        2.293466
       0.6857752



~~Before r <- r - alpha * q ~~

r =
   -6.180257e-06
   -5.498528e-06
   -4.611909e-06


alpha = 0.302987

ngtAlpha = -0.302987

q =
   -2.047406e-05
   -1.825217e-05
   -1.499482e-05



~~After r <- r - alpha * q ~~

r =
    2.310708e-08
     3.16345e-08
   -6.868186e-08


alpha = 0.302987


q =
   -2.047406e-05
   -1.825217e-05
   -1.499482e-05


delta_old =  0.000000

delta_new =  0.000000

beta =  0.000070

d =
    2.267628e-08
    3.125131e-08
    -6.90033e-08



Relative residual = 0.000000


my_pcg() converged at iteration 4


Relative residual = 0.000000

x_sol =
       -1.751153
        2.293466
       0.6857752


~~Answer Key~~
Solve Ax = b with pcg()
pcg converged at iteration 4 to a solution with relative residual 3.4e-08.
       -1.751153
        2.293466
       0.6857752

Max error: 0.000000e+00
%}














