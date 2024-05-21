%Print Hello World,　✅
%disp('Hello, World');

%%Create a random 3 X 3 matrix S
%S = rand(3, 3);
%disp('Matrix S');
%disp(S);
%
%% Create SPD matrix with A^{T}*A
%A = S' * S;
%disp('SPD: A');
%disp(A);

%Set up 3 x 3 SPD matrix hardcoded
A = [1.5004 1.3293 0.8439; 1.3293 1.2436 0.6936; 0.8439 0.6936 1.2935];
disp(A)