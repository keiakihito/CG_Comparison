%q
format long g

% Values in CUDA iteration
r = single([	-0.043923; 0.084984; -0.041059]);
q = single([-0.001952; 0.003781; -0.001825]);
alpha = single(22.483810);
ngtAlpha = -alpha;

r1 = single(r + ngtAlpha * q);
disp(r1);

%{
expect
r =
-0.000041
-0.000037
-0.000030
%}