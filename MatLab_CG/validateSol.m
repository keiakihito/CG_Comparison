%
%% Define the validate function
function[] = validateSol(x_ans, x_myPcg)
    % Initialize maximum error
    max_error = 0;

    % Calculate the absolute error
    error_vector = abs(single(x_ans - x_myPcg));

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