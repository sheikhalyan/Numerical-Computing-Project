function RectangularField
    % Given values
    perimeter_val = 140;
    area_val = 1200;

    % Exact solution
    exact_length = sqrt(area_val);
    exact_width = exact_length;

    fprintf('Exact Length of the rectangular field: %.6f m\n', exact_length);
    fprintf('Exact Width of the rectangular field: %.6f m\n\n', exact_width);

    % Perimeter equation: 2L + 2W = P
    perimeter_eq = @(L, W) 2 * L + 2 * W;

    % Area equation: L*W = A
    area_eq = @(L, W) L * W;

    function [L, W, iterations] = secant_method(f, x0, x1, tol, max_iter)
        for iterations = 1:max_iter
            fx1 = f(x1);
            fx0 = f(x0);
            x_new = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0);
            if abs(x_new - x1) < tol
                break;
            end
            x0 = x1;
            x1 = x_new;
        end
        L = x_new;
        W = x_new;
    end

    function [L, W, iterations] = bisection_method(f, a, b, tol, max_iter)
        for iterations = 1:max_iter
            c = (a + b) / 2;
            if f(a) * f(c) < 0
                b = c;
            else
                a = c;
            end
            if abs(b - a) < tol
                break;
            end
        end
        L = c;
        W = c;
    end

    % Solve using Secant method
    [L_secant, W_secant, iterations_secant] = secant_method(@(L) perimeter_eq(L, 30) - perimeter_val, 0, 100, 1e-3, 100);
    [L_bisection, W_bisection, iterations_bisection] = bisection_method(@(L) perimeter_eq(L, 30) - perimeter_val, 0, 100, 1e-3, 100);

    % Print results
    fprintf('Length of the rectangular field (Secant method): %.6f m\n', L_secant);
    fprintf('Width of the rectangular field (Secant method): %.6f m\n', W_secant);
    fprintf('Number of iterations for Secant method: %d\n\n', iterations_secant);

    fprintf('Length of the rectangular field (Bisection method): %.6f m\n', L_bisection);
    fprintf('Width of the rectangular field (Bisection method): %.6f m\n', W_bisection);
    fprintf('Number of iterations for Bisection method: %d\n\n', iterations_bisection);

    % Calculate errors
    error_length_secant = abs(L_secant - exact_length);
    error_width_secant = abs(W_secant - exact_width);
    error_length_bisection = abs(L_bisection - exact_length);
    error_width_bisection = abs(W_bisection - exact_width);

    fprintf('Error in Length (Secant): %.6f m\n', error_length_secant);
    fprintf('Error in Width (Secant): %.6f m\n\n', error_width_secant);

    fprintf('Error in Length (Bisection): %.6f m\n', error_length_bisection);
    fprintf('Error in Width (Bisection): %.6f m\n\n', error_width_bisection);

    % Plotting
    figure;
    plot(1:iterations_secant+1, [exact_length*ones(1, iterations_secant+1); exact_width*ones(1, iterations_secant+1)], 'k--', 'DisplayName', 'Exact');
    hold on;
    plot(1:iterations_secant+1, [L_secant*ones(1, iterations_secant+1); W_secant*ones(1, iterations_secant+1)], '-o', 'DisplayName', 'Secant');
    xlabel('Iteration');
    ylabel('Value');
    title('Convergence of Secant Method for Rectangular Field');
    legend('Location', 'best');
    grid on;
    hold off;

    figure;
    plot(1:iterations_bisection+1, [exact_length*ones(1, iterations_bisection+1); exact_width*ones(1, iterations_bisection+1)], 'k--', 'DisplayName', 'Exact');
    hold on;
    plot(1:iterations_bisection+1, [L_bisection*ones(1, iterations_bisection+1); W_bisection*ones(1, iterations_bisection+1)], '-o', 'DisplayName', 'Bisection');
    xlabel('Iteration');
    ylabel('Value');
    title('Convergence of Bisection Method for Rectangular Field');
    legend('Location', 'best');
    grid on;
    hold off;
end
