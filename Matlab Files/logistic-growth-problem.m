function [root_newton_logistic, root_secant_logistic, root_bisection_logistic, x_vals_newton, x_vals_secant, x_vals_bisection] = logistic_growth_methods()
    % Logistic Growth Parameters
    r = 0.1;
    K = 50;
    target_population = 1.25; % Target logistic growth rate
    P0 = 10;

    % Newton's Method
    [root_newton_logistic, ~, x_vals_newton] = newton_method(@f_logistic, @dfdx_logistic, P0, 1e-6, 100);

    % Secant Method
    [root_secant_logistic, ~, x_vals_secant] = secant_method(@f_logistic, P0, P0 + 1, 1e-6, 100);

    % Bisection Method
    [root_bisection_logistic, ~, x_vals_bisection] = bisection_method(@f_logistic, 0.0, K, 1e-6, 100);

    % Nested Functions
    function result = f_logistic(P)
        result = logistic_growth(P, r, K) - target_population;
    end

    function result = dfdx_logistic(P)
        result = r * (1 - 2*P / K);
    end

    % Logistic Growth Model Function
    function result = logistic_growth(P, r, K)
        result = r * P * (1 - P / K);
    end
end

% Newton's Method
function [x, iterations, x_vals] = newton_method(f, dfdx, x0, tol, max_iter)
    x = x0;
    x_vals = [x];
    for iterations = 1:max_iter
        dx = -f(x) / dfdx(x);
        x = x + dx;
        x_vals = [x_vals; x];
        if abs(dx) < tol
            break
        end
    end
end

% Secant Method
function [x, iterations, x_vals] = secant_method(f, x0, x1, tol, max_iter)
    x_vals = [x0; x1];
    for iterations = 1:max_iter
        dx = -f(x1) * (x1 - x0) / (f(x1) - f(x0));
        x0 = x1;
        x1 = x1 + dx;
        x_vals = [x_vals; x1];
        if abs(dx) < tol
            break
        end
    end
    x = x1; % Return the root found by the Secant method
end

% Bisection Method
function [x, iterations, x_vals] = bisection_method(f, a, b, tol, max_iter)
    x_vals = [];
    for iterations = 1:max_iter
        c = (a + b) / 2;
        x_vals = [x_vals; c];
        if abs(b - a) < tol || abs(f(c)) < tol
            break
        elseif f(a) * f(c) < 0
            b = c;
        else
            a = c;
        end
    end
    x = c;
end

% Example Usage
[root_newton_logistic, root_secant_logistic, root_bisection_logistic, x_vals_newton, x_vals_secant, x_vals_bisection] = logistic_growth_methods();
disp(['Newton Root: ', num2str(root_newton_logistic)]);
disp(['Secant Root: ', num2str(root_secant_logistic)]);
disp(['Bisection Root: ', num2str(root_bisection_logistic)]);

% Plotting
figure;
subplot(3, 1, 1);
plot(1:size(x_vals_newton, 1), x_vals_newton, '-o');
title('Convergence of Newton''s Method');
xlabel('Iteration');
ylabel('Approximation');
grid on;

subplot(3, 1, 2);
plot(1:size(x_vals_secant, 1), x_vals_secant, '-o');
title('Convergence of Secant Method');
xlabel('Iteration');
ylabel('Approximation');
grid on;

subplot(3, 1, 3);
plot(1:size(x_vals_bisection, 1), x_vals_bisection, '-o');
title('Convergence of Bisection Method');
xlabel('Iteration');
ylabel('Approximation');
grid on;
