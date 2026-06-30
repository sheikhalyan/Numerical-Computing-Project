function [root_newton_diode, root_bisection_diode, root_false_position_diode] = DiodeeSolver()
    % Diode Equation Parameters
    Is = 1e-6;
    n = 1.5;
    Vt = 0.025;
    V_guess = 3;

    % Newton's Method for Diode Equation
    [root_newton_diode, ~, ~, x_vals_newton] = newton_method(@f_diode, @dfdx_diode, V_guess, 1e-6, 100);

    % Bisection Method for Diode Equation
    [root_bisection_diode, ~, ~, x_vals_bisection] = bisection_method(@f_diode, 0.0, 5.0, 1e-6, 100);

    % False Position (Regula Falsi) Method for Diode Equation
    [root_false_position_diode, ~, ~, x_vals_false_position] = false_position_method(@f_diode, 0.0, 5.0, 1e-6, 100);

    % Nested Functions
    function result = f_diode(V)
        result = Is * (exp(V / (n * Vt)) - 1) - 5 / 1000;
    end

    function result = dfdx_diode(V)
        result = Is * exp(V / (n * Vt)) / (n * Vt);
    end

   % Plotting
figure;

subplot(3, 1, 1);
plot(1:length(x_vals_newton), x_vals_newton, '-o', 'DisplayName', "Newton's Method");
xlabel('Iteration');
ylabel('Value of V');
title("Convergence of Newton's Method for Diode Equation");
legend('Location', 'best');

subplot(3, 1, 2);
plot(1:length(x_vals_bisection), x_vals_bisection, '-o', 'DisplayName', "Bisection Method");
hold on;
plot(1:length(x_vals_false_position), x_vals_false_position, '-o', 'DisplayName', "False Position Method");
xlabel('Iteration');
ylabel('Value of V');
title("Convergence of Bisection and False Position Methods for Diode Equation");
legend('Location', 'best');

subplot(3, 1, 3);
plot(1:length(x_vals_newton), x_vals_newton, '-o', 'DisplayName', "Newton's Method");
hold on;
plot(1:length(x_vals_bisection), x_vals_bisection, '-o', 'DisplayName', "Bisection Method");
plot(1:length(x_vals_false_position), x_vals_false_position, '-o', 'DisplayName', "False Position Method");
xlabel('Iteration');
ylabel('Value of V');
title("Convergence of Newton's, Bisection, and False Position Methods for Diode Equation");
legend('Location', 'best');

end

% Newton's Method
function [x, iterations, x_vals, root] = newton_method(f, dfdx, x0, tol, max_iter)
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
    root = x;
end

% Bisection Method
function [x, iterations, x_vals, root] = bisection_method(f, a, b, tol, max_iter)
    x_vals = [a; b];
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
    root = x;
end

% False Position (Regula Falsi) Method
function [x, iterations, x_vals, root] = false_position_method(f, a, b, tol, max_iter)
    x_vals = [a; b];
    for iterations = 1:max_iter
        fa = f(a);
        fb = f(b);
        if abs(fb - fa) < 1e-10 % Check if denominator is close to zero
            break
        end
        c = (a * fb - b * fa) / (fb - fa);
        fc = f(c);
        x_vals = [x_vals; c];
        if abs(fc) < tol
            break
        end
        if fa * fc < 0
            b = c;
        else
            a = c;
        end
    end
    x = c;
    root = x;
end

% Example Usage
[root_newton_diode, root_bisection_diode, root_false_position_diode] = DiodeeSolver();
disp(['Newton Root: ', num2str(root_newton_diode)]);
disp(['Bisection Root: ', num2str(root_bisection_diode)]);
disp(['False Position Root: ', num2str(root_false_position_diode)]);
