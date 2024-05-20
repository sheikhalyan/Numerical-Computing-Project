import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

# LOGISTIC GROWTH
# Function definitions for logistic growth and numerical methods
def logistic_growth(P, r, K):
    return r * P * (1 - P / K)

def newton_method(f, dfdx, x0, tol=1e-6, max_iter=100):
    x = x0
    x_vals = []
    for i in range(max_iter):
        dx = -f(x) / dfdx(x)
        x += dx
        x_vals.append(x)
        if abs(dx) < tol:
            break
    return x, i+1, x_vals

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    x_prev = x0
    x = x1
    x_vals = []
    for i in range(max_iter):
        dx = -f(x) * (x - x_prev) / (f(x) - f(x_prev))
        x_prev = x
        x += dx
        x_vals.append(x)
        if abs(dx) < tol:
            break
    return x, i+1, x_vals

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    x_vals = []
    for i in range(max_iter):
        c = (a + b) / 2
        x_vals.append(c)
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            break
        elif f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, i+1, x_vals

# Function to plot errors for each method
def plot_errors(method, errors):
    plt.figure()
    plt.plot(errors, label=method, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title(f'Convergence of {method} for Logistic Growth Model')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot exact vs approximate solutions
def plot_exact_vs_approx(method, x_vals, exact_value):
    plt.figure()
    plt.plot(range(1, len(x_vals)+1), x_vals, label=f'Approximate ({method})', marker='o')
    plt.axhline(y=exact_value, color='r', linestyle='-', label='Exact Solution')
    plt.xlabel('Iteration')
    plt.ylabel('Population')
    plt.title(f'Exact vs Approximate Solution ({method})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to calculate logistic growth using selected method and display results
def calculate_logistic():
    target_population = float(target_population_entry.get())
    P0 = float(P0_entry.get())
    r = float(r_entry.get())
    K = float(K_entry.get())

    def f_logistic(P):
        return logistic_growth(P, r, K) - target_population

    def dfdx_logistic(P):
        return r * (1 - 2 * P / K)

    method = method_combobox_logistic.get()

    # Calculate a high-precision solution to use as the "exact" value

    high_precision_root, _, _ = newton_method(f_logistic, dfdx_logistic, P0, tol=1e-12)

    if method == "Newton's Method":
        root, iterations, x_vals = newton_method(f_logistic, dfdx_logistic, P0)
    elif method == "Secant Method":
        root, iterations, x_vals = secant_method(f_logistic, P0, P0 + 1)
    elif method == "Bisection Method":
        root, iterations, x_vals = bisection_method(f_logistic, 0.0, K)

    # Calculate errors
    errors = [abs(f_logistic(x)) for x in x_vals]

    # Display results, errors, exact, and approximate values
    result_message = f"{method}: {root:.15f}\nIterations: {iterations}\n"
    error_message = "\n".join(f"Iteration {i + 1}: {error:.15f}" for i, error in enumerate(errors))
    exact_solution_message = f"Exact Solution: {high_precision_root:.15f}"
    approximate_solution_message = f"Approximate Solution: {root:.15f}"
    messagebox.showinfo("Results",result_message+ "\n" + exact_solution_message + "\n" + approximate_solution_message + "\n" + error_message )

    # Plot errors
    plot_errors(method, errors)

    # Plot exact vs approximate solutions
    plot_exact_vs_approx(method, x_vals, high_precision_root)


# Function definitions for diode equation and numerical methods
def diode_equation(V, Is, n, Vt, I_load):
    return Is * (np.exp(V / (n * Vt)) - 1) - I_load

def diode_equation_derivative(V, Is, n, Vt):
    return Is * np.exp(V / (n * Vt)) / (n * Vt)

def newton_method_diode(f, dfdx, x0, Is, n, Vt, I_load, tol=1e-6, max_iter=100):
    x = x0
    x_vals = [x]
    errors = []
    for _ in range(max_iter):
        dx = -f(x, Is, n, Vt, I_load) / dfdx(x, Is, n, Vt)
        x += dx
        x_vals.append(x)
        error = np.abs(f(x, Is, n, Vt, I_load))
        errors.append(error)
        if error < tol:
            break
    return x, x_vals, errors, len(x_vals) - 1

def bisection_method_diode(f, a, b, Is, n, Vt, I_load, tol=1e-6, max_iter=100):
    x_vals = []
    errors = []
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c, Is, n, Vt, I_load)
        x_vals.append(c)
        error = np.abs(fc)
        errors.append(error)
        if error < tol:
            break
        if f(a, Is, n, Vt, I_load) * fc < 0:
            b = c
        else:
            a = c
    return c, x_vals, errors, len(x_vals) - 1

def false_position_method(f, a, b, Is, n, Vt, I_load, tol=1e-6, max_iter=100):
    x_vals = []
    errors = []
    for _ in range(max_iter):
        fa = f(a, Is, n, Vt, I_load)
        fb = f(b, Is, n, Vt, I_load)
        if np.abs(fb - fa) < 1e-10:  # Check if denominator is close to zero
            break
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c, Is, n, Vt, I_load)
        x_vals.append(c)
        error = np.abs(fc)
        errors.append(error)
        if error < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, x_vals, errors, len(x_vals) - 1

# Function to plot exact vs approximate solutions
def plot_exact_vs_approx(method, x_vals, exact_value):
    plt.figure()
    plt.plot(range(1, len(x_vals)+1), x_vals, label=f'Approximate ({method})', marker='o')
    plt.axhline(y=exact_value, color='r', linestyle='-', label='Exact Solution')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage (V)')
    plt.title(f'Exact vs Approximate Solution ({method})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to calculate diode equation using selected method and display results
def calculate_diode():
    Is = float(Is_entry.get())
    n = float(n_entry.get())
    Vt = float(Vt_entry.get())
    I_load = float(I_load_entry.get())
    V_guess = float(V_guess_entry.get())
    method = method_combobox_diode.get()

    # Calculate a high-precision solution to use as the "exact" value
    exact_root, _, _, _ = newton_method_diode(diode_equation, diode_equation_derivative, V_guess, Is, n, Vt, I_load, tol=1e-12)

    if method == "Newton's Method":
        root, x_vals, errors, iterations = newton_method_diode(diode_equation, diode_equation_derivative, V_guess, Is, n, Vt, I_load)
    elif method == "Bisection Method":
        root, x_vals, errors, iterations = bisection_method_diode(diode_equation, 0.0, 5.0, Is, n, Vt, I_load)
    elif method == "False Position Method":
        root, x_vals, errors, iterations = false_position_method(diode_equation, 0.0, 5.0, Is, n, Vt, I_load)

    # Display results and errors
    exact_solution_message = f"Exact Solution: {exact_root:.6f} V"
    approximate_solution_message = f"Approximate Solution: {root:.6f} V"
    result_message = f"Root of the Diode Equation: {root:.6f} V\nIterations: {iterations}\n"
    error_message = "Errors: " + ", ".join([f"{error:.6e}" for error in errors])
    messagebox.showinfo("Result", f"{exact_solution_message}\n{approximate_solution_message}\n{result_message}\n{error_message}")


    # Plot error convergence
    plt.figure()
    plt.plot(range(1, len(errors)+1), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.show()

    # Plot exact vs approximate solutions
    plot_exact_vs_approx(method, x_vals, exact_root)

# Function definitions for rectangular field problem
# Function definitions for rectangular field problem
# Function definitions for rectangular field problem
def perimeter_eq(L, W):
    return 2 * L + 2 * W

def area_eq(L, W):
    return L * W

def solve_rectangular_field(perimeter_val, area_val, method):
    L0 = 30.0  # Initial guess for L

    def f(L, W):
        return np.array([perimeter_eq(L, W) - perimeter_val, area_eq(L, W) - area_val])

    def dfdx(L, W):
        return np.array([[2, 2], [W, L]])

    def newton_method(f, dfdx, x0, tol=1e-3, max_iter=100):
        x = x0.astype(float)
        x_vals = [x.copy()]
        errors = []
        for i in range(1, max_iter+1):
            dx = np.linalg.lstsq(dfdx(*x), -f(*x), rcond=None)[0]
            x += dx
            x_vals.append(x.copy())
            error = np.linalg.norm(f(*x))
            errors.append(error)
            if error < tol:
                break
        return x, i, errors, x_vals

    def bisection_method(f, a, b, tol=1e-3, max_iter=100):
        x_vals = [(a, a), (b, b)]
        errors = []
        for i in range(1, max_iter+1):
            c = (a + b) / 2
            x_vals.append((c, c))
            fc = f(c)
            error = np.abs(fc)
            errors.append(error)
            if error < tol or np.abs(b - a) / 2 < tol:
                break
            if f(a) * fc < 0:
                b = c
            else:
                a = c
        return (c, c), i, errors, x_vals

    def secant_method(f, x0, x1, tol=1e-3, max_iter=100):
        x_vals = [(x0, x0), (x1, x1)]
        errors = []
        for i in range(1, max_iter+1):
            x_next = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            x_vals.append((x_next, x_next))
            error = np.abs(f(x_next))
            errors.append(error)
            if error < tol:
                break
            x0, x1 = x1, x_next
        return (x_next, x_next), i, errors, x_vals

    if method == "Newton's Method":
        result, iterations, errors, x_vals = newton_method(f, dfdx, np.array([L0, L0]))
    elif method == "Bisection Method":
        result, iterations, errors, x_vals = bisection_method(lambda L: f(L, L0)[0], 0, 100)
    elif method == "Secant Method":
        result, iterations, errors, x_vals = secant_method(lambda L: f(L, L0)[0], 0, 100)
    else:
        return None, None, None, None

    return result, iterations, errors, x_vals

def calculate_rectangular_field():
    perimeter_val = float(perimeter_entry_rec.get())
    area_val = float(area_entry_rec.get())
    method = method_combobox_rectangular.get()

    if method not in ["Newton's Method", "Bisection Method", "Secant Method"]:
        messagebox.showerror("Error", "Invalid method selected.")
        return

    result, iterations, errors, x_vals = solve_rectangular_field(perimeter_val, area_val, method)

    if result is None:
        messagebox.showerror("Error", "An error occurred while solving.")
    else:
        approximate_solution_str = f"Approximate Solution - L: {x_vals[-1][0]:.6f} m, W: {x_vals[-1][1]:.6f} m"
        exact_solution_str = f"Exact Solution - L: {result[0]:.6f} m, W: {result[1]:.6f} m"
        error_string = "\n".join([f"Iteration {i+1}: {error:.15f}" for i, error in enumerate(errors)])
        messagebox.showinfo("Result", f"{exact_solution_str}\n{approximate_solution_str}\n\nNumber of iterations: {iterations}\nErrors:\n{error_string}")

        # Plotting the error convergence graph
        plt.figure()
        plt.plot(range(1, len(errors)+1), errors, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title(f'Error Convergence for {method}')
        plt.grid(True)
        plt.show()

        # Plotting the exact vs. approximate solution
        L_vals = [x[0] for x in x_vals]
        plt.figure()
        plt.plot(range(1, len(L_vals)+1), L_vals, label=f'Approximate ({method})', marker='o')
        plt.axhline(y=result[0], color='r', linestyle='-', label='Exact Solution')
        plt.xlabel('Iteration')
        plt.ylabel('Length (L)')
        plt.title(f'Exact vs Approximate Solution ({method})')
        plt.legend()
        plt.grid(True)
        plt.show()


#root = tk.Tk()
#root.title("Rectangular Field Solver")


root = tk.Tk()
root.title("Numerical Example Solver")
root.geometry("390x340")

# Tab Control
tab_control = ttk.Notebook(root)

# Tab 1 - Logistic Growth Model
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Logistic Growth Model')

# Widgets
tk.Label(tab1, text="Initial Population (P0):").grid(row=0, column=0, padx=10, pady=10)
P0_entry = tk.Entry(tab1, width=30)  # Fixed width
P0_entry.grid(row=0, column=1, padx=10, pady=10)
P0_entry.insert(0, "10")

tk.Label(tab1, text="Growth Rate (r):").grid(row=1, column=0, padx=10, pady=10)
r_entry = tk.Entry(tab1, width=30)  # Fixed width
r_entry.grid(row=1, column=1, padx=10, pady=10)
r_entry.insert(0, "0.1")

tk.Label(tab1, text="Carrying Capacity (K):").grid(row=2, column=0, padx=10, pady=10)
K_entry = tk.Entry(tab1, width=30)  # Fixed width
K_entry.grid(row=2, column=1, padx=10, pady=10)
K_entry.insert(0, "50")

tk.Label(tab1, text="Target Population:").grid(row=3, column=0, padx=10, pady=10)
target_population_entry = tk.Entry(tab1, width=30)  # Fixed width
target_population_entry.grid(row=3, column=1, padx=10, pady=10)
target_population_entry.insert(0, "1.25")

tk.Label(tab1, text="Select Method:").grid(row=4, column=0, padx=10, pady=10)
method_combobox_logistic = ttk.Combobox(tab1, values=["Newton's Method", "Secant Method", "Bisection Method"], width=30)  # Fixed width
method_combobox_logistic.grid(row=4, column=1, padx=10, pady=10)

calculate_button_logistic = tk.Button(tab1, text="Calculate", command=calculate_logistic, width=30, font=("Arial", 12))  # Fixed width and font size
calculate_button_logistic.grid(row=5, columnspan=2, pady=20)

# Tab 2 - Diode Equation Solver
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Diode Equation Solver')

# Widgets
tk.Label(tab2, text="Is:").grid(row=0, column=0, padx=10, pady=10)
Is_entry = tk.Entry(tab2, width=30)  # Fixed width
Is_entry.grid(row=0, column=1, padx=10, pady=10)
Is_entry.insert(0, "1e-6")

tk.Label(tab2, text="n:").grid(row=1, column=0, padx=10, pady=10)
n_entry = tk.Entry(tab2, width=30)  # Fixed width
n_entry.grid(row=1, column=1, padx=10, pady=10)
n_entry.insert(0, "1.5")

tk.Label(tab2, text="Vt:").grid(row=2, column=0, padx=10, pady=10)
Vt_entry = tk.Entry(tab2, width=30)  # Fixed width
Vt_entry.grid(row=2, column=1, padx=10, pady=10)
Vt_entry.insert(0, "0.025")

tk.Label(tab2, text="I_load:").grid(row=3, column=0, padx=10, pady=10)
I_load_entry = tk.Entry(tab2, width=30)  # Fixed width
I_load_entry.grid(row=3, column=1, padx=10, pady=10)
I_load_entry.insert(0, "0.005")

tk.Label(tab2, text="V_guess:").grid(row=4, column=0, padx=10, pady=10)
V_guess_entry = tk.Entry(tab2, width=30)  # Fixed width
V_guess_entry.grid(row=4, column=1, padx=10, pady=10)
V_guess_entry.insert(0, "3")

tk.Label(tab2, text="Method:").grid(row=5, column=0, padx=10, pady=10)
method_combobox_diode = ttk.Combobox(tab2, values=["Newton's Method", "Bisection Method", "False Position Method"], width=30)  # Fixed width
method_combobox_diode.grid(row=5, column=1, padx=10, pady=10)
method_combobox_diode.current(0)

calculate_button_diode = tk.Button(tab2, text="Calculate", command=calculate_diode, width=30, font=("Arial", 12))  # Fixed width and font size
calculate_button_diode.grid(row=6, columnspan=2, pady=20)

# Tab 3 - Rectangular Field Solver
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Rectangular Field Solver')

# Widgets
tk.Label(tab3, text="Perimeter:").grid(row=0, column=0, padx=10, pady=10)
perimeter_entry_rec = tk.Entry(tab3, width=30)  # Fixed width
perimeter_entry_rec.grid(row=0, column=1, padx=10, pady=10)
perimeter_entry_rec.insert(0, "140")

tk.Label(tab3, text="Area:").grid(row=1, column=0, padx=10, pady=10)
area_entry_rec = tk.Entry(tab3, width=30)  # Fixed width
area_entry_rec.grid(row=1, column=1, padx=10, pady=10)
area_entry_rec.insert(0, "1200")

tk.Label(tab3, text="Method:").grid(row=2, column=0, padx=10, pady=10)
method_combobox_rectangular = ttk.Combobox(tab3, values=["Newton's Method", "Bisection Method", "Secant Method"], width=30)  # Fixed width
method_combobox_rectangular.grid(row=2, column=1, padx=10, pady=10)
method_combobox_rectangular.current(0)

tk.Button(tab3, text="Calculate", command=calculate_rectangular_field, width=30, font=("Arial", 12)).grid(row=3, columnspan=2, pady=20)

# Packing the tab control
tab_control.grid(row=0, column=0, sticky="nsew")

# Make all tabs expandable
for tab in [tab1, tab2, tab3]:
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_columnconfigure(1, weight=1)
    tab.grid_rowconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)
    tab.grid_rowconfigure(2, weight=1)
    tab.grid_rowconfigure(3, weight=1)
    tab.grid_rowconfigure(4, weight=1)
    tab.grid_rowconfigure(5, weight=1)
    tab.grid_rowconfigure(6, weight=1)

root.mainloop()





