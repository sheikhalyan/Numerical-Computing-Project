import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt

# Function definitions for numerical methods
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
    plt.title(f'Convergence of {method}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot exact vs approximate solutions
def plot_exact_vs_approx(method, x_vals, exact_value):
    plt.figure()
    plt.plot(range(1, len(x_vals)+1), x_vals, label=f'Approximate ({method})', marker='o')
    plt.axhline(y=exact_value, color='r', linestyle='-', label='Exact Solution')
    plt.xlabel('Iteration')
    plt.ylabel('Solution')
    plt.title(f'Exact vs Approximate Solution ({method})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Logistic Growth Model
def logistic_growth(P, r, K):
    return r * P * (1 - P / K)

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
    high_precision_root, _, _ = newton_method(f_logistic, dfdx_logistic, P0, tol=1e-12)

    if method == "Newton's Method":
        root, iterations, x_vals = newton_method(f_logistic, dfdx_logistic, P0)
    elif method == "Secant Method":
        root, iterations, x_vals = secant_method(f_logistic, P0, P0 + 1)
    elif method == "Bisection Method":
        root, iterations, x_vals = bisection_method(f_logistic, 0.0, K)

    errors = [abs(f_logistic(x)) for x in x_vals]

    details_window = tk.Toplevel(main_window)
    details_window.title("Iteration Details")

    tree = ttk.Treeview(details_window, columns=("Iteration", "Approximate Solution", "Exact Solution", "Error"), show="headings")
    tree.heading("Iteration", text="Iteration")
    tree.heading("Approximate Solution", text="Approximate Solution")
    tree.heading("Exact Solution", text="Exact Solution")
    tree.heading("Error", text="Error")

    for i in range(len(x_vals)):
        exact_val = high_precision_root
        error = abs(x_vals[i] - exact_val)
        tree.insert("", "end", values=(i + 1, x_vals[i], exact_val, error))

    tree.pack(fill="both", expand=True)
    plot_errors(method, errors)
    plot_exact_vs_approx(method, x_vals, high_precision_root)

# Diode Equation Model
def diode_eq(V, Is, n, Vt, I):
    return Is * (np.exp(V / (n * Vt)) - 1) - I

def dfdx_diode_eq(V, Is, n, Vt):
    return (Is / (n * Vt)) * np.exp(V / (n * Vt))

def calculate_diode():
    I = float(I_entry.get())
    Is = float(Is_entry.get())
    n = float(n_entry.get())
    Vt = float(Vt_entry.get())
    V0 = float(V0_entry.get())

    def f_diode(V):
        return diode_eq(V, Is, n, Vt, I)

    def dfdx_diode(V):
        return dfdx_diode_eq(V, Is, n, Vt)

    method = method_combobox_diode.get()

    if method == "Newton's Method":
        root, iterations, x_vals = newton_method(f_diode, dfdx_diode, V0)
    elif method == "Secant Method":
        root, iterations, x_vals = secant_method(f_diode, V0, V0 + 0.1)
    elif method == "Bisection Method":
        root, iterations, x_vals = bisection_method(f_diode, 0.0, 5.0)

    errors = [abs(f_diode(x)) for x in x_vals]

    details_window = tk.Toplevel(main_window)
    details_window.title("Iteration Details - Diode Equation")

    tree = ttk.Treeview(details_window, columns=("Iteration", "Approximate Solution", "Error"), show="headings")
    tree.heading("Iteration", text="Iteration")
    tree.heading("Approximate Solution", text="Approximate Solution")
    tree.heading("Error", text="Error")

    for i in range(len(x_vals)):
        error = abs(x_vals[i] - root)
        tree.insert("", "end", values=(i + 1, x_vals[i], error))

    tree.pack(fill="both", expand=True)
    plot_errors(method, errors)
    plot_exact_vs_approx(method, x_vals, root)

# Rectangular Field Model
def rectangular_field(L, P, A):
    return 2 * L + 2 * (A / L) - P

def dfdx_rectangular_field(L, A):
    return 2 - 2 * (A / L ** 2)

def calculate_field():
    P = float(P_entry.get())
    A = float(A_entry.get())
    L0 = float(L0_entry.get())

    def f_field(L):
        return rectangular_field(L, P, A)

    def dfdx_field(L):
        return dfdx_rectangular_field(L, A)

    method = method_combobox_field.get()

    if method == "Newton's Method":
        root, iterations, x_vals = newton_method(f_field, dfdx_field, L0)
    elif method == "Secant Method":
        root, iterations, x_vals = secant_method(f_field, L0, L0 + 0.1)
    elif method == "Bisection Method":
        root, iterations, x_vals = bisection_method(f_field, 0.1, P / 2)

    errors = [abs(f_field(x)) for x in x_vals]

    details_window = tk.Toplevel(main_window)
    details_window.title("Iteration Details - Rectangular Field")

    tree = ttk.Treeview(details_window, columns=("Iteration", "Approximate Solution", "Error"), show="headings")
    tree.heading("Iteration", text="Iteration")
    tree.heading("Approximate Solution", text="Approximate Solution")
    tree.heading("Error", text="Error")

    for i in range(len(x_vals)):
        error = abs(x_vals[i] - root)
        tree.insert("", "end", values=(i + 1, x_vals[i], error))

    tree.pack(fill="both", expand=True)
    plot_errors(method, errors)
    plot_exact_vs_approx(method, x_vals, root)

# Main window
main_window = tk.Tk()
main_window.title("Numerical Methods for Various Equations")

# Notebook for tabs
notebook = ttk.Notebook(main_window)
notebook.pack(pady=10, expand=True)

# Logistic Growth Tab
logistic_frame = ttk.Frame(notebook, width=400, height=400)
logistic_frame.pack(fill="both", expand=True)
notebook.add(logistic_frame, text="Logistic Growth")

ttk.Label(logistic_frame, text="Target Population:").grid(column=0, row=0, padx=10, pady=5)
target_population_entry = ttk.Entry(logistic_frame)
target_population_entry.grid(column=1, row=0, padx=10, pady=5)
target_population_entry.insert(0, "1.25")

ttk.Label(logistic_frame, text="Initial Population (P0):").grid(column=0, row=1, padx=10, pady=5)
P0_entry = ttk.Entry(logistic_frame)
P0_entry.grid(column=1, row=1, padx=10, pady=5)
P0_entry.insert(0, "0.1")

ttk.Label(logistic_frame, text="Growth Rate (r):").grid(column=0, row=2, padx=10, pady=5)
r_entry = ttk.Entry(logistic_frame)
r_entry.grid(column=1, row=2, padx=10, pady=5)
r_entry.insert(0, "1.0")

ttk.Label(logistic_frame, text="Carrying Capacity (K):").grid(column=0, row=3, padx=10, pady=5)
K_entry = ttk.Entry(logistic_frame)
K_entry.grid(column=1, row=3, padx=10, pady=5)
K_entry.insert(0, "10.0")

ttk.Label(logistic_frame, text="Method:").grid(column=0, row=4, padx=10, pady=5)
method_combobox_logistic = ttk.Combobox(logistic_frame, values=["Newton's Method", "Secant Method", "Bisection Method"])
method_combobox_logistic.grid(column=1, row=4, padx=10, pady=5)
method_combobox_logistic.current(0)

calculate_button_logistic = ttk.Button(logistic_frame, text="Calculate", command=calculate_logistic)
calculate_button_logistic.grid(column=0, row=5, columnspan=2, pady=10)

# Diode Equation Tab
diode_frame = ttk.Frame(notebook, width=400, height=400)
diode_frame.pack(fill="both", expand=True)
notebook.add(diode_frame, text="Diode Equation")

ttk.Label(diode_frame, text="Current (I):").grid(column=0, row=0, padx=10, pady=5)
I_entry = ttk.Entry(diode_frame)
I_entry.grid(column=1, row=0, padx=10, pady=5)
I_entry.insert(0, "0.01")

ttk.Label(diode_frame, text="Saturation Current (Is):").grid(column=0, row=1, padx=10, pady=5)
Is_entry = ttk.Entry(diode_frame)
Is_entry.grid(column=1, row=1, padx=10, pady=5)
Is_entry.insert(0, "1e-12")

ttk.Label(diode_frame, text="Emission Coefficient (n):").grid(column=0, row=2, padx=10, pady=5)
n_entry = ttk.Entry(diode_frame)
n_entry.grid(column=1, row=2, padx=10, pady=5)
n_entry.insert(0, "1.0")

ttk.Label(diode_frame, text="Thermal Voltage (Vt):").grid(column=0, row=3, padx=10, pady=5)
Vt_entry = ttk.Entry(diode_frame)
Vt_entry.grid(column=1, row=3, padx=10, pady=5)
Vt_entry.insert(0, "0.025")

ttk.Label(diode_frame, text="Initial Guess (V0):").grid(column=0, row=4, padx=10, pady=5)
V0_entry = ttk.Entry(diode_frame)
V0_entry.grid(column=1, row=4, padx=10, pady=5)
V0_entry.insert(0, "0.1")

ttk.Label(diode_frame, text="Method:").grid(column=0, row=5, padx=10, pady=5)
method_combobox_diode = ttk.Combobox(diode_frame, values=["Newton's Method", "Secant Method", "Bisection Method"])
method_combobox_diode.grid(column=1, row=5, padx=10, pady=5)
method_combobox_diode.current(0)

calculate_button_diode = ttk.Button(diode_frame, text="Calculate", command=calculate_diode)
calculate_button_diode.grid(column=0, row=6, columnspan=2, pady=10)

# Rectangular Field Tab
field_frame = ttk.Frame(notebook, width=400, height=400)
field_frame.pack(fill="both", expand=True)
notebook.add(field_frame, text="Rectangular Field")

ttk.Label(field_frame, text="Perimeter (P):").grid(column=0, row=0, padx=10, pady=5)
P_entry = ttk.Entry(field_frame)
P_entry.grid(column=1, row=0, padx=10, pady=5)
P_entry.insert(0, "100.0")

ttk.Label(field_frame, text="Area (A):").grid(column=0, row=1, padx=10, pady=5)
A_entry = ttk.Entry(field_frame)
A_entry.grid(column=1, row=1, padx=10, pady=5)
A_entry.insert(0, "500.0")

ttk.Label(field_frame, text="Initial Guess (L0):").grid(column=0, row=2, padx=10, pady=5)
L0_entry = ttk.Entry(field_frame)
L0_entry.grid(column=1, row=2, padx=10, pady=5)
L0_entry.insert(0, "10.0")

ttk.Label(field_frame, text="Method:").grid(column=0, row=3, padx=10, pady=5)
method_combobox_field = ttk.Combobox(field_frame, values=["Newton's Method", "Secant Method", "Bisection Method"])
method_combobox_field.grid(column=1, row=3, padx=10, pady=5)
method_combobox_field.current(0)

calculate_button_field = ttk.Button(field_frame, text="Calculate", command=calculate_field)
calculate_button_field.grid(column=0, row=4, columnspan=2, pady=10)

main_window.mainloop()
