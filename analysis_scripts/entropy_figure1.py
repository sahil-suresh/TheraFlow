import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

# Set the path to the folder containing your data files
downloads_folder = "C:\\Users\\sahil\\Documents\\Datathon"

# Function to calculate Shannon entropy
def shannon_entropy(data):
    counts = Counter(data)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)
    return entropy

# Dictionary to store the slope values at specific percentage points for each file
entropy_slope_data = {}

# Iterate over each file in the folder that starts with "CASE" and ends with ".csv"
for filename in os.listdir(downloads_folder):
    if filename.startswith("CASE") and filename.endswith(".csv"):
        file_path = os.path.join(downloads_folder, filename)
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Extract the first item in the list from every row of the Diagnoses column
        first_diagnoses = []
        for diagnosis in df['Diagnoses']:
            if pd.notna(diagnosis):  # Check if diagnosis is not NaN
                first_item = diagnosis.split(',')[0].strip()  # Take the first item and strip any whitespace
                first_diagnoses.append(first_item)

        # Calculate cumulative entropy over the rows
        cumulative_entropies = [shannon_entropy(first_diagnoses[:i]) for i in range(1, len(first_diagnoses) + 1)]
        
        # Calculate the slope (discrete derivative) of cumulative entropy
        entropy_slope = np.diff(cumulative_entropies)

        # Determine the row indices at specified percentage points
        num_rows = len(entropy_slope)
        if num_rows >= 4:
            indices = [0, int(0.25 * num_rows), int(0.5 * num_rows), int(0.75 * num_rows), num_rows - 1]

            # Extract entropy slope values at specified indices
            selected_slopes = [entropy_slope[i] for i in indices]
            entropy_slope_data[filename] = selected_slopes
        else:
            print(f"Not enough data points in {filename} for calculation.")

# Check if we have data
if not entropy_slope_data:
    print("No data to process. Exiting.")
    exit()

# Prepare data for fitting
percentages = np.array([0, 25, 50, 75, 100])
all_slopes = np.array([entropy_slope_data[filename] for filename in entropy_slope_data.keys()])

# Calculate the mean slope at each percentage point
mean_slopes = np.mean(all_slopes, axis=0)

print(f"percentages shape: {percentages.shape}")
print(f"mean_slopes shape: {mean_slopes.shape}")

# Define your models
def linear_model(x, m, c):
    return m * x + c

def exponential_decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

def logarithmic_model(x, a, b, c):
    return a * np.log(np.maximum(x + b, 1e-8)) + c  # Ensure positive argument for log

def hyperbolic_decay_model(x, a, b, c):
    return a / np.maximum(x + b, 1e-8) + c  # Avoid division by zero

def power_law_decay_model(x, a, k, b, c):
    return a * np.power(np.maximum(x + b, 1e-8), -k) + c

def logistic_model(x, L, k, x0, c):
    return L / (1 + np.exp(-k * (x - x0))) + c

# Initialize a dictionary to store models and their parameters
models = {
    'Linear': {
        'func': linear_model,
        'equation': 'y = m * x + c',
        'params': ['m', 'c'],
        'initial_guess': [1, 0],
        'bounds': (-np.inf, np.inf)
    },
    'Exponential Decay': {
        'func': exponential_decay_model,
        'equation': 'y = a * exp(-b * x) + c',
        'params': ['a', 'b', 'c'],
        'initial_guess': [1, 0.01, 0],
        'bounds': (-np.inf, np.inf)
    },
    'Logarithmic': {
        'func': logarithmic_model,
        'equation': 'y = a * ln(x + b) + c',
        'params': ['a', 'b', 'c'],
        'initial_guess': [1, 1, 0],
        'bounds': ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
    },
    'Hyperbolic Decay': {
        'func': hyperbolic_decay_model,
        'equation': 'y = a / (x + b) + c',
        'params': ['a', 'b', 'c'],
        'initial_guess': [1, 1, 0],
        'bounds': ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
    },
    'Power Law Decay': {
        'func': power_law_decay_model,
        'equation': 'y = a * (x + b)^(-k) + c',
        'params': ['a', 'k', 'b', 'c'],
        'initial_guess': [1, 1, 1, 0],
        'bounds': ([-np.inf, 0, 1e-8, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    },
    'Logistic': {
        'func': logistic_model,
        'equation': 'y = L / (1 + exp(-k * (x - x0))) + c',
        'params': ['L', 'k', 'x0', 'c'],
        'initial_guess': [max(mean_slopes), 0.01, 50, min(mean_slopes)],
        'bounds': ([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    }
}

r_squared_values = {}

# Fit each model and calculate R-squared
for name in models.keys():
    model = models[name]
    model_func = model['func']
    initial_guess = model['initial_guess']
    bounds = model['bounds']

    try:
        # Fit the model
        popt, _ = curve_fit(model_func, percentages, mean_slopes, p0=initial_guess, bounds=bounds, maxfev=10000)

        y_pred = model_func(percentages, *popt)
        residuals = mean_slopes - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mean_slopes - np.mean(mean_slopes))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values[name] = r_squared
        models[name]['popt'] = popt
        models[name]['r_squared'] = r_squared

        # Print out the equation with fitted parameters
        equation = model['equation']
        params = model['params']
        param_values = popt
        param_str = ', '.join(f"{param}={value:.4f}" for param, value in zip(params, param_values))
        print(f"\n{name} Model Equation: {equation}")
        print(f"Fitted Parameters: {param_str}")
        print(f"R-squared: {r_squared:.4f}")

    except RuntimeError:
        print(f"RuntimeError fitting model {name}.")
        r_squared_values[name] = -np.inf
    except Exception as e:
        print(f"Error fitting model {name}: {e}")
        r_squared_values[name] = -np.inf

# Plotting
for name in models.keys():
    model = models[name]
    if 'popt' not in model:
        continue  # Skip models that failed to fit

    model_func = model['func']
    popt = model['popt']
    r_squared = model['r_squared']
    equation = model['equation']
    params = model['params']
    param_values = popt
    param_str = ', '.join(f"{param}={value:.4f}" for param, value in zip(params, param_values))

    plt.figure(figsize=(12, 8))

    # Plot individual entropy slope values for each file in gray
    for filename, slopes in entropy_slope_data.items():
        plt.plot(percentages, slopes, marker='o', color='gray', linestyle='-', alpha=0.3)


    # Plot the model fit
    x_fit = np.linspace(0, 100, 500)
    y_fit = model_func(x_fit, *popt)
    label = f"{name} Fit"
    color = "#3eb265"
    plt.plot(x_fit, y_fit, color=color, linestyle='-', label=label)

    # Add R-squared and equation as text on the plot
    textstr = f"Equation: {equation}\n{param_str}\n$R^2$ = {r_squared:.3f}"
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Labels and title
    plt.xlabel("Percentage of Conversation")
    plt.ylabel("ΔCumulative Shannon Entropy")
    plt.title(f"ΔCumulative Shannon Entropy of Predicted Diagnosis - {name} Model")
    plt.legend(loc="upper right")

    plt.show()
