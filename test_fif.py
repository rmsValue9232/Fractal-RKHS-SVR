import numpy as np
import matplotlib.pyplot as plt
from fif import *

# Example usage:
if __name__ == "__main__":
    # The original grid of interpolation nodes.
    x_grid = np.linspace(0, 1, 11) 
    i_val = 3
    u_i_obj = u(i_val)
    r_i_obj = r(i_val)
    
    # Simulate actual data at the nodes (e.g., measured data) with a perturbation.
    data_values = [u_i_obj(x) + 0.2 * np.cos(5 * np.pi * x) for x in x_grid]
    
    # Define scaling factors for each subinterval.
    s_values = [0.1 if i%2!=0 else 0.5 for i in range((len(x_grid) - 1))]
    
    # Compute the fractal graph using the vectorized IFS approach.
    fractal_points = compute_F_ui(u_i_obj, r_i_obj, s_values, x_grid, data_values, num_iter=3)
    
    # Plot the resulting fractal interpolation function.
    plt.figure(figsize=(8, 4))
    plt.plot(fractal_points[:, 0], fractal_points[:, 1], 'b-', linewidth=0.5, label=f"Fractal F[u_{i_val+1}](x)")
    plt.scatter(x_grid, data_values, color='red', label="Data Points", zorder=5, marker="x")
    plt.xlabel("x")
    plt.ylabel(f"F[u_{i_val+1}](x)")
    plt.title("Fractal Interpolation Function via Vectorized IFS")
    plt.legend()
    plt.show()
