import numpy as np

class u:
    """
    Function object for u_i(x) = cos(i*pi*x).
    """
    def __init__(self, i: int = 10):
        self.i = i
    
    def __call__(self, x):
        return np.cos(self.i * np.pi * x)

class r:
    """
    Function object for r_i(x) = x*(u_i(1) - u_i(0)) + u_i(0).
    """
    def __init__(self, i: int = 10):
        self.i = i
    
    def __call__(self, x):
        u_i_obj = u(self.i)
        return x * (u_i_obj(1) - u_i_obj(0)) + u_i_obj(0)

class L:
    """
    Affine map L_j : [0,1] -> [x_{j-1}, x_j].
    """
    def __init__(self, j: int, x_left, x_right):
        assert j >= 1, "j must be at least 1."
        self.x_left = x_left
        self.x_right = x_right
        self.j = j
    
    def __call__(self, x):
        # Works for scalars or numpy arrays.
        return (self.x_right - self.x_left) * x + self.x_left

class M:
    """
    Map M_j: [0,1] x R -> R defined by
      M_j(x, z) = s_j*(z - r_i(x)) + u_i(L_j(x)).
    """
    def __init__(self, j: int, s_j: float,x_left, x_right, u_i, r_i):
        assert j >= 1, "j must be at least 1."
        assert -1 < s_j < 1, "s_j must be in (-1, 1)."
        self.j = j
        self.x_left = x_left
        self.x_right = x_right
        self.s_j = s_j
        self.u_i = u_i
        self.r_i = r_i
        self.L_j = L(j, self.x_left, self.x_right)
    
    def __call__(self, x, z):
        return self.s_j * (z - self.r_i(x)) + self.u_i(self.L_j(x))

class W:
    """
    Mapping W_j: [0, 1] x R -> [x_{j-1}, x_j] x R defined by
      W_j(x, z) = (L_j(x), M_j(x, z)),
    where L_j and M_j are the maps for the j-th subinterval.
    This class is vectorized and accepts an array of points.
    """
    def __init__(self, j: int, s_j: float,x_left, x_right, u_i, r_i):
        self.j = j
        self.x_left = x_left
        self.x_right = x_right
        self.s_j = s_j
        self.u_i = u_i
        self.r_i = r_i
        self.L_j = L(j, self.x_left, self.x_right)
        self.M_j = M(j, s_j,self.x_left, self.x_right, u_i, r_i)
    
    def __call__(self, points: np.ndarray):
        """
        Apply W_j to an array of points.
        Each row in 'points' is a point [x, z].
        """
        x_vals = points[:, 0]
        z_vals = points[:, 1]
        new_x = self.L_j(x_vals)       # Vectorized application of L_j.
        new_z = self.M_j(x_vals, z_vals)
        return np.column_stack((new_x, new_z))

def compute_F_ui(u_i, r_i, s_values, x_grid, data_values, num_iter=2):
    """
    Computes a pointwise approximation to the graph of the fractal interpolation function
    using an iterated function system (IFS) in a vectorized manner.
    
    Parameters:
      u_i         : Approximant function object.
      r_i         : Corresponding r function object.
      s_values    : List/array of scaling factors s_j for each subinterval.
      x_grid      : The original grid (interpolation nodes).
      data_values : Data values at the nodes.
      num_iter    : Number of IFS iterations.
      
    Returns:
      S_arr : A NumPy array of shape (n_points, 2) approximating the fractal graph.
    """
    # Initialize with the given interpolation points.
    S_arr = np.column_stack((x_grid, data_values))
    N = len(x_grid) - 1  # Number of subintervals.
    
    for iteration in range(num_iter):
        new_points_list = []
        # For each subinterval, apply the corresponding W mapping.
        for j in range(1, N + 1):
            s_j = s_values[j - 1]
            W_j_obj = W(j, s_j, x_grid[j-1], x_grid[j], u_i, r_i)
            new_pts = W_j_obj(S_arr)
            new_points_list.append(new_pts)
        # Combine new points with existing ones.
        all_points = np.vstack([S_arr] + new_points_list)
        # Remove duplicates by rounding and using np.unique.
        rounded = np.round(all_points, decimals=10)
        S_arr = np.unique(rounded, axis=0)
    # Sort by the x-coordinate.
    S_arr = S_arr[np.argsort(S_arr[:, 0])]
    return S_arr