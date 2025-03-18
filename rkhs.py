import numpy as np
import fif
import scipy.integrate as integrate_using
import scipy.interpolate as interpolate_using

class FIF:
    """
    - Returns a function object F_i which is a fractal perturbation of u_i
    - based on the provided vertical scaling factors s_j for all sub intervals I_j
    """
    def __init__(self, i:int, x_grid: np.ndarray, f_values:np.ndarray, s_values: list, depth=2):
        self.i = i
        self.x_grid = x_grid
        self.f_values = f_values
        
        self.u_i = fif.u(self.i)
        self.r_i = fif.r(self.i)

        self.s_values = s_values
        self.depth = depth

        self._buildFIF_over_grid()
    
    def _buildFIF_over_grid(self):
        self.fractal_points = fif.compute_F_ui(u_i=self.u_i, r_i=self.r_i,
                                          s_values=self.s_values,
                                          x_grid=self.x_grid, data_values=self.f_values,
                                          num_iter=self.depth)
    
    def __call__(self, x):
        return np.interp(x, self.fractal_points[:, 0], self.fractal_points[:, 1])

class RKHS:
    """
    - Builds the Fractal RKHS by developing the basis set of linearly independent FIFs over the x-domain.
    - Provide an inner product evaluated as integral of the two basis FIFs over x_domain.

    Parameters
    ----------
    `m`: `int`, default = 10
        Number of FIFs in the basis.
    
    `data_series`: `numpy.ndarray`
        - A two-dim numpy array of shape (n, 2) defining the given data.
        - n is the number of data points.
        - `data_series[:, 0]` defines the x-domain.
        - `data_series[:, 1]` defines the f-domain.
    
    `fractal_depth`: `int`, default = 2
        - Defines how many times to pass through the data during construction of a FIF.
        - The more `fractal_depth` is, the more finely FIF will be computed over the x-domain.
    
    `s_values`: `numpy.ndarray`, default = None
        - A (m, n-1) shaped numpy array containing vertical scaling factors.
        - `s_values[i, j]` gives the vertical scaling factor for i-th basis function over the sub-interval I_{j+1} = [x_j, x_{j+1}].
        - Initialised randomly if `None`.
    
    """
    def __init__(self, data_series:np.ndarray,m: int = 10,
                 fractal_depth = 2, s_values: np.ndarray = None):
        self.m = m
        self.x_grid = data_series[:, 0]
        self.f_grid = data_series[:, 1]
        self.N = self.x_grid.shape[0] - 1 # defining so that can say j = 0, 1, ..., N
        self.depth = fractal_depth

        np.random.seed(42)
        if s_values is None:
            self.s_values = np.random.uniform(low=-0.999999, high=0.999999, size=(self.m, self.N))
            print(f"Vertical scaling factors s_values were not provided, so they were initialised randomly.")
        else:
            assert (s_values.shape == (self.m, self.N)), "Sufficient number of scaling factors not provided."
            self.s_values = s_values
        
        self._build_basis()
        self._build_matrixA()
        print(f"Built matrix A = [<F_[u_i], F_[u_j]>].")
        self._build_matrixB()
        print(f"Built matrix B = inverse(A).")
        
    def _build_Fi(self, i):
        Fi = FIF(i=i, x_grid=self.x_grid, f_values=fif.u(i)(self.x_grid),
               s_values=list(self.s_values[i, :]), depth=self.depth)
        return Fi

    def _build_basis(self):
        print(f"Building the Basis functions for the RKHS:")
        self.fractal_basis = dict()
        
        for i in range(self.m):
            self.fractal_basis[f"{i+1}"] = self._build_Fi(i)
            print(f"\tBuilt F_[u_{i+1}].")
            if i == 0:
                self.refined_x_grid = self.fractal_basis[f"{i+1}"].fractal_points[:, 0]
            else:
                if len(self.fractal_basis[f"{i+1}"].fractal_points) < len(self.refined_x_grid):
                    self.refined_x_grid = self.fractal_basis[f"{i+1}"].fractal_points[:, 0]
        print(f"Established the refined x grid with {self.refined_x_grid.shape[0]} points.")
            
    
    def inner_product(self, Fi:FIF, Fj:FIF):
        fi_points = Fi(self.refined_x_grid)
        fj_points = Fj(self.refined_x_grid)
        return np.trapz(y = fi_points*fj_points, x = self.refined_x_grid)
        # return integrate_using.simpson(y=fi_points*fj_points,x=self.refined_x_grid)

        # integrand = lambda x: Fi(x)*Fj(x)
        # return integrate_using.quad(integrand, a=0, b=1)[0]

        # return integrate_using.simpson(Fi.fractal_points[:, 1] * Fj.fractal_points[:, 1], x = Fi.fractal_points[:, 0])
    
    def _build_matrixA(self):
        self.A = np.zeros(shape=(self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                self.A[i][j] = self.inner_product(Fi = self.fractal_basis[f"{i+1}"], Fj = self.fractal_basis[f"{j+1}"])
    
    def _build_matrixB(self):
        self.B = np.linalg.inv(self.A)
    
    def F(self, x:float):
        f_vec_at_x = np.zeros(shape=(self.m, 1))
        for i in range(self.m):
            f_vec_at_x[i] = self.fractal_basis[f"{i+1}"](x)
        
        return f_vec_at_x
    
    def kernel_function(self, x:float, x_:float):
        return np.squeeze(self.F(x_).T @ self.B @ self.F(x))[()]
    
    def kernel_array(self, x: np.ndarray, x_:np.ndarray):
        assert(x.shape == x_.shape), "shape of x and x_ must be same."
        kf = np.vectorize(self.kernel_function)
        return kf(x, x_)
