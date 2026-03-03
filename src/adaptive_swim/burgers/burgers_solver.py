from dataclasses import dataclass
import numpy as np
from .domain import Domain
from adaptive_swim.ansatz import BasicAnsatz as Ansatz
from typing import Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
from dataclasses import field
import copy
from scipy.stats import rv_discrete
import time 
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

@dataclass
class BurgersSolver:
    """
    Solver for the Burgers equation
    ∂u(x,t)/∂t + u(x,t) * ∂u(x,t)/∂x =  nu * ∂^2u(x,t)/∂^2x + f(x, t)

    with initial condition for u(t,x)

    Attributes:
    -----------
    domain: Domain
    ansatz: Ansatz
        basis functions from which the solution will be built by linear combination
        (use BoundaryCompliantAnsatz for this solver to ensure the boundary conditions are fulfilled)
    u0: Callable
        solution at time t0
    boundary_condition: str 
        boundary condition, one of "zero neumann"/"zero derivative" or "zero dirichlet"/"zero"
    forcing: Callable  
        forcing, a function of x and t
    nu: float = 1 
        diffusivity, constant
    regularization_scale: float
        regularization scale for computing the matrix inverse and solving least squares roblems
    ode_solver: str
        ode solver (to be used as 'method' in scipy.integrate.solve_ivp)

    """
    domain: Domain
    ansatz: Ansatz
    u0: Callable
    boundary_condition: str 
    forcing: Callable
    c: float = 1
    regularization_scale: float = 1e-8
    ode_solver: str = 'DOP853' #'DOP853' #'RK45'
    ansatz_collection: list = field(default_factory=list)
    c_collection: list = field(default_factory=list)
    svd_collecion: list = field(default_factory=list)
    scale_boundary_correction: float = 100

    def __post_init__(self):
        # initialize internal parameters
        # ode solution for the time-dependent coefficients
        self._coefficients_c: Callable = None # time dependent coefficients, solution from solver

        # matrices for the ODE
        self._B: np.ndarray[np.float64] = None
        self._A: np.ndarray[np.float64] = None
        self._A_inv: np.ndarray[np.float64] = None
        self._C: np.ndarray[np.float64] = None
        self._V_a: np.ndarray[np.float64] = None
       
    def evaluate(self, x_eval, t_eval, svd_on = True):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_burger: (n, t)
        '''
        
                       
        sol_c = self._coefficients_c(t_eval.reshape((np.shape(t_eval)[0], ))).T
        #sol_c = self._coefficients_c(t_eval).T
        if svd_on:
            sol_burger = self.ansatz.transform(x_eval) @ (self._V_a).T @ sol_c.T
        else: 
            sol_burger = self.ansatz.transform(x_eval) @ sol_c.T

        return sol_burger

    def evaluate_gradient(self, x_eval, t_eval, svd_on = True):
        '''
        Evaluate the gradient of the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_burger: (n, t)
        '''
        sol_c = self._coefficients_c(t_eval.reshape((np.shape(t_eval)[0], ))).T
        grad_feature_matrix = self.ansatz.transform(x_eval, operator="gradient")
        grad_feature_matrix = grad_feature_matrix.reshape((grad_feature_matrix.shape[0], grad_feature_matrix.shape[1]))
        if svd_on:
            gradient = grad_feature_matrix @ (self._V_a).T @ sol_c.T
        else:
            gradient = grad_feature_matrix @ sol_c.T

        return gradient


    def _init_matrices(self, svd_cutoff, outer_basis=True, svd_on = True):
        '''
        Set all matrices that occur in the ODE

        Parameters:
        rcond: regularization scale for inverse in gamma_A_inv
        '''
        self._A = self.ansatz.transform(self.domain.all_points)
        self._A_boundary = self.ansatz.transform(self.domain.boundary_points)
        self._B = (self.ansatz.transform(self.domain.all_points, operator="gradient"))
        self._C = (self.ansatz.transform(self.domain.all_points, operator="gradient"))
        if svd_on:
            U_a, S_a, V_a = np.linalg.svd(self._A, full_matrices=False)
            
            # Truncate singular values and corresponding columns of U and rows of Vt
            mask = S_a / np.max(S_a) > svd_cutoff
            S_a = S_a[mask]
            U_a = U_a[:, mask]
            V_a = V_a[mask, :]
            self._V_a = V_a
            self._A = self._A @ self._V_a.T
            self._A_boundary = self._A_boundary @ self._V_a.T

            self._B = (self._B).reshape((self._B.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1
        
            # CD: Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
            self._A_inv = np.linalg.pinv(self._A, svd_cutoff)
            self._A_boundary_inv = np.linalg.pinv(np.row_stack([self._A, self._A_boundary]), rcond = svd_cutoff)
        

            # Laplacian 
            self._C = self._C @ (self._V_a).T
        else:
            self._A_inv = np.linalg.pinv(self._A, svd_cutoff)
            self._A_boundary_inv = np.linalg.pinv(np.row_stack([self._A, self._A_boundary]), svd_cutoff)
        

        def ODE_coefficients(t, c, outer_basis=outer_basis, svd_on =svd_on):
            """
            The ODE to be solved for the time-dependent coefficients
            """
            f = self.forcing(self.domain.all_points,t)
            diff_term = (self.c * c @ self._C.T)
            t1 = (c @ self._A.T)
            t2 = (c @ self._B.T)
            non_lin_term =  t1 * t2
            #print("ode:",  np.shape(t1), np.shape(t2), np.max(t1), np.max(t2))
            rhs = (f + diff_term - non_lin_term).T
            if outer_basis:
                c_t = self._A_inv @ rhs # this is without adding zero boundary condition
            else:
            # The following is by adding the zero boundary condition.
                boundary_correction = -c @ self._A_boundary.T
                if svd_on:
                    c_t = self._A_boundary_inv @ np.concatenate([rhs, self.scale_boundary_correction * boundary_correction])
                else:
                    c_t = self._A_boundary_inv @ np.concatenate([rhs, self.scale_boundary_correction * boundary_correction.reshape(-1, 1)])
            
            return c_t.ravel()
        self.ODE_coefficients = ODE_coefficients

        return self
    
    def _get_c0(self, initial_sol=None, outer_basis=True):
        '''
        Initial condition of the time-dependent coefficients for the ODE solver
        initial_sol =  Solution at the end of the previous time-block evaluated at all domain points
        Returns:
            c0: shape ((k+1)*2, ); initial condition for c (first k+1 entries) and d (= c_t, last k+1 entries)
        '''
        if initial_sol is None:
            if outer_basis:
                c0 = np.linalg.lstsq(self._A, self.u0(self.domain.all_points), self.regularization_scale)[0]
            else:
                c0 = self._A_boundary_inv @ np.concatenate([self.u0(self.domain.all_points), self.u0(self.domain.boundary_points)])
        else:
            if outer_basis:
                c0 = np.linalg.lstsq(self._A, initial_sol, self.regularization_scale)[0]
            else:
                c0 = self._A_inv @ initial_sol
        return c0
        

    def fit(self, t_span, rtol=1e-8, atol=1e-8, svd_cutoff=None, time_blocks = 1):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 

        # compute the matrices needed in the ODE
        if svd_cutoff is None:
            svd_cutoff = self.regularization_scale * 10
        self._init_matrices(svd_cutoff=svd_cutoff)

        # get the initial value for the ODE
        c_0 = self._get_c0().reshape(-1)
        
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e10

        event_func.terminal = False

        # solve the ODE
        solver = solve_ivp(fun=self.ODE_coefficients, t_span=t_span, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
        self._coefficients_c = solver.sol 
        
        # Check if the integration was successful and the event was triggered
        if solver.status != 0:
            print("Integration failed or terminated due to exceeding the maximum absolute value.")

        return self

    def resample_data_points(self,gradient, p_distr_resampling, n_col = 1000):
        # We set the new interior and boundary points for the next time step by resampling
        rng = np.random.default_rng(seed=5) # Random seed
        
        # Compute probability distribution
        probabilities = p_distr_resampling(gradient).reshape(-1)
        cdf_values = cumtrapz(probabilities, self.domain.sample_points.reshape(-1,), initial=0)
        cdf_values /= cdf_values[-1]  # Normalize to make sure it goes from 0 to 1

        # Create an interpolation function for the inverse CDF
        inv_cdf_interp = interp1d(cdf_values, self.domain.sample_points.reshape(-1,), kind='quadratic', bounds_error=True)

        # Generate random samples from a uniform distribution
        uniform_samples = rng.random(n_col)

        # Use the inverse CDF interpolation function to map uniform samples to samples from the desired distribution
        x_collocation = inv_cdf_interp(uniform_samples)
        self.domain.interior_points = x_collocation.reshape((-1, 1))
        self.domain.set_all_points()


    def fit_time_blocks(self, t_span, rtol=1e-8, atol=1e-8, svd_cutoff=None, 
                        time_blocks = 1, prob_distr_resampling=None, n_col = 2000, 
                        outer_basis=True, svd_on = True, plots=False):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e10
        
        event_func.terminal = True # Terminate initial value problem if event_func is satisfied
        self.ansatz_collection = []
        self.c_collection = []
        self.svd_collecion = []
        
        # Uncomment the following 3 lines for plotting the collocation points after re-sampling
        self.collocation_point_collection = []
        self.gradient_collection_collocation = []
        self.gradient_collection_sample = []
        self.sol_collection = []
        
        t_block_size = (t_span[-1] - t_span[0])/time_blocks
        for i in range(time_blocks):
            # Solve the ODE for one time block
            t_block = [i * t_block_size, (i+1) * t_block_size]
            time_st = time.time()
            if i == 0:
                # set up the model for the ansatz function
                self.ansatz.init_model(self.domain, self.boundary_condition, initial_condition=None)
                self.ansatz_collection.append(copy.deepcopy(self.ansatz))

            else:
                # set up the model for the ansatz function: Pass previous solution as the target function
                self.resample_data_points(gradient=gradient, p_distr_resampling = prob_distr_resampling, n_col = n_col)

                sol_approx_all = self.evaluate(self.domain.all_points, t_block[0].reshape(-1,), svd_on = svd_on)
                sol_approx_interior = sol_approx_all[np.shape(self.domain.boundary_points)[0]:,:]

                if plots:
                    # Save collocation points (only for plotting): Can be deleted later!
                    self.collocation_point_collection.append(copy.deepcopy(self.domain.interior_points))
                    self.sol_collection.append(copy.deepcopy(sol_approx_interior))
            
                    # Delete the following 4 lines later
                    gradient_collocation = self.evaluate_gradient(self.domain.interior_points, t_block[0].reshape(-1,), svd_on = svd_on)
                    gradient_sample = self.evaluate_gradient(self.domain.sample_points, t_block[0].reshape(-1,), svd_on = svd_on)
                    
                    self.gradient_collection_collocation.append(copy.deepcopy(gradient_collocation))
                    self.gradient_collection_sample.append(copy.deepcopy(gradient_sample))

                """
                fig = plt.figure()
                plt.title("Solution evaluated at resampled points at the time t = " + str(t_block[0]))
                plt.scatter(self.domain.interior_points[:, 0], sol_all_points[2:, 0], label='u(t)')
                plt.legend()
                plt.show()
                """
                self.ansatz.init_model(self.domain, self.boundary_condition, initial_condition=sol_approx_interior) #, initial_condition=initial_sol
                self.ansatz_collection.append(copy.deepcopy(self.ansatz))
            
            time_end = time.time()

            # Compute the matrices needed in the ODE (using boundary + interior/collocation points)
            if svd_cutoff is None:
                svd_cutoff = self.regularization_scale * 10

            time_st = time.time()
            self._init_matrices(svd_cutoff=svd_cutoff, outer_basis=outer_basis, svd_on = svd_on)
            time_end = time.time()
            # print('Time (init_matrices) for block: ', i, ' = ', time_end - time_st)

            # Store the SVD: Required for evaluating later in the time-blocking approach
            if svd_on:
                self.svd_collecion.append(copy.deepcopy(self._V_a))
            
            # Initialize coeffcients for the (re)-sampled weights
            if i == 0:
                c_0 = self._get_c0(outer_basis=outer_basis).reshape(-1)
                self._coefficients_c = c_0
                # CD: Plot of initial condition
                """
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.set_title('Initialization at time t = ' + str(t_block[0]))
                #ax.plot(self.domain.all_points, initial_sol_new, label="solution") # Collocation + boundary
                ax.scatter(self.domain.all_points, self.ansatz.evaluate_model(self.domain.all_points) @ (self._V_a).T @ c_0.T , label="u(x,0)") # Collocation + boundary
                plt.legend()
                plt.show()
                """

                # Dims of c_0 = num_svd (A: (interior + boundary pts) * svd, u_0 = (int + boundary_pts) * 1)
            else:
                c_0 = self._get_c0(initial_sol=sol_approx_all, outer_basis=outer_basis).reshape(-1) # Boundary + collocation
                
                # Plot solution ta the time boundary after re-sampling
                """
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                self._coefficients_c = c_0
                ax.set_title('After re-sampling: Initial state for the new time block at t = ' + str(t_block[0]))
                ax.scatter(self.domain.all_points, self.ansatz.evaluate_model(self.domain.all_points) @ (self._V_a).T @ c_0.T , label="solution") # Collocation + boundary
                ax.plot(self.domain.sample_points, gradient, label="gradient")
                plt.legend()
                plt.show()
                """
                
            # solve the ODE
            time_st = time.time()
            solver = solve_ivp(fun=self.ODE_coefficients, t_span=t_block, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
            time_end = time.time()
            time_solver = time_end - time_st
            # print('Time (ODE solve) for time block ', i, ' = ', time_solver)
            self._coefficients_c = solver.sol
            #print("solver success: ", solver.success)
            
            # Store the interpolant to evaluate afterwards
            

            if solver.status == 0:
                #self._coefficients_c.append((copy.deepcopy(solver.sol)))
                self.c_collection.append(copy.deepcopy(self._coefficients_c))

            else:
                print("Integration failed or terminated due to exceeding the maximum absolute value.")
                for j in range(i, time_blocks):
                    #self._coefficients_c.append(lambda t : t * 100)
                    self.c_collection.append(lambda t : t * 100)
                break
            
            #self.c_collection.append(copy.deepcopy(self._coefficients_c))

            # Domain for evaluation:

            #self.domain.compute_all_points() # Set all_points = sample_pts + boundary points
            if i < time_blocks - 1:
                if solver.success:
                    gradient = np.abs(self.evaluate_gradient(self.domain.sample_points, t_block[1].reshape(-1,), svd_on = svd_on)) # Sample points
                    #self.gradient_collection.append(copy.deepcopy(gradient))
                else:
                    break
                # Following line is just for printing: remove it afterwards
                """
                initial_sol_new = self.evaluate(np.row_stack([self.domain.boundary_points, self.domain.sample_points]), t_block[1].reshape(-1,)) 
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.set_title('Gradient at the time t = ' + str(t_block[1]))
                #ax.plot(np.row_stack([self.domain.boundary_points, self.domain.sample_points]), initial_sol_new, label="solution")
                ax.plot(self.domain.sample_points, gradient, label="gradient")
                plt.legend()

                fig = plt.figure()
                #plt.semilogy(x, np.abs(f(x)), label='f(x)')
                plt.title("Immediately after solver: solution at the time t = " + str(t_block[1]))
                plt.scatter(np.row_stack([self.domain.boundary_points, self.domain.sample_points]), initial_sol_new[:, 0], label='u(t)')
                #plt.scatter(self.domain.all_points[:, 0], initial_sol_new[:, 0], label='u(t)')
                
                plt.legend()
                plt.show()
                """
        return self, solver.status
    
    
    def evaluate_blocks(self, x_eval, t_eval, time_blocks = 1, solver_status=0, svd_on = True):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_burger: (n, t)
        '''
        if solver_status == 0:
            t_block_size = (t_eval[-1] - t_eval[0])/time_blocks
            for i in range(time_blocks):
                if i < time_blocks - 1:
                    sol_c = self.c_collection[i](t_eval[(i*t_block_size <= t_eval) & (t_eval < (i+1)*t_block_size)]).T
                else:
                    sol_c = self.c_collection[i](t_eval[(i*t_block_size <= t_eval) & (t_eval <= (i+1)*t_block_size)]).T
                
                # Compute solution of Burgers equation using appropriate basis functions for the particular time-block
                if svd_on:
                    sol_burger_block = self.ansatz_collection[i].evaluate_model(x_eval) @ self.svd_collecion[i].T @ sol_c.T
                else:
                    sol_burger_block = self.ansatz_collection[i].evaluate_model(x_eval) @ sol_c.T

                if i == 0:
                    sol_burger = sol_burger_block
                else:
                    sol_burger = np.hstack((sol_burger, sol_burger_block))
        else:
            sol_burger = np.ones((np.shape(x_eval)[0], np.shape(t_eval)[0])) * 1000
        return sol_burger

