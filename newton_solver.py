import numpy as np
from sympy import symbols, Matrix, diff, lambdify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def newton_method_with_hessian(func_str, variables, initial_guess, tolerance=1e-7, max_iterations=100):
    """
    Newton's Method with Hessian for solving systems of equations.

    Args:
        func_str (str): The function as a string.
        variables (list): List of variable names (e.g., ["x", "y"]).
        initial_guess (list): Initial guess for the variables.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        dict: Contains solution, iterations, and status.
    """
    logger.info("Starting Newton's Method with Hessian")
    logger.info(f"Function: {func_str}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Initial Guess: {initial_guess}")
    logger.info(f"Tolerance: {tolerance}, Max Iterations: {max_iterations}")

    if len(variables) > 3:
        raise ValueError("This implementation supports up to 3 variables.")

    # Define symbols and function
    vars_symbols = symbols(variables)
    f = Matrix([func_str]) if isinstance(func_str, str) else Matrix(func_str)

    # Compute gradient and Hessian
    gradient = f.jacobian(vars_symbols).transpose()
    hessian = gradient.jacobian(vars_symbols)

    # Convert to Python-callable functions
    f_callable = lambdify(vars_symbols, f, "numpy")
    grad_callable = lambdify(vars_symbols, gradient, "numpy")
    hess_callable = lambdify(vars_symbols, hessian, "numpy")

    x = np.array(initial_guess, dtype=float)
    iterations = []

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}: Current guess {x}")

        # Evaluate function, gradient, and Hessian
        fx = np.array(f_callable(*x), dtype=float).flatten()
        grad = np.array(grad_callable(*x), dtype=float).flatten()
        hess = np.array(hess_callable(*x), dtype=float)

        logger.info(f"f(x): {fx}, Gradient: {grad}")
        logger.debug(f"Hessian matrix:\n{hess}")

        # Check convergence
        if np.linalg.norm(grad) < tolerance:
            logger.info(f"Converged to solution: {x} at iteration {iteration + 1}")
            return {
                "status": "success",
                "solution": x.tolist(),
                "iterations": iterations
            }

        # Check Hessian invertibility
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            error_msg = "Hessian is singular or nearly singular. Cannot proceed."
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "iterations": iterations
            }

        # Newton step
        delta_x = -hess_inv @ grad
        x_new = x + delta_x

        # Log iteration details
        iterations.append({
            "iteration": iteration + 1,
            "x": x.tolist(),
            "f(x)": fx.tolist(),
            "gradient": grad.tolist(),
            "hessian": hess.tolist(),
            "delta_x": delta_x.tolist(),
            "x_new": x_new.tolist()
        })

        # Check if the step size is below tolerance
        if np.linalg.norm(delta_x) < tolerance:
            logger.info(f"Converged to solution: {x_new} at iteration {iteration + 1}")
            return {
                "status": "success",
                "solution": x_new.tolist(),
                "iterations": iterations
            }

        x = x_new

    logger.warning("Maximum iterations reached without convergence.")
    return {
        "status": "error",
        "message": "Maximum iterations reached without convergence.",
        "iterations": iterations
    }
