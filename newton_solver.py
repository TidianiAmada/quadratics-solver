import logging
from sympy import symbols, sympify, diff, lambdify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def newton_method_multi_variable(func_str, variables, initial_guesses, tolerance=1e-7, max_iterations=100):
    """
    Newton's Method for functions with 1-3 variables with logging.

    Args:
        func_str (str): The function as a string (e.g., "x**2 + y**2 - 4").
        variables (list): List of variable names (e.g., ["x", "y"]).
        initial_guesses (list): List of initial guesses for each variable.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations allowed.

    Returns:
        dict: Contains the final solution, iteration details, and status message.
    """
    logger.info("Starting Newton's Method")
    logger.info(f"Function: {func_str}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Initial Guesses: {initial_guesses}")
    logger.info(f"Tolerance: {tolerance}, Max Iterations: {max_iterations}")

    if len(variables) > 3:
        error_msg = "This implementation supports up to 3 variables only."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Try to define symbols and parse the function
    try:
        vars_symbols = symbols(variables)
        f = sympify(func_str)
        logger.info("Successfully parsed the function and symbols.")
    except Exception as e:
        error_msg = f"Invalid function or variables: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

    # Compute derivatives
    try:
        partial_derivatives = [diff(f, var) for var in vars_symbols]
        logger.info("Computed partial derivatives.")
    except Exception as e:
        error_msg = f"Error computing partial derivatives: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

    # Convert to Python-callable functions
    try:
        f_callable = lambdify(vars_symbols, f, "numpy")
        partials_callable = [lambdify(vars_symbols, partial, "numpy") for partial in partial_derivatives]
        logger.info("Successfully converted the function and derivatives to callable forms.")
    except Exception as e:
        error_msg = f"Error creating callable functions: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

    # Initialize variables
    x = initial_guesses
    iterations = []

    for i in range(max_iterations):
        try:
            logger.info(f"Starting iteration {i + 1}. Current guess: {x}")
            # Evaluate function and partial derivatives
            fx = f_callable(*x)
            grad = [partial(*x) for partial in partials_callable]
            logger.info(f"f(x): {fx}, Gradient: {grad}")

            # Check if the gradient is zero or near-zero
            if all(abs(g) < 1e-12 for g in grad):
                error_msg = "Gradient is zero or close to zero. No solution found."
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "iterations": iterations,
                }

            # Newton step
            x_new = [xi - fx / g if abs(g) > 1e-12 else xi for xi, g in zip(x, grad)]
            logger.info(f"New guess: {x_new}")

            # Log iteration details
            iterations.append({
                "iteration": i + 1,
                "x": x.copy(),
                "f(x)": fx,
                "gradient": grad.copy(),
                "x_new": x_new.copy(),
            })

            # Check for convergence
            if all(abs(xn - xi) < tolerance for xn, xi in zip(x_new, x)):
                logger.info(f"Convergence reached at iteration {i + 1}. Solution: {x_new}")
                return {
                    "status": "success",
                    "solution": x_new,
                    "iterations": iterations,
                }

            x = x_new
        except Exception as e:
            logger.exception(f"Exception during iteration {i + 1}: {e}")
            return {
                "status": "error",
                "message": f"Error during iteration {i + 1}: {e}",
                "iterations": iterations,
            }

    logger.warning("Exceeded maximum iterations. No solution found.")
    return {
        "status": "error",
        "message": "Exceeded maximum iterations. No solution found.",
        "iterations": iterations,
    }
