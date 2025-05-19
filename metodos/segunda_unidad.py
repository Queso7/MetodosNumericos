# segunda_unidad.py
from metodos.Polinomio import PolinomioMultivariable
import numpy as np
# math is not directly used in this module's functions but imported for eval_polinomio if needed
import math

def lagrange_interpolation(x_values, y_values):
    """
    Interpolación de Lagrange usando PolinomioMultivariable

    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y correspondientes

    Returns:
        PolinomioMultivariable: Polinomio de interpolación
    """
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")

    if n == 0:
        raise ValueError("Se requiere al menos un punto para la interpolación")

    # Check for duplicate x-values
    if len(x_values) != len(set(x_values)):
         # Find the duplicate value(s) for a more informative error
         seen = set()
         duplicates = {x for x in x_values if x in seen or seen.add(x)}
         raise ValueError(f"Los valores de x deben ser distintos para la interpolación de Lagrange. Valores duplicados encontrados: {list(duplicates)}")


    # Polinomio resultante (inicialmente cero)
    # Create a zero polynomial with 1 variable ('x')
    polinomio_final = PolinomioMultivariable({(0,): 0.0})


    for i in range(n):
        # Construir el polinomio base L_i(x)
        L_i = PolinomioMultivariable({(0,): 1.0}) # Start with a constant 1

        for j in range(n):
            if j != i:
                # Crear término (x - x_j)
                # Use PolinomioMultivariable to represent (x - x_j) assuming 'x' is the first variable
                term_poly = PolinomioMultivariable({(1,): 1.0, (0,): -x_values[j]}) # Represents 1*x^1 + (-x_j)*x^0

                # Calcular denominador (x_i - x_j)
                denom = x_values[i] - x_values[j]
                if abs(denom) < 1e-9: # Check if denominator is close to zero
                    # This case should ideally be caught by the duplicate x-value check earlier, but as a safeguard
                    raise ValueError(f"División por cero al calcular L_{i}. Los valores de x deben ser distintos. Problema en x[{i}] = x[{j}] = {x_values[i]}")

                # Multiply L_i by the term (x - x_j) and then by the scalar 1/denom
                L_i = L_i * term_poly * (1.0 / denom)

        # Sumar y_i * L_i(x) al polinomio final
        termino_final = L_i * y_values[i]
        polinomio_final = polinomio_final + termino_final

    # Simplify the final polynomial by combining like terms (already handled by agregar_termino)
    # and clean up very small coefficients if needed (handled by agregar_termino and __init__)

    return polinomio_final

def diferencias_divididas(x_values, y_values):
    """
    Interpolación por diferencias divididas (Newton)

    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y correspondientes

    Returns:
        tuple: (coeficientes, polinomio como string)
    """
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")

    if n == 0:
        raise ValueError("Se requiere al menos un punto para la interpolación")

    # Check for duplicate x-values
    if len(x_values) != len(set(x_values)):
         seen = set()
         duplicates = {x for x in x_values if x in seen or seen.add(x)}
         raise ValueError(f"Los valores de x deben ser distintos para las diferencias divididas. Valores duplicados encontrados: {list(duplicates)}")


    # Calcular tabla de diferencias divididas
    # We store the divided differences f[x_i, x_{i+1}, ..., x_{i+j}] at f[i][j]
    f = np.zeros((n, n))
    f[:,0] = y_values # First column is the y-values (0-th divided differences)

    for j in range(1, n): # Column index (order of difference)
        for i in range(n - j): # Row index (starting point x_i)
            numerator = f[i+1][j-1] - f[i][j-1]
            denominator = x_values[i+j] - x_values[i]
            if abs(denominator) < 1e-9:
                 # This case should be caught by the duplicate x-value check, but as a safeguard
                 raise ValueError(f"División por cero al calcular diferencias divididas. Problema con x[{i+j}] ({x_values[i+j]}) - x[{i}] ({x_values[i]}). Los valores de x deben ser distintos.")
            f[i][j] = numerator / denominator

    # The coefficients of the Newton polynomial are the top diagonal of the differences table
    # c_i = f[x_0, x_1, ..., x_i] which is at table[0][i]
    coeficientes = f[0,:]

    # Construir el polinomio de Newton usando PolinomioMultivariable for consistent string output
    # P(x) = c0 + c1*(x-x0) + c2*(x-x0)*(x-x1) + ... + cn*(x-x0)...(x-x_{n-1})
    # Start with the constant term c0
    polinomio_newton = PolinomioMultivariable({(0,): coeficientes[0]})

    for i in range(1, n): # For each term from c1 onwards (corresponding to coefficient coeficientes[i])
        # The term is coeficientes[i] * (x-x0) * (x-x1) * ... * (x-x_{i-1})
        term_poly = PolinomioMultivariable({(0,): coeficientes[i]}) # Start the term with the coefficient c_i

        # Multiply by (x - x_j) factors for j from 0 up to i-1
        for j in range(i):
             # Create the factor (x - x_j) as a PolinomioMultivariable
             # Assuming 'x' is the first variable
             factor_poly = PolinomioMultivariable({(1,): 1.0, (0,): -x_values[j]}) # Represents (1*x^1 + (-x_j)*x^0)
             term_poly = term_poly * factor_poly # Multiply the current term (initially just c_i) by (x - x_j)

        polinomio_newton = polinomio_newton + term_poly # Add the completed term to the total polynomial P(x)

    # Format coefficients for display (optional, keeping the numpy array is also fine)
    coeficientes_list = coeficientes.tolist() # Convert numpy array to list for consistency with return type hint
    # Format coefficients as strings for cleaner output
    coeficientes_str = [f"{c:.6f}".rstrip('0').rstrip('.') if '.' in f"{c:.6f}" else f"{int(c)}" for c in coeficientes_list]


    return coeficientes_str, str(polinomio_newton) # Return coefficients as strings and the polynomial string

def minimos_cuadrados(x_values, y_values, degree):
    """
    Aproximación por mínimos cuadrados

    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y correspondientes
        degree (int): Grado del polinomio de aproximación

    Returns:
        tuple: (coeficientes, polinomio como string)
    """
    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")

    if degree < 0:
        raise ValueError("El grado del polinomio no puede ser negativo.")

    # np.polyfit requires at least degree + 1 points
    if len(x_values) <= degree:
         raise ValueError(f"Se necesitan al menos {degree + 1} puntos para ajustar un polinomio de grado {degree} por mínimos cuadrados. Solo se proporcionaron {len(x_values)} puntos.")


    # Ajustar el polinomio usando numpy's polyfit
    # np.polyfit returns coefficients in descending order of power: [a_degree, a_{degree-1}, ..., a_1, a_0]
    coeficientes = np.polyfit(x_values, y_values, degree)

    # Construir el polinomio como string using PolinomioMultivariable for consistent formatting
    polinomio_mc = PolinomioMultivariable()
    # Iterate through coefficients and add terms to the PolinomioMultivariable
    # The power of x for coeficientes[i] is degree - i
    for i, coef in enumerate(coeficientes):
        power = degree - i
        # Assuming 'x' is the first variable, exponent is (power_of_x,)
        polinomio_mc.agregar_termino(coef, [power]) # Add term coef * x^power

    # Format coefficients for display (optional)
    coeficientes_list = coeficientes.tolist()
    coeficientes_str = [f"{c:.6f}".rstrip('0').rstrip('.') if '.' in f"{c:.6f}" else f"{int(c)}" for c in coeficientes_list]


    return coeficientes_str, str(polinomio_mc)

def evaluar_polinomio(polinomio_str, x):
    """
    Evalúa un polinomio dado como string en un valor específico de x.
    Se basa en la función evaluar del PolinomioMultivariable.

    Args:
        polinomio_str (str): String representation of the polynomial (e.g., "2x^2 + 3x - 1")
        x (float): The value of x at which to evaluate the polynomial.

    Returns:
        float: The result of the polynomial evaluation.
    """
    try:
        # Use the PolinomioMultivariable class to parse the string and then evaluate
        poly = PolinomioMultivariable.desde_string(polinomio_str)
        # Pass the evaluation point as a keyword argument for the variable 'x'
        return poly.evaluar(x=x)
    except ValueError as e:
        # Catch ValueErrors from PolinomioMultivariable (parsing or evaluation issues)
        # Re-raise with context about the input string and value
        raise ValueError(f"Error al evaluar el polinomio '{polinomio_str}' en x={x}: {str(e)}")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise RuntimeError(f"Error inesperado al evaluar el polinomio '{polinomio_str}' en x={x}: {type(e).__name__}: {str(e)}")


# Simple test block to verify the functions in this module
if __name__ == '__main__':
    print("--- segunda_unidad Tests ---")

    # Test data for interpolation
    x_vals_interp = [0, 1, 2, 3]
    y_vals_interp = [1, 2, 0, 5]

    print("\n--- Lagrange Interpolation Test ---")
    try:
        poly_lagrange = lagrange_interpolation(x_vals_interp, y_vals_interp)
        poly_lagrange_str = str(poly_lagrange)
        print(f"Polinomio de Lagrange: {poly_lagrange_str}")
        print(f"Eval at x=1.5: {evaluar_polinomio(poly_lagrange_str, 1.5):.6f}")
        # Verify interpolation points
        for x, y in zip(x_vals_interp, y_vals_interp):
             eval_y = evaluar_polinomio(poly_lagrange_str, x)
             print(f"Eval at x={x}: {eval_y:.6f} (Expected: {y})")

    except Exception as e:
        print(f"Error in Lagrange Interpolation test: {e}")

    print("\n--- Newton's Divided Differences Test ---")
    try:
        coef_newton, poly_newton_str = diferencias_divididas(x_vals_interp, y_vals_interp)
        print(f"Coeficientes de Newton: {coef_newton}")
        print(f"Polinomio de Newton: {poly_newton_str}")
        print(f"Eval at x=1.5: {evaluar_polinomio(poly_newton_str, 1.5):.6f}")
        # Verify interpolation points
        for x, y in zip(x_vals_interp, y_vals_interp):
             eval_y = evaluar_polinomio(poly_newton_str, x)
             print(f"Eval at x={x}: {eval_y:.6f} (Expected: {y})")

    except Exception as e:
        print(f"Error in Newton's Divided Differences test: {e}")


    print("\n--- Least Squares Approximation Test (Degree 2) ---")
    # Test data for approximation
    x_vals_mc = [0, 1, 2, 3, 4, 5]
    y_vals_mc = [1, 1.8, 3.3, 4.5, 6.3, 7.9]
    degree_mc = 2
    try:
        coef_mc, poly_mc_str = minimos_cuadrados(x_vals_mc, y_vals_mc, degree_mc)
        print(f"Coeficientes Minimos Cuadrados (grado {degree_mc}): {coef_mc}")
        print(f"Polinomio Minimos Cuadrados: {poly_mc_str}")
        print(f"Eval at x=2.5: {evaluar_polinomio(poly_mc_str, 2.5):.6f}")
        # Evaluate at original points (approximation, not exact)
        for x, y in zip(x_vals_mc, y_vals_mc):
            eval_y = evaluar_polinomio(poly_mc_str, x)
            print(f"Eval at x={x}: {eval_y:.6f} (Original y: {y})")


    except Exception as e:
        print(f"Error in Least Squares Approximation test: {e}")

    print("\n--- Test evaluar_polinomio ---")
    test_poly_str = "3x^2 - 5x + 10"
    test_x = 2
    try:
        eval_result = evaluar_polinomio(test_poly_str, test_x)
        print(f"Evaluating '{test_poly_str}' at x={test_x}: {eval_result}") # Expected: 3*4 - 5*2 + 10 = 12 - 10 + 10 = 12
    except Exception as e:
        print(f"Error evaluating '{test_poly_str}': {e}")

    test_poly_str_complex = "-x^3 + 2.5x^2 - 0.5"
    test_x_complex = 1.5
    try:
         eval_result_complex = evaluar_polinomio(test_poly_str_complex, test_x_complex)
         print(f"Evaluating '{test_poly_str_complex}' at x={test_x_complex}: {eval_result_complex}")
    except Exception as e:
         print(f"Error evaluating '{test_poly_str_complex}': {e}")