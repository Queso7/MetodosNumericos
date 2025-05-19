# integracion_numerica.py
import numpy as np
import math

class IntegracionNumerica:
    @staticmethod
    def trapezoidal(f, a, b, n):
        if n <= 0:
            raise ValueError("El número de intervalos debe ser positivo")
        if a >= b:
            raise ValueError("El límite inferior debe ser menor que el superior")

        h = (b - a) / n
        suma = 0.5 * (f(a) + f(b))

        for i in range(1, n):
            suma += f(a + i * h)

        return suma * h

    @staticmethod
    def simpson13(f, a, b, n):
        if n <= 0:
            raise ValueError("El número de intervalos debe ser positivo")
        if a >= b:
            raise ValueError("El límite inferior debe ser menor que el superior")
        if n % 2 != 0:
            raise ValueError("Simpson 1/3 requiere un número par de intervalos")

        h = (b - a) / n
        suma = f(a) + f(b)

        for i in range(1, n):
            x = a + i * h
            if i % 2 == 0:
                suma += 2 * f(x)
            else:
                suma += 4 * f(x)

        return suma * h / 3

    @staticmethod
    def simpson38(f, a, b, n):
        if n <= 0:
            raise ValueError("El número de intervalos debe ser positivo")
        if a >= b:
            raise ValueError("El límite inferior debe ser menor que el superior")
        if n % 3 != 0:
            raise ValueError("Simpson 3/8 requiere un número de intervalos múltiplo de 3")

        h = (b - a) / n
        suma = f(a) + f(b)

        for i in range(1, n):
            x = a + i * h
            if i % 3 == 0:
                suma += 2 * f(x)
            else:
                suma += 3 * f(x)

        return suma * 3 * h / 8

    @staticmethod
    def evaluar_funcion(func_str, x):
        """
        Evalúa una función matemática dada como string en un valor específico de x.
        Permite el uso de funciones de math (sin, cos, exp, etc.) y la variable 'x'.
        Soporta '^' para potencias.
        """
        try:
            # Replace '^' with '**' for powers
            func_str = func_str.replace('^', '**')

            # Create a safe environment for eval
            # Only allow functions/constants from the math module and the variable 'x'
            safe_dict = {
                'math': math,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
                'exp': math.exp, 'log': math.log, 'log10': math.log10,
                'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
                'fabs': math.fabs, 'factorial': math.factorial,
                'gamma': math.gamma, 'lgamma': math.lgamma,
                'erf': math.erf, 'erfc': math.erfc,
                'radians': math.radians, 'degrees': math.degrees,
                'floor': math.floor, 'ceil': math.ceil, 'trunc': math.trunc,
                'fmod': math.fmod, 'modf': math.modf, 'ldexp': math.ldexp,
                'frexp': math.frexp, 'fsum': math.fsum, 'prod': math.prod,
                'tau': math.tau, 'inf': math.inf, 'nan': math.nan,

                # Allow numpy functions that might be used in numerical contexts
                'np': np,
                'power': np.power, # Handle x^power case explicitly if needed, though '**' should work
                'abs': abs, # built-in abs is also useful

                'x': x # The variable x
            }
            # Restrict builtins
            restricted_globals = {'__builtins__': None}


            # Evaluate the expression
            # Use restricted_globals to prevent access to unwanted builtins
            # Use safe_dict as local environment
            return eval(func_str, restricted_globals, safe_dict)
        except (ValueError, TypeError, SyntaxError, NameError) as e:
            raise ValueError(f"Error al evaluar la función '{func_str}' en x={x}: {str(e)}")
        except Exception as e:
             raise ValueError(f"Error inesperado al evaluar la función '{func_str}' en x={x}: {type(e).__name__}: {str(e)}")

# Simple test cases
if __name__ == '__main__':
    print("--- Integracion Numerica Tests ---")

    # Define a simple function using the string evaluator
    def test_func(x):
        return IntegracionNumerica.evaluar_funcion("x^2 + 2*math.sin(x)", x)

    a = 0
    b = 1
    n_trap = 100
    n_simp13 = 100 # Must be even
    n_simp38 = 99 # Must be multiple of 3

    print(f"Integrating x^2 + 2*sin(x) from {a} to {b}")

    try:
        integral_trap = IntegracionNumerica.trapezoidal(test_func, a, b, n_trap)
        print(f"Trapezoidal ({n_trap} intervals): {integral_trap:.6f}")
    except Exception as e:
        print(f"Error in Trapezoidal method: {e}")

    try:
        integral_simp13 = IntegracionNumerica.simpson13(test_func, a, b, n_simp13)
        print(f"Simpson 1/3 ({n_simp13} intervals): {integral_simp13:.6f}")
    except Exception as e:
        print(f"Error in Simpson 1/3 method: {e}")

    try:
        integral_simp38 = IntegracionNumerica.simpson38(test_func, a, b, n_simp38)
        print(f"Simpson 3/8 ({n_simp38} intervals): {integral_simp38:.6f}")
    except Exception as e:
        print(f"Error in Simpson 3/8 method: {e}")

    print("\n--- Function Evaluation Tests ---")
    try:
        result1 = IntegracionNumerica.evaluar_funcion("x**2 + math.sqrt(x)", 4)
        print(f"evaluar_funcion('x**2 + math.sqrt(x)', 4): {result1}")
    except Exception as e:
        print(f"Error evaluating function: {e}")

    try:
        result2 = IntegracionNumerica.evaluar_funcion("2*x + 5", 3)
        print(f"evaluar_funcion('2*x + 5', 3): {result2}")
    except Exception as e:
        print(f"Error evaluating function: {e}")

    try:
        result3 = IntegracionNumerica.evaluar_funcion("math.cos(math.pi)", 0)
        print(f"evaluar_funcion('math.cos(math.pi)', 0): {result3}")
    except Exception as e:
        print(f"Error evaluating function: {e}")

    try:
        # Test with an invalid function string
        IntegracionNumerica.evaluar_funcion("sin(x) + log(x", 1)
    except Exception as e:
        print(f"Expected error caught: {e}")

    try:
        # Test with unsupported operation
        IntegracionNumerica.evaluar_funcion("x + 100", 1) # This should work now with eval and safe_dict
        print(f"evaluar_funcion('x + 100', 1): {IntegracionNumerica.evaluar_funcion('x + 100', 1)}")
    except Exception as e:
        print(f"Error evaluating function: {e}")