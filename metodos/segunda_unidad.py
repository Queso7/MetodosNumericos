# segunda_unidad.py (actualizado)
from metodos.Polinomio import PolinomioMultivariable
import numpy as np

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
    
    # Polinomio resultante (inicialmente cero)
    polinomio_final = PolinomioMultivariable.desde_string("0")
    
    for i in range(n):
        # Construir el polinomio base L_i(x)
        L_i = PolinomioMultivariable.desde_string("1")
        
        for j in range(n):
            if j != i:
                # Crear término (x - x_j)
                term_str = f"(x - {x_values[j]})"
                term_poly = PolinomioMultivariable.desde_string(term_str)
                
                # Calcular denominador (x_i - x_j)
                denom = x_values[i] - x_values[j]
                if denom == 0:
                    raise ValueError("Los valores de x deben ser distintos")
                
                # Multiplicar por el término actual
                L_i = L_i * term_poly * (1/denom)
        
        # Sumar y_i * L_i(x) al polinomio final
        termino_final = L_i * y_values[i]
        polinomio_final = polinomio_final + termino_final
    
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
    
    # Calcular tabla de diferencias divididas
    tabla = np.zeros((n, n))
    tabla[:,0] = y_values
    
    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x_values[i+j] - x_values[i])
    
    coeficientes = tabla[0,:]
    
    # Construir el polinomio como string
    polinomio = f"{coeficientes[0]:.6f}"
    for i in range(1, n):
        term = f" + {coeficientes[i]:.6f}"
        for j in range(i):
            term += f"*(x - {x_values[j]:.4f})"
        polinomio += term
    
    return coeficientes, polinomio

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
    
    if degree < 0 or degree >= len(x_values):
        raise ValueError("Grado del polinomio no válido")
    
    # Ajustar el polinomio
    coeficientes = np.polyfit(x_values, y_values, degree)
    
    # Construir el polinomio como string
    polinomio = ""
    for i, coef in enumerate(coeficientes):
        potencia = degree - i
        if potencia == 0:
            term = f"{coef:.6f}"
        else:
            term = f"{coef:.6f}x^{potencia}" if potencia > 1 else f"{coef:.6f}x"
        
        if i == 0:
            polinomio = term
        else:
            if coef >= 0:
                polinomio += f" + {term}"
            else:
                polinomio += f" - {abs(coef):.6f}x^{potencia}" if potencia > 1 else f" - {abs(coef):.6f}x"
    
    return coeficientes, polinomio

def evaluar_polinomio(polinomio_str, x):
    """
    Evalúa un polinomio dado como string en un punto x
    
    Args:
        polinomio_str (str): Polinomio como string (ej. "3x^2 + 2x - 1")
        x (float): Valor de x a evaluar
        
    Returns:
        float: Resultado de evaluar el polinomio en x
    """
    try:
        # Convertir el polinomio a una expresión evaluable
        expr = polinomio_str.replace('^', '**')
        # Reemplazar x por el valor dado
        expr = expr.replace('x', f'({x})')
        # Evaluar la expresión
        return eval(expr)
    except:
        raise ValueError("No se pudo evaluar el polinomio")