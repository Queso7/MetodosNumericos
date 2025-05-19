from metodos.Polinomio import PolinomioMultivariable
import numpy as np
import math

def lagrange_interpolation(x_values, y_values):
    """
    Interpolación de Lagrange mejorada y funcional
    
    Args:
        x_values (list): Lista de valores x (deben ser distintos)
        y_values (list): Lista de valores y correspondientes
        
    Returns:
        PolinomioMultivariable: Polinomio de interpolación en formato PolinomioMultivariable
        
    Raises:
        ValueError: Si los inputs son inválidos
    """
    # Validación de entradas
    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")
    if len(x_values) == 0:
        raise ValueError("Se requiere al menos un punto para la interpolación")
    if len(set(x_values)) != len(x_values):
        duplicates = {x for x in x_values if x_values.count(x) > 1}
        raise ValueError(f"Valores de x deben ser distintos. Duplicados encontrados: {duplicates}")

    n = len(x_values)
    polinomio_final = PolinomioMultivariable({(0,): 0.0})  # Polinomio cero inicial

    for i in range(n):
        # Construir el polinomio base L_i(x)
        L_i = PolinomioMultivariable({(0,): 1.0})  # Iniciar con 1

        for j in range(n):
            if j != i:
                # Crear término (x - x_j)
                term = PolinomioMultivariable({
                    (1,): 1.0,    # x^1
                    (0,): -x_values[j]  # -x_j
                })
                
                # Calcular denominador (x_i - x_j)
                denom = x_values[i] - x_values[j]
                if denom == 0:
                    raise ValueError("División por cero en cálculo de coeficientes")
                
                # Multiplicar L_i por (x - x_j)/(x_i - x_j)
                L_i = L_i * term * (1.0 / denom)

        # Sumar y_i * L_i(x) al polinomio final
        termino_final = L_i * y_values[i]
        polinomio_final = polinomio_final + termino_final

    return polinomio_final

def diferencias_divididas(x_values, y_values):
    """
    Interpolación por diferencias divididas (Newton) mejorada
    
    Args:
        x_values (list): Lista de valores x (deben ser distintos)
        y_values (list): Lista de valores y correspondientes
        
    Returns:
        tuple: (coeficientes, polinomio como string)
    """
    # Validación de entradas
    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")
    if len(x_values) == 0:
        raise ValueError("Se requiere al menos un punto para la interpolación")
    if len(set(x_values)) != len(x_values):
        duplicates = {x for x in x_values if x_values.count(x) > 1}
        raise ValueError(f"Valores de x deben ser distintos. Duplicados encontrados: {duplicates}")

    n = len(x_values)
    f = np.zeros((n, n))
    f[:,0] = y_values  # Primera columna son los valores y

    # Calcular tabla de diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            denom = x_values[i+j] - x_values[i]
            if denom == 0:
                raise ValueError("División por cero en cálculo de diferencias divididas")
            f[i,j] = (f[i+1,j-1] - f[i,j-1]) / denom

    coeficientes = f[0,:]

    # Construir el polinomio de Newton
    polinomio = PolinomioMultivariable({(0,): coeficientes[0]})
    
    for i in range(1, n):
        term = PolinomioMultivariable({(0,): coeficientes[i]})
        for j in range(i):
            factor = PolinomioMultivariable({
                (1,): 1.0,
                (0,): -x_values[j]
            })
            term = term * factor
        polinomio = polinomio + term

    # Formatear coeficientes para mostrar
    coef_str = [f"{c:.6f}".rstrip('0').rstrip('.') if '.' in f"{c:.6f}" else f"{c}" 
                for c in coeficientes]
    
    return coef_str, str(polinomio)

def minimos_cuadrados(x_values, y_values, degree):
    """
    Aproximación por mínimos cuadrados mejorada
    
    Args:
        x_values (list): Lista de valores x
        y_values (list): Lista de valores y
        degree (int): Grado del polinomio de aproximación
        
    Returns:
        tuple: (coeficientes, polinomio como string)
    """
    # Validación de entradas
    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud")
    if degree < 0:
        raise ValueError("El grado del polinomio no puede ser negativo")
    if len(x_values) <= degree:
        raise ValueError(f"Se necesitan al menos {degree+1} puntos para ajustar un polinomio de grado {degree}")

    # Ajustar el polinomio usando numpy
    coeficientes = np.polyfit(x_values, y_values, degree)
    
    # Construir el polinomio
    polinomio = PolinomioMultivariable()
    for i, coef in enumerate(coeficientes):
        power = degree - i
        polinomio.agregar_termino(coef, [power])

    # Formatear coeficientes para mostrar
    coef_str = [f"{c:.6f}".rstrip('0').rstrip('.') if '.' in f"{c:.6f}" else f"{c}" 
                for c in coeficientes]
    
    return coef_str, str(polinomio)

def evaluar_polinomio(polinomio_str, x):
    """
    Evalúa un polinomio dado como string en un valor x
    
    Args:
        polinomio_str (str): Polinomio como string (ej. "3x^2 + 2x - 1")
        x (float): Valor donde evaluar
        
    Returns:
        float: Resultado de la evaluación
    """
    try:
        poly = PolinomioMultivariable.desde_string(polinomio_str)
        return poly.evaluar(x=x)
    except Exception as e:
        raise ValueError(f"Error al evaluar polinomio: {str(e)}")

# Tests
if __name__ == '__main__':
    print("--- Pruebas de Interpolación Mejorada ---")
    
    # Datos de prueba
    x_data = [0, 1, 2, 3]
    y_data = [1, 2, 0, 5]
    
    print("\nLagrange:")
    poly_lag = lagrange_interpolation(x_data, y_data)
    print(f"Polinomio: {poly_lag}")
    print(f"Evaluación en x=1.5: {evaluar_polinomio(str(poly_lag), 1.5)}")
    
    print("\nNewton:")
    coef_new, poly_new = diferencias_divididas(x_data, y_data)
    print(f"Coeficientes: {coef_new}")
    print(f"Polinomio: {poly_new}")
    print(f"Evaluación en x=1.5: {evaluar_polinomio(poly_new, 1.5)}")
    
    print("\nMínimos Cuadrados (grado 2):")
    coef_mc, poly_mc = minimos_cuadrados(x_data, y_data, 2)
    print(f"Coeficientes: {coef_mc}")
    print(f"Polinomio: {poly_mc}")
    print(f"Evaluación en x=1.5: {evaluar_polinomio(poly_mc, 1.5)}")