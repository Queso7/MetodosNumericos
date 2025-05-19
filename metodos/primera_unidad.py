# primera_unidad.py (actualizado)
import numpy as np
from metodos.Polinomio import PolinomioMultivariable
from metodos.Ecuacion import EcuacionMultivariable

def newton_raphson_sistema(ecuaciones, variables, x0, tol=1e-6, max_iter=100):
    """
    Resuelve un sistema de ecuaciones no lineales usando el método de Newton-Raphson
    
    Args:
        ecuaciones (list): Lista de strings con las ecuaciones
        variables (list): Lista de variables (ej. ['x', 'y'])
        x0 (list): Valores iniciales
        tol (float): Tolerancia para la convergencia
        max_iter (int): Máximo número de iteraciones
        
    Returns:
        dict: {'solucion': lista con solución, 'iteraciones': número de iteraciones, 'error': error final}
    """
    n = len(ecuaciones)
    if n != len(variables) or n != len(x0):
        raise ValueError("Número de ecuaciones, variables y valores iniciales debe coincidir")
    
    # Convertir ecuaciones a objetos EcuacionMultivariable
    ec_objs = [EcuacionMultivariable(ec) for ec in ecuaciones]
    
    x = np.array(x0, dtype=float)
    iteraciones = 0
    
    for _ in range(max_iter):
        F = np.zeros(n)
        J = np.zeros((n, n))
        
        # Evaluar funciones y jacobiano
        for i in range(n):
            # Evaluar F_i
            F[i] = ec_objs[i].polinomio.evaluar(**dict(zip(variables, x)))
            
            # Calcular derivadas parciales para el Jacobiano
            for j in range(n):
                derivada = ec_objs[i].polinomio.derivada_parcial(variables[j])
                J[i,j] = derivada.evaluar(**dict(zip(variables, x)))
        
        # Verificar si hemos convergido
        if np.linalg.norm(F) < tol:
            break
        
        # Resolver sistema lineal J * delta = -F
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            raise ValueError("Jacobiano singular, no se puede continuar")
        
        # Actualizar solución
        x += delta
        iteraciones += 1
    
    error = np.linalg.norm(F)
    if error >= tol and iteraciones >= max_iter:
        raise ValueError("El método no convergió en el número máximo de iteraciones")
    
    return {
        'solucion': x.tolist(),
        'iteraciones': iteraciones,
        'error': error
    }

def secante_sistema(ecuaciones, variables, x0, x1=None, tol=1e-6, max_iter=100):
    """
    Resuelve un sistema de ecuaciones no lineales usando el método de la secante
    
    Args:
        ecuaciones (list): Lista de strings con las ecuaciones
        variables (list): Lista de variables (ej. ['x', 'y'])
        x0 (list): Primer conjunto de valores iniciales
        x1 (list): Segundo conjunto de valores iniciales (opcional)
        tol (float): Tolerancia para la convergencia
        max_iter (int): Máximo número de iteraciones
        
    Returns:
        dict: {'solucion': lista con solución, 'iteraciones': número de iteraciones, 'error': error final}
    """
    n = len(ecuaciones)
    if n != len(variables) or n != len(x0):
        raise ValueError("Número de ecuaciones, variables y valores iniciales debe coincidir")
    
    if x1 is None:
        x1 = [xi + 0.1 for xi in x0]  # Pequeño desplazamiento si no se proporciona x1
    
    # Convertir ecuaciones a objetos EcuacionMultivariable
    ec_objs = [EcuacionMultivariable(ec) for ec in ecuaciones]
    
    x_prev = np.array(x0, dtype=float)
    x_curr = np.array(x1, dtype=float)
    iteraciones = 0
    
    for _ in range(max_iter):
        F_prev = np.zeros(n)
        F_curr = np.zeros(n)
        
        # Evaluar funciones en los puntos actual y anterior
        for i in range(n):
            F_prev[i] = ec_objs[i].polinomio.evaluar(**dict(zip(variables, x_prev)))
            F_curr[i] = ec_objs[i].polinomio.evaluar(**dict(zip(variables, x_curr)))
        
        # Verificar si hemos convergido
        if np.linalg.norm(F_curr) < tol:
            break
        
        # Aproximación del Jacobiano usando diferencias finitas
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                J[i,j] = (F_curr[i] - F_prev[i]) / (x_curr[j] - x_prev[j]) if (x_curr[j] - x_prev[j]) != 0 else 1e-10
        
        # Resolver sistema lineal J * delta = -F
        try:
            delta = np.linalg.solve(J, -F_curr)
        except np.linalg.LinAlgError:
            raise ValueError("Matriz de aproximación singular, no se puede continuar")
        
        # Actualizar solución
        x_next = x_curr + delta
        x_prev = x_curr
        x_curr = x_next
        iteraciones += 1
    
    error = np.linalg.norm(F_curr)
    if error >= tol and iteraciones >= max_iter:
        raise ValueError("El método no convergió en el número máximo de iteraciones")
    
    return {
        'solucion': x_curr.tolist(),
        'iteraciones': iteraciones,
        'error': error
    }