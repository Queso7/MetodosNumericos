# integracion_numerica.py (nuevo archivo)
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
        try:
            # Reemplazar '^' por '**' para potencias
            func_str = func_str.replace('^', '**')
            # Asegurarse de que math esté disponible
            safe_dict = {'math': math, 'sin': math.sin, 'cos': math.cos, 
                        'tan': math.tan, 'exp': math.exp, 'log': math.log,
                        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e}
            # Evaluar la expresión
            return eval(func_str, {'__builtins__': None}, safe_dict)
        except Exception as e:
            raise ValueError(f"No se pudo evaluar la función: {str(e)}")