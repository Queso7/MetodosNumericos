from .Polinomio import PolinomioMultivariable 

class EcuacionMultivariable:
    def __init__(self, ecuacion_str):
        self.ecuacion_original = ecuacion_str
        self.lado_izq, self.lado_der = self._parsear_ecuacion(ecuacion_str)
        self.polinomio = self._construir_polinomio()
    
    def _parsear_ecuacion(self, ecuacion_str):
        ecuacion = ecuacion_str.replace(" ", "").lower()
        if '=' not in ecuacion:
            raise ValueError("La ecuación debe contener '='")
        partes = ecuacion.split('=', 1)
        return partes[0], partes[1]
    
    def _construir_polinomio(self):
        pol_izq = PolinomioMultivariable.desde_string(self.lado_izq)
        pol_der = PolinomioMultivariable.desde_string(self.lado_der)
        resultado = PolinomioMultivariable()
        for exp, coef in pol_izq.terminos.items():
            resultado.agregar_termino(coef, exp)
        for exp, coef in pol_der.terminos.items():
            resultado.agregar_termino(-coef, exp)
        return resultado
    
    def __str__(self):
        return f"Ecuación: {self.ecuacion_original}\nForma estándar: {self.polinomio} = 0"

