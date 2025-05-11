import re
from collections import defaultdict

class PolinomioMultivariable:
    def __init__(self, terminos=None):
        self.terminos = terminos if terminos else defaultdict(int)
        self.variables = {'x', 'y', 'z'}
    
    def agregar_termino(self, coeficiente, exponentes):
        clave = tuple(exponentes)
        self.terminos[clave] += coeficiente
        if self.terminos[clave] == 0:
            del self.terminos[clave]
            
    @classmethod
    def desde_string(cls, expr):
        if not expr:
            return cls()
        p = cls()
        patron = r'([+-]?\d*[a-z](?:\^\d+)?(?:[a-z](?:\^\d+)?)*)|([+-]?\d+)'
        for term in re.findall(patron, expr.replace(" ", "").lower()):
            term = term[0] or term[1]
            if not term:
                continue
            signo = -1 if term.startswith('-') else 1
            term = term.lstrip('+-')
            coef_match = re.match(r'^\d+', term)
            coef = signo * (int(coef_match.group()) if coef_match else 1)
            exp = [0, 0, 0]
            for var in re.finditer(r'([a-z])(?:\^(\d+))?', term):
                letra = var.group(1)
                potencia = int(var.group(2)) if var.group(2) else 1
                idx = ord(letra) - ord('x')
                if 0 <= idx <= 2:
                    exp[idx] = potencia
            p.agregar_termino(coef, exp)
        return p
    
    def derivada_parcial(self, variable):
        if variable not in self.variables:
            raise ValueError(f"Variable debe ser una de {self.variables}")
        nuevo = defaultdict(int)
        idx = ord(variable) - ord('x')
        for exp, coef in self.terminos.items():
            if exp[idx] > 0:
                nuevo_exp = list(exp)
                nuevo_coef = coef * exp[idx]
                nuevo_exp[idx] -= 1
                nuevo[tuple(nuevo_exp)] = nuevo_coef
        return PolinomioMultivariable(nuevo)
    
    def evaluar(self, **valores):
        total = 0
        for exp, coef in self.terminos.items():
            term_val = coef
            for i, e in enumerate(exp):
                var = chr(ord('x') + i)
                term_val *= valores.get(var, 0)**e
            total += term_val
        return total
    
    def __str__(self):
        terminos = []
        for exp, coef in sorted(self.terminos.items(), reverse=True):
            if coef == 0:
                continue
                
            partes = []
            if abs(coef) != 1 or all(e == 0 for e in exp):
                partes.append(f"{coef}")
            elif coef == -1:
                partes.append("-")
            
            for i, e in enumerate(exp):
                if e > 0:
                    var = chr(ord('x') + i)
                    partes.append(f"{var}^{e}" if e > 1 else var)
            
            term = "".join(partes).replace("^1", "")
            terminos.append(term)
        
        if not terminos:
            return "0"
            
        expr = " + ".join(terminos).replace("+ -", " - ")
        if expr.startswith(" + "):
            expr = expr[3:]
        if expr.startswith(" - "):
            expr = "-" + expr[3:]
        return expr