import re
from collections import defaultdict

class PolinomioMultivariable:
    def __init__(self, terminos=None):
        self.terminos = terminos if terminos is not None else defaultdict(float) # Use float for coefficients
        self.variables = {'x', 'y', 'z'} # Define supported variables. Order matters for exponent tuples.
        self._variable_order = sorted(list(self.variables)) # Consistent order for exponent tuple indexing


    def agregar_termino(self, coeficiente, exponentes):
        # Ensure exponents are a tuple for hashability and are the correct length
        # Adjust expected length based on defined variables
        expected_len = len(self._variable_order)
        if len(exponentes) != expected_len:
             raise ValueError(f"Exponentes deben tener longitud {expected_len} para variables {self._variable_order}")

        # Ensure exponents are non-negative integers
        if not all(isinstance(e, int) and e >= 0 for e in exponentes):
            raise ValueError(f"Exponentes deben ser enteros no negativos: {exponentes}")

        exp_tuple = tuple(exponentes)

        self.terminos[exp_tuple] += float(coeficiente) # Ensure coefficient is float
        if abs(self.terminos[exp_tuple]) < 1e-9: # Remove terms close to zero
            del self.terminos[exp_tuple]

    @classmethod
    def desde_string(cls, expr):
        # This class method assumes the polynomial string is in a format it can parse
        # It is primarily tuned for single-variable polynomials in 'x' for interpolation/approximation.
        # It attempts to handle simple multivariable terms based on the regex,
        # assuming variables are x, y, z and appear in that order or with explicit powers.

        if not expr:
            return cls()

        p = cls()
        # Regex to capture terms: Optional sign, optional coefficient (int or float),
        # followed by variable(s) with optional powers, or just a constant.
        patron = r'([+-]?\s*\d*\.?\d*\s*(?:[a-z](?:\s*\^\s*\d+)?)+)|([+-]?\s*\d+\.?\d*)'
        expr = expr.strip().replace(" ", "").lower() # Remove spaces and convert to lower case

        if not expr: # Handle case where expr was just spaces
             return cls()

        # Add a leading + if the first term doesn't have a sign, makes parsing easier
        if expr[0] not in ['+', '-']:
            expr = '+' + expr

        # Find all potential terms using the pattern
        matches = list(re.finditer(patron, expr))

        if not matches and expr.strip() not in ('+', '-'):
             # If no matches but the expression is not just a sign, it's invalid
             # Simple constant or single variable term without explicit coef might not match the first group initially,
             # but should match the second or be handled by the logic below.
             # Let's re-check for simple cases not covered by the main pattern if no matches are found.
             simple_term_match = re.fullmatch(r'([+-]?\s*\d*\.?\d*[a-z]?\s*(?:\^\s*\d+)?)', expr)
             if simple_term_match:
                  # Re-parse as a single term using a simpler logic or specific patterns
                  pass # Continue to parsing logic below which handles constant and simple variable terms

             else:
                 raise ValueError(f"No se pudo parsear la expresión: '{expr}'")

        processed_until = 0
        for match in re.finditer(patron, expr) if matches else re.finditer(r'([+-]?\s*\d*\.?\d*\s*(?:[a-z](?:\s*\^\s*\d+)?)+)|([+-]?\s*\d+\.?\d*)|([+-]?\s*[a-z])', expr): # Use a slightly broader pattern if initial match fails, or iterate through original matches
             # Use the original matches list if available, otherwise the broader pattern finditer
             if matches:
                  term_str = match.group(0).strip() # Get the matched string for the term
             else:
                  # If using the broader pattern for fallback, group handling might differ
                  # Need to get the full matched string
                  term_str = match.group(0).strip()


             if not term_str or term_str in ('+', '-'):
                 # Skip empty matches or terms that are just signs (should be handled by leading sign)
                 continue

             # Ensure continuous parsing - check if there is a gap between matches
             if match.start() > processed_until:
                 unparsed_part = expr[processed_until : match.start()]
                 if unparsed_part.strip() and unparsed_part not in ('+', '-'): # Allow '+' or '-' as separators
                      raise ValueError(f"No se pudo parsear la parte de la expresión: '{unparsed_part}' antes de '{term_str}'")

             sign = -1.0 if term_str.startswith('-') else 1.0
             term_str_abs = term_str.lstrip('+-').strip()

             if not term_str_abs: # Should be handled by the check above, but safety
                 continue


             coef = sign * 1.0 # Default coefficient is 1 or -1 if no number is present
             coef_part_match = re.match(r'^(\d*\.?\d*)', term_str_abs) # Match leading number (coefficient)

             remaining_term_str = term_str_abs

             if coef_part_match and coef_part_match.group(1):
                 coef_str_val = coef_part_match.group(1)
                 if coef_str_val in ['', '.']: # Case like 'x' or '.5x' where coefficient is implicit or partial float
                      # If coefficient part is empty or just '.', default coef = sign * 1.0 (already set)
                      remaining_term_str = term_str_abs[coef_part_match.end():]
                      # If there's no remaining string, it was just a number like '.', handle below
                      if not remaining_term_str and coef_str_val == '.':
                           raise ValueError(f"Término inválido: '.'")

                 else:
                      try:
                           coef = sign * float(coef_str_val)
                           remaining_term_str = term_str_abs[coef_part_match.end():]
                      except ValueError:
                           # Should not happen with the regex, but as a safeguard
                           raise ValueError(f"Coeficiente no válido en el término: '{term_str}'")


             exp = [0] * len(p._variable_order) # Initialize exponents for all variables in order
             variable_part_match = re.match(r'^\s*((?:[a-z](?:\s*\^\s*\d+)?)+)', remaining_term_str) # Match the variable part


             if not variable_part_match:
                 # If no variable part matched, it must be a constant term
                 if remaining_term_str.strip(): # If there's leftover text, it's an error
                      raise ValueError(f"Parte no parseada después del coeficiente: '{remaining_term_str}' en el término '{term_str}'")
                 # The coefficient was already determined above
                 p.agregar_termino(coef, exp) # Add as a constant term
             else:
                # Term with variables
                variable_str_part = variable_part_match.group(1)
                if remaining_term_str[variable_part_match.end():].strip():
                     # If there is text after the variable part, it's an error
                     raise ValueError(f"Parte no parseada después de las variables: '{remaining_term_str[variable_part_match.end():]}' en el término '{term_str}'")


                # Parse variables and powers within the variable_str_part
                used_variable_chars = set()
                for var_power_match in re.finditer(r'([a-z])(?:\s*\^\s*(\d+))?', variable_str_part):
                    letra = var_power_match.group(1)
                    if letra in used_variable_chars:
                         raise ValueError(f"Variable duplicada '{letra}' en el término: '{term_str}'")
                    used_variable_chars.add(letra)

                    potencia_str = var_power_match.group(2)
                    potencia = int(potencia_str) if potencia_str else 1 # Default power is 1 if no exponent

                    try:
                        idx = p._variable_order.index(letra) # Get index based on consistent order
                    except ValueError:
                         raise ValueError(f"Variable '{letra}' no soportada. Use solo: {p._variable_order}")

                    exp[idx] = potencia # Set the exponent for the corresponding variable

                p.agregar_termino(coef, exp) # Add the variable term


             processed_until = match.end() # Update the processed position


        # Final check to ensure the entire string was parsed
        if processed_until != len(expr):
            unparsed_part = expr[processed_until:].strip()
            if unparsed_part and unparsed_part not in ('+', '-'):
                raise ValueError(f"Parte final de la expresión no parseada: '{unparsed_part}'")


        return p


    def derivada_parcial(self, variable):
        if variable not in self.variables:
            raise ValueError(f"Variable de derivación debe ser una de {list(self.variables)}")
        nuevo = defaultdict(float)
        # Find the index corresponding to the variable based on consistent order
        try:
            idx = self._variable_order.index(variable)
        except ValueError:
             # This should not happen if the variable is in self.variables, but as safeguard
             raise ValueError(f"Variable '{variable}' no encontrada en las variables del polinomio.")

        for exp, coef in self.terminos.items():
            if exp[idx] > 0:
                nuevo_exp = list(exp)
                nuevo_coef = coef * exp[idx]
                nuevo_exp[idx] -= 1
                nuevo[tuple(nuevo_exp)] = nuevo_coef
        return PolinomioMultivariable(nuevo)

    def evaluar(self, **valores):
        total = 0.0
        # Ensure all defined variables have a value for evaluation, default to 0 if not provided
        # Use the consistent variable order
        eval_valores = {var: float(valores.get(var, 0.0)) for var in self._variable_order}


        for exp, coef in self.terminos.items():
            term_val = float(coef) # Start with the coefficient
            try:
                for i, var_char in enumerate(self._variable_order):
                     e = exp[i]
                     if e != 0: # Only multiply if the exponent is not zero
                         # Check if the value for the variable is zero and exponent is negative
                         if eval_valores[var_char] == 0.0 and e < 0:
                              raise ValueError(f"Evaluación en 0 para la variable '{var_char}' con exponente negativo ({e}) no definida en el término con exponentes {exp}. Término: {coef}{''.join([f'{v}^{exp[j]}' for j,v in enumerate(self._variable_order) if exp[j]!=0])}")
                         term_val *= (eval_valores[var_char] ** e) # Calculate variable part

            except ZeroDivisionError:
                # This specific error should be caught by the explicit check above, but keeping as safeguard
                raise ValueError(f"División por cero durante la evaluación del término con coef {coef} y exp {exp}.")
            except TypeError:
                 # Occurs if a non-numeric value is used in evaluation (should be caught by float conversion earlier, but safety)
                 raise ValueError(f"Valores de evaluación deben ser numéricos.")
            except Exception as e:
                 # Catch any other unexpected errors during term evaluation
                 raise RuntimeError(f"Error al evaluar el término con coef {coef} y exp {exp}: {str(e)}")


            total += term_val
        return total

    def __add__(self, other):
        if isinstance(other, PolinomioMultivariable):
            # Ensure both polynomials have the same variable order for consistent exponent handling
            if self._variable_order != other._variable_order:
                 raise ValueError("No se pueden sumar polinomios con diferente orden de variables.")

            resultado = PolinomioMultivariable(self.terminos.copy())
            resultado._variable_order = self._variable_order # Keep the same variable order

            for exp, coef in other.terminos.items():
                # Exponents from other polynomial should already match length/order due to the check above
                resultado.agregar_termino(coef, exp)
            return resultado
        elif isinstance(other, (int, float)):
            resultado = PolinomioMultivariable(self.terminos.copy())
            resultado._variable_order = self._variable_order # Keep the same variable order
            # Add the scalar as a constant term (exponents [0, 0, ...])
            resultado.agregar_termino(other, [0] * len(self._variable_order))
            return resultado
        else:
             return NotImplemented # Allow Python to try the other object's __radd__

    def __radd__(self, other):
        return self.__add__(other) # Addition is commutative

    def __sub__(self, other):
        if isinstance(other, PolinomioMultivariable):
            # Ensure both polynomials have the same variable order
            if self._variable_order != other._variable_order:
                 raise ValueError("No se pueden restar polinomios con diferente orden de variables.")

            resultado = PolinomioMultivariable(self.terminos.copy())
            resultado._variable_order = self._variable_order # Keep the same variable order

            for exp, coef in other.terminos.items():
                 # Exponents from other polynomial should already match length/order
                resultado.agregar_termino(-coef, exp) # Subtract by adding the negative coefficient
            return resultado
        elif isinstance(other, (int, float)):
            resultado = PolinomioMultivariable(self.terminos.copy())
            resultado._variable_order = self._variable_order # Keep the same variable order
            # Subtract the scalar by adding the negative constant term
            resultado.agregar_termino(-other, [0] * len(self._variable_order))
            return resultado
        else:
             return NotImplemented

    def __rsub__(self, other):
         if isinstance(other, (int, float)):
             # If a scalar is subtracting the polynomial: other - self
             resultado = PolinomioMultivariable()
             resultado._variable_order = self._variable_order # Keep the same variable order
             resultado.agregar_termino(other, [0] * len(self._variable_order)) # Start with the scalar as a constant
             for exp, coef in self.terminos.items():
                 resultado.agregar_termino(-coef, exp) # Subtract each term of self
             return resultado
         else:
              return NotImplemented


    def __mul__(self, other):
        if isinstance(other, PolinomioMultivariable):
            # Ensure both polynomials have the same variable order for consistent exponent handling
            if self._variable_order != other._variable_order:
                 raise ValueError("No se pueden multiplicar polinomios con diferente orden de variables.")

            resultado = PolinomioMultivariable()
            resultado._variable_order = self._variable_order # Keep the same variable order

            for exp1, coef1 in self.terminos.items():
                for exp2, coef2 in other.terminos.items():
                    # Add exponents for multiplication. Assumes exp1 and exp2 have the same order/correspondence.
                    nuevo_exp = [e1 + e2 for e1, e2 in zip(exp1, exp2)]
                    # Ensure the new exponent tuple has the correct length
                    if len(nuevo_exp) != len(self._variable_order):
                         raise RuntimeError("Error interno: La suma de exponentes resultó en una longitud incorrecta.")
                    resultado.agregar_termino(coef1 * coef2, nuevo_exp)
            return resultado
        elif isinstance(other, (int, float)):
            # If multiplying by a scalar
            resultado = PolinomioMultivariable()
            resultado._variable_order = self._variable_order # Keep the same variable order

            for exp, coef in self.terminos.items():
                resultado.agregar_termino(coef * other, exp)
            return resultado
        else:
             return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other) # Multiplication is commutative

    def __str__(self):
        terminos = []
        # Sort terms for consistent output: by total degree descending, then by exponents in variable order descending
        sorted_terms = sorted(
            self.terminos.items(),
            key=lambda item: (-sum(item[0]),) + tuple(-e for e in item[0]),
        )

        for exp, coef in sorted_terms:
            if abs(coef) < 1e-9: # Skip terms close to zero
                continue

            partes_termino = []

            # Add sign and coefficient
            if coef > 0 and terminos: # Add '+' before positive terms after the first one
                 partes_termino.append(" + ")
            elif coef < 0:
                 partes_termino.append(" - ")

            abs_coef = abs(coef)

            # Add coefficient number unless it's 1 or -1 AND it's a variable term
            is_constant_term = all(e == 0 for e in exp)
            if abs_coef != 1.0 or is_constant_term:
                 # Format coefficient: use .nf for floats, integer for whole numbers
                 # Use a reasonable precision, e.g., 6 decimal places, and strip trailing zeros/decimal if integer
                 if abs(abs_coef - round(abs_coef)) < 1e-9: # Check if coefficient is very close to an integer
                      coef_str = str(int(round(abs_coef)))
                 else:
                      coef_str = f"{abs_coef:.6f}".rstrip('0').rstrip('.')
                      if not coef_str: coef_str = "0" # Handle case where .rstrip makes it empty

                 partes_termino.append(coef_str)

            # Add variable parts
            variables_part = []
            # Iterate through exponents in the defined variable order
            for i, var_char in enumerate(self._variable_order):
                 e = exp[i]
                 if e > 0:
                     if e == 1:
                          variables_part.append(var_char)
                     else:
                          variables_part.append(f"{var_char}^{e}")

            # Combine coefficient and variable parts
            if not variables_part and not is_constant_term:
                # This case implies a term like "x^0" which should be a constant and handled above
                # Or a coefficient of 1 or -1 with no variables which is also handled by constant term
                pass # The coefficient is already added if needed

            terminos.append("".join(partes_termino) + "".join(variables_part))


        if not terminos:
            return "0" # Represent the zero polynomial

        # Join terms and clean up potential issues from manual '+' '-'
        # The logic for adding signs ensures we don't get " + -" or start with " + " if it's the first term.
        # The individual term strings already contain their sign/spacing relative to the *previous* term.
        # So we just need to join them directly.
        expr = "".join(terminos).strip()

        # Final check for leading sign if it resulted in "+ "
        if expr.startswith("+ "):
            expr = expr[2:]

        # If the expression is just a single negative number (e.g., "- 5"), remove the space
        if expr.startswith("- ") and len(expr) > 2 and all(c.isdigit() or c == '.' for c in expr[2:].replace(" ", "")):
             expr = "-" + expr[2:].replace(" ", "")

        return expr

# Simple test block for PolinomioMultivariable
if __name__ == '__main__':
    print("--- PolinomioMultivariable Tests ---")

    p1 = PolinomioMultivariable.desde_string("2x^2 + 3x - 1")
    p2 = PolinomioMultivariable.desde_string("x - 5")
    p3 = PolinomioMultivariable.desde_string("4")

    print(f"p1: {p1}")
    print(f"p2: {p2}")
    print(f"p3: {p3}")

    p_add = p1 + p2
    print(f"p1 + p2: {p_add}")

    p_sub = p1 - p2
    print(f"p1 - p2: {p_sub}")

    p_mul = p1 * p2
    print(f"p1 * p2: {p_mul}")

    p_add_scalar = p1 + 5
    print(f"p1 + 5: {p_add_scalar}")

    p_sub_scalar = p1 - 3
    print(f"p1 - 3: {p_sub_scalar}")

    p_mul_scalar = p1 * 2
    print(f"p1 * 2: {p_mul_scalar}")

    scalar_add_p = 10 + p1
    print(f"10 + p1: {scalar_add_p}")

    scalar_sub_p = 5 - p2
    print(f"5 - p2: {scalar_sub_p}")

    scalar_mul_p = 3 * p1
    print(f"3 * p1: {scalar_mul_p}")

    eval_p1 = p1.evaluar(x=2)
    print(f"Eval p1(2): {eval_p1}") # Expected: 2*4 + 3*2 - 1 = 8 + 6 - 1 = 13

    p_multivar = PolinomioMultivariable.desde_string("3x^2y - 2xz + 5y^3 + 7")
    print(f"Multivariable: {p_multivar}")
    eval_multivar = p_multivar.evaluar(x=1, y=2, z=3)
    print(f"Eval multivar(x=1, y=2, z=3): {eval_multivar}") # Expected: 3*(1)^2*2 - 2*1*3 + 5*2^3 + 7 = 6 - 6 + 40 + 7 = 47

    deriv_x = p_multivar.derivada_parcial('x')
    print(f"Derivada parcial respecto a x: {deriv_x}") # Expected: 6xy - 2z

    deriv_y = p_multivar.derivada_parcial('y')
    print(f"Derivada parcial respecto a y: {deriv_y}") # Expected: 3x^2 + 15y^2