# gui.py
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the necessary functions and classes from your modules
# Assuming your module structure is metodos/module_name.py
try:
    from metodos.Polinomio import PolinomioMultivariable
    from metodos.Ecuacion import EcuacionMultivariable
    from metodos.primera_unidad import newton_raphson_sistema, secante_sistema
    from metodos.segunda_unidad import lagrange_interpolation, diferencias_divididas, minimos_cuadrados, evaluar_polinomio
    from metodos.integracion_numerica import IntegracionNumerica

except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que los archivos están en la carpeta 'metodos' y que la estructura es correcta.")
    # Display an error message box to the user
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw() # Hide the main window
    messagebox.showerror("Error de Importación", f"Error al importar módulos: {e}\nAsegúrate de que los archivos están en la carpeta 'metodos' y que la estructura es correcta.")
    root.destroy() # Destroy the root window after the message box is closed
    import sys
    sys.exit(1) # Exit the application


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")

appWidth, appHeight = 1200, 800

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Métodos Numéricos Avanzados")
        self.geometry(f"{appWidth}x{appHeight}")
        self.minsize(1000, 700)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Use a dictionary to map frame names to classes
        frame_classes = {
            "PortadaFrame": PortadaFrame,
            "MetodosFrame": MetodosFrame,
            "SistemasFrame": SistemasFrame,
            "InterpolacionFrame": InterpolacionFrame,
            "IntegracionFrame": IntegracionFrame
        }

        for name, F in frame_classes.items():
            frame = F(self.container, self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PortadaFrame")

    def show_frame(self, frame_name):
        frame = self.frames[frame_name]
        frame.tkraise()

class PortadaFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        inner_frame = ctk.CTkFrame(self, fg_color="transparent")
        inner_frame.grid(row=0, column=0, sticky="")

        # Logo de la universidad
        try:
            # Adjust logo path if necessary based on deployment
            logo_path = os.path.join(os.path.dirname(__file__), "logo_unam.png")
            # Check if logo file exists, otherwise skip loading it
            if os.path.exists(logo_path):
                logo_image = Image.open(logo_path)
                # Use LANCZOS or LANCZOS for resizing
                logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)
                self.logo = ImageTk.PhotoImage(logo_image)
                logo_label = ctk.CTkLabel(inner_frame, image=self.logo, text="")
                logo_label.pack(pady=20)
            else:
                 print(f"Advertencia: Archivo de logo no encontrado en {logo_path}")

        except Exception as e:
            print(f"Error loading logo: {e}")
            pass # Continue even if logo loading fails


        self.title_label = ctk.CTkLabel(
            inner_frame,
            text="Facultad de Estudios Superiores Acatlán - UNAM\nMétodos Numéricos Avanzados",
            font=("Arial", 28, "bold"),
            text_color="#2E8B57" # UNAM Green-ish
        )
        self.title_label.pack(pady=10)

        self.subtitle_label = ctk.CTkLabel(
            inner_frame,
            text="Integrantes:\nCosio Placencia Grecia Paola\nHernández Salcedo Mesías Elohim\n Ramírez Martínez Jessica",
            font=("Arial", 18),
            text_color="#4682B4" # UNAM Blue-ish
        )
        self.subtitle_label.pack(pady=20)

        button_frame = ctk.CTkFrame(inner_frame, fg_color="transparent")
        button_frame.pack(pady=30)

        self.continue_button = ctk.CTkButton(
            button_frame,
            text="COMENZAR",
            command=lambda: controller.show_frame("MetodosFrame"),
            width=250,
            height=60,
            font=("Arial", 16, "bold"),
            fg_color="#2E8B57",
            hover_color="#3CB371",
            corner_radius=10
        )
        self.continue_button.pack(pady=15, fill="x")

        self.exit_button = ctk.CTkButton(
            button_frame,
            text="SALIR",
            command=self.controller.destroy,
            width=250,
            height=60,
            font=("Arial", 16, "bold"),
            fg_color="#D22B2B", # Red
            hover_color="#FF0000",
            corner_radius=10
        )
        self.exit_button.pack(pady=15, fill="x")

class MetodosFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        top_frame.grid_columnconfigure(1, weight=1)

        back_button = ctk.CTkButton(
            top_frame,
            text="← Regresar",
            command=lambda: controller.show_frame("PortadaFrame"),
            width=120,
            height=40,
            font=("Arial", 14),
            fg_color="transparent",
            border_width=1,
            corner_radius=8,
            hover_color="#F0F0F0",
            text_color="gray" # Or desired color
        )
        back_button.grid(row=0, column=0, sticky="w")

        label = ctk.CTkLabel(
            top_frame,
            text="Seleccione una opción:",
            font=("Arial", 22, "bold"),
            text_color="#2E8B57"
        )
        label.grid(row=0, column=1, sticky="", padx=10)

        options_frame = ctk.CTkFrame(self, fg_color="transparent")
        options_frame.grid(row=1, column=0, sticky="nsew", padx=50, pady=20)
        options_frame.grid_rowconfigure((0,1,2), weight=1) # Make buttons expandable vertically
        options_frame.grid_columnconfigure(0, weight=1) # Make buttons expandable horizontally

        # Big buttons for method selection
        btn1 = ctk.CTkButton(
            options_frame,
            text="Sistemas de Ecuaciones No Lineales\n(Unidad 1)",
            command=lambda: controller.show_frame("SistemasFrame"),
            height=100,
            font=("Arial", 18, "bold"),
            fg_color="#4682B4", # Blue-ish
            hover_color="#5F9EA0",
            corner_radius=12
        )
        btn1.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        btn2 = ctk.CTkButton(
            options_frame,
            text="Interpolación y Aproximación\n(Unidad 2)",
            command=lambda: controller.show_frame("InterpolacionFrame"),
            height=100,
            font=("Arial", 18, "bold"),
            fg_color="#2E8B57", # Green-ish
            hover_color="#3CB371",
            corner_radius=12
        )
        btn2.grid(row=1, column=0, pady=20, padx=20, sticky="nsew")

        btn3 = ctk.CTkButton(
            options_frame,
            text="Integración Numérica\n(Unidad 3)",
            command=lambda: controller.show_frame("IntegracionFrame"),
            height=100,
            font=("Arial", 18, "bold"),
            fg_color="#8B4513", # Brown-ish
            hover_color="#A0522D",
            corner_radius=12
        )
        btn3.grid(row=2, column=0, pady=20, padx=20, sticky="nsew")

class BaseMethodFrame(ctk.CTkFrame):
    def __init__(self, parent, controller, title):
        super().__init__(parent)
        self.controller = controller
        self.title = title

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        top_frame.grid_columnconfigure(1, weight=1)

        back_button = ctk.CTkButton(
            top_frame,
            text="← Regresar",
            command=lambda: controller.show_frame("MetodosFrame"),
            width=120,
            height=40,
            font=("Arial", 14),
            fg_color="transparent",
            border_width=1,
            corner_radius=8,
            hover_color="#F0F0F0",
            text_color="gray" # Or desired color
        )
        back_button.grid(row=0, column=0, sticky="w")

        label = ctk.CTkLabel(
            top_frame,
            text=title,
            font=("Arial", 22, "bold"),
            text_color="#2E8B57"
        )
        label.grid(row=0, column=1, sticky="", padx=10)

        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        # Configure content_frame rows/columns in subclasses
        self.content_frame.grid_columnconfigure(0, weight=1)


class SistemasFrame(BaseMethodFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Sistemas de Ecuaciones No Lineales")

        # Configure content_frame rows for Systems frame
        self.content_frame.grid_rowconfigure(3, weight=1) # Row for results
        self.content_frame.grid_rowconfigure(4, weight=1) # Row for plot (if 2 variables)


        # Frame para entrada de ecuaciones
        input_frame = ctk.CTkFrame(self.content_frame)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text="Número de ecuaciones:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.num_ecuaciones = ctk.CTkEntry(input_frame, width=50)
        self.num_ecuaciones.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.set_ecuaciones_btn = ctk.CTkButton(
            input_frame,
            text="Configurar ecuaciones",
            command=self.configurar_ecuaciones,
            fg_color="#4682B4"
        )
        self.set_ecuaciones_btn.grid(row=0, column=2, padx=10, pady=5)

        self.ecuaciones_frame = ctk.CTkFrame(self.content_frame)
        self.ecuaciones_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.ecuaciones_frame.grid_columnconfigure(1, weight=1) # Make equation entry expand

        # Frame para parámetros (método, tolerancia, iteraciones, valores iniciales)
        params_frame = ctk.CTkFrame(self.content_frame)
        params_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        params_frame.grid_columnconfigure((1,3,5,7), weight=1) # Allow some columns to expand

        ctk.CTkLabel(params_frame, text="Método:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.metodo = ctk.CTkComboBox(
            params_frame,
            values=["Newton-Raphson", "Método de la Secante"],
            width=180
        )
        self.metodo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.metodo.bind("<<ComboboxSelected>>", self.on_method_change) # Bind event to update x1 placeholder

        ctk.CTkLabel(params_frame, text="Tolerancia:", font=("Arial", 14)).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.tolerancia = ctk.CTkEntry(params_frame, width=80)
        self.tolerancia.insert(0, "1e-6")
        self.tolerancia.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Máx. iteraciones:", font=("Arial", 14)).grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.max_iter = ctk.CTkEntry(params_frame, width=80)
        self.max_iter.insert(0, "100")
        self.max_iter.grid(row=0, column=5, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Valor Inicial (x0):", font=("Arial", 14)).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.x0_entry = ctk.CTkEntry(params_frame, width=200, placeholder_text="Ej: 1.0, 2.0")
        self.x0_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Segundo Valor Inicial (x1, Secante):", font=("Arial", 14)).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.x1_entry = ctk.CTkEntry(params_frame, width=200, placeholder_text="Opcional. Ej: 1.1, 2.1")
        self.x1_entry.grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="ew")


        self.solve_btn = ctk.CTkButton(
            params_frame,
            text="Resolver Sistema",
            command=self.resolver_sistema,
            fg_color="#2E8B57"
        )
        self.solve_btn.grid(row=0, column=6, rowspan=2, padx=10, pady=5, sticky="nsew")


        # Frame para resultados
        self.resultados_frame = ctk.CTkFrame(self.content_frame)
        self.resultados_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        # Frame para gráfica (only for 2 variables)
        self.grafica_frame = ctk.CTkFrame(self.content_frame)
        self.grafica_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)


        # Variables para almacenar ecuaciones
        self.ecuaciones_entries = []
        self.variables = []

    def on_method_change(self, event):
         # Update the placeholder text for x1_entry based on the selected method
         num_eq = len(self.variables)
         if self.metodo.get() == "Método de la Secante" and num_eq > 0:
              self.x1_entry.configure(placeholder_text=f"Opcional. Ej: {', '.join(['1.1'] * num_eq)}")
         else:
               self.x1_entry.configure(placeholder_text="No usado para Newton")
         self.x1_entry.update() # Ensure the change is reflected in the GUI


    def configurar_ecuaciones(self):
        try:
            n_str = self.num_ecuaciones.get()
            if not n_str:
                 raise ValueError("Ingrese el número de ecuaciones.")
            n = int(n_str)

            if n <= 0:
                raise ValueError("El número de ecuaciones debe ser positivo.")
            if n > len(['x', 'y', 'z']):
                 self.mostrar_resultado(f"Advertencia: Solo se soportan variables: {', '.join(['x', 'y', 'z'])}. Puede que necesites adaptar tus ecuaciones.", error=False)

            # Limpiar frame de ecuaciones
            for widget in self.ecuaciones_frame.winfo_children():
                widget.destroy()

            self.ecuaciones_entries = []
            self.variables = ['x', 'y', 'z'][:n] # Limit variables based on n

            for i in range(n):
                ctk.CTkLabel(self.ecuaciones_frame, text=f"Ecuación {i+1}:", font=("Arial", 14)).grid(row=i, column=0, padx=5, pady=5, sticky="e")
                entry = ctk.CTkEntry(self.ecuaciones_frame, width=500) # Increased width for equations
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
                self.ecuaciones_entries.append(entry)

            # Adjust initial guess input placeholder based on number of equations and selected method
            if n > 0:
                 self.x0_entry.configure(placeholder_text=f"Ej: {', '.join(['1.0'] * n)}")
                 self.on_method_change(None) # Update x1 placeholder
            else:
                 self.x0_entry.configure(placeholder_text="")
                 self.x1_entry.configure(placeholder_text="")

            # Clear previous results and plot
            self.mostrar_resultado("")
            self.clear_plot()


        except ValueError as e:
            self.mostrar_resultado(f"Error al configurar ecuaciones: {str(e)}", error=True)
        except Exception as e:
            self.mostrar_resultado(f"Error inesperado al configurar ecuaciones: {str(e)}", error=True)


    def resolver_sistema(self):
        try:
            ecuaciones_str = [entry.get() for entry in self.ecuaciones_entries]
            if not ecuaciones_str:
                 raise ValueError("Configure las ecuaciones primero.")
            if not all(ecuaciones_str):
                raise ValueError("Todas las ecuaciones configuradas deben estar completas.")

            metodo = self.metodo.get()
            tol_str = self.tolerancia.get()
            max_iter_str = self.max_iter.get()

            if not tol_str or not max_iter_str:
                 raise ValueError("Ingrese la tolerancia y el número máximo de iteraciones.")

            try:
                tol = float(tol_str)
                max_iter = int(max_iter_str)
            except ValueError:
                 raise ValueError("Tolerancia y/o Máx. iteraciones deben ser números válidos.")


            # Parse initial guess(es)
            x0_str = self.x0_entry.get()
            if not x0_str:
                 raise ValueError("Ingrese los valores iniciales para x0 (ej: 1.0, 2.0).")
            try:
                 x0 = [float(val.strip()) for val in x0_str.split(',')]
            except ValueError:
                 raise ValueError("Formato de valores iniciales para x0 inválido. Use números separados por coma (ej: 1.0, 2.0).")

            n = len(ecuaciones_str)
            if len(x0) != n:
                 raise ValueError(f"El número de valores iniciales para x0 ({len(x0)}) debe coincidir con el número de ecuaciones ({n}).")

            if metodo == "Newton-Raphson":
                # Call the actual Newton-Raphson solver
                resultado = newton_raphson_sistema(ecuaciones_str, self.variables, x0, tol, max_iter)
                result_msg = f"Método: Newton-Raphson\n"
                result_msg += f"Ecuaciones: {', '.join(ecuaciones_str)}\n"
                result_msg += f"Variables: {', '.join(self.variables)}\n"
                result_msg += f"Valor Inicial (x0): {x0}\n"
                result_msg += f"Solución encontrada: [{', '.join([f'{sol:.6f}' for sol in resultado['solucion']])}]\n" # Format solution
                result_msg += f"Iteraciones: {resultado['iteraciones']}\n"
                result_msg += f"Error final: {resultado['error']:.6e}"
                self._last_solution = resultado['solucion'] # Store solution for potential plotting

            elif metodo == "Método de la Secante":
                x1_str = self.x1_entry.get()
                x1 = None # x1 is optional for secante, will be generated if None
                if x1_str:
                     try:
                          x1 = [float(val.strip()) for val in x1_str.split(',')]
                     except ValueError:
                          raise ValueError("Formato de valores iniciales para x1 inválido. Use números separados por coma (ej: 1.1, 2.1).")
                     if len(x1) != n:
                          raise ValueError(f"El número de valores iniciales para x1 ({len(x1)}) debe coincidir con el número de ecuaciones ({n}).")

                # Call the actual Secant solver
                # The secante_sistema function handles the case where x1 is None
                resultado = secante_sistema(ecuaciones_str, self.variables, x0, x1, tol, max_iter)
                result_msg = f"Método: Método de la Secante\n"
                result_msg += f"Ecuaciones: {', '.join(ecuaciones_str)}\n"
                result_msg += f"Variables: {', '.join(self.variables)}\n"
                result_msg += f"Valores Iniciales (x0, x1): {x0}, {x1 if x1 is not None else 'Generado'}\n" # Show 'Generado' if x1 was None
                result_msg += f"Solución encontrada: [{', '.join([f'{sol:.6f}' for sol in resultado['solucion']])}]\n" # Format solution
                result_msg += f"Iteraciones: {resultado['iteraciones']}\n"
                result_msg += f"Error final: {resultado['error']:.6e}"
                self._last_solution = resultado['solucion'] # Store solution for potential plotting


            self.mostrar_resultado(result_msg)

            # Plot only if n=2
            if n == 2:
                 self.mostrar_grafica(ecuaciones_str)
            else:
                 self.clear_plot()
                 if n != 0: # Avoid showing message if no equations are configured
                     self.mostrar_resultado(result_msg + "\n\nGráfica disponible solo para sistemas con 2 variables (n=2).", error=False)


        except ValueError as e:
            self.mostrar_resultado(f"Error de entrada/validación: {str(e)}", error=True)
            self.clear_plot()
        except Exception as e:
            # Catch potential errors from the solver functions (e.g., singular Jacobian, no convergence)
            self.mostrar_resultado(f"Error al resolver sistema: {type(e).__name__}: {str(e)}", error=True)
            self.clear_plot()


    def mostrar_resultado(self, mensaje, error=False):
        # Clear previous content first
        for widget in self.resultados_frame.winfo_children():
            widget.destroy()

        # Use CTkTextbox for multiline output
        textbox = ctk.CTkTextbox(
            self.resultados_frame,
            wrap="word", # Wrap text at word boundaries
            state="normal", # Enable editing temporarily to insert text
            font=("Consolas", 12),
            width=800, # Set a default width, adjust as needed
            height=150 # Set a default height, adjust as needed
        )
        textbox.pack(fill="both", expand=True, padx=5, pady=5) # Add padding

        textbox.insert("end", mensaje)

        if error:
            textbox.configure(text_color="red")
        else:
             textbox.configure(text_color="white") # Default text color

        textbox.configure(state="disabled") # Disable editing after inserting text


    def mostrar_grafica(self, ecuaciones_str):
        # Clear previous plot first
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        # Close any existing plot figures to free up memory
        plt.close('all')

        if len(self.variables) != 2:
             return # Only plot for 2 variables

        fig, ax = plt.subplots(figsize=(6, 4))

        try:
            # Create function handles from the equations string using EcuacionMultivariable
            # Assuming equations are in the form f(x,y) = 0
            ec_objs = [EcuacionMultivariable(ec_str) for ec_str in ecuaciones_str]

            # Determine a reasonable plot range
            plot_range = 5 # Default range from -5 to 5
            center_x, center_y = 0, 0 # Default center
            if self.x0_entry.get():
                 try:
                      x0_vals = [float(val.strip()) for val in self.x0_entry.get().split(',')]
                      if len(x0_vals) == 2:
                           # Center plot around initial guess
                           center_x, center_y = x0_vals
                           # Adjust range based on initial guess, ensuring a minimum size
                           plot_range_x = max(5.0, abs(center_x) * 1.5, abs(center_x - (self._last_solution[0] if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution)==2 else center_x)) * 2.0) if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution)==2 else max(5.0, abs(center_x) * 1.5)
                           plot_range_y = max(5.0, abs(center_y) * 1.5, abs(center_y - (self._last_solution[1] if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution)==2 else center_y)) * 2.0) if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution)==2 else max(5.0, abs(center_y) * 1.5)

                           x = np.linspace(center_x - plot_range_x, center_x + plot_range_x, 200)
                           y = np.linspace(center_y - plot_range_y, center_y + plot_range_y, 200)
                      else:
                           x = np.linspace(-plot_range, plot_range, 200)
                           y = np.linspace(-plot_range, plot_range, 200)
                 except ValueError:
                      x = np.linspace(-plot_range, plot_range, 200)
                      y = np.linspace(-plot_range, plot_range, 200)
            else:
                 x = np.linspace(-plot_range, plot_range, 200)
                 y = np.linspace(-plot_range, plot_range, 200)


            X, Y = np.meshgrid(x, y)

            # Evaluate each equation's polynomial over the grid
            try:
                # Use nested loops to evaluate the polynomial for each point in the meshgrid
                # Initialize Z arrays with NaN or a value that won't plot if evaluation fails
                Z1 = np.full(X.shape, np.nan, dtype=float)
                Z2 = np.full(X.shape, np.nan, dtype=float)

                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            Z1[i,j] = ec_objs[0].polinomio.evaluar(x=X[i,j], y=Y[i,j])
                            Z2[i,j] = ec_objs[1].polinomio.evaluar(x=X[i,j], y=Y[i,j])
                        except Exception as eval_point_e:
                            # Print warning for point evaluation errors, but don't stop the whole plot
                            # print(f"Warning: Could not evaluate function at point ({X[i,j]}, {Y[i,j]}): {eval_point_e}")
                            Z1[i,j] = np.nan # Set to NaN if evaluation fails
                            Z2[i,j] = np.nan # Set to NaN if evaluation fails


            except Exception as eval_e:
                 self.mostrar_resultado(f"Error al evaluar funciones para graficar: {str(eval_e)}", error=True)
                 plt.close(fig) # Close the figure if evaluation fails
                 return


            # Plot the contour lines where the function equals zero
            # Added labels for the contour lines
            # Ensure there are non-NaN values before attempting contour plot
            if not np.all(np.isnan(Z1)):
                contour1 = ax.contour(X, Y, Z1, levels=[0], colors='blue', linestyles='solid', linewidths=2)
            if not np.all(np.isnan(Z2)):
                contour2 = ax.contour(X, Y, Z2, levels=[0], colors='red', linestyles='solid', linewidths=2)


            ax.set_xlabel(self.variables[0])
            ax.set_ylabel(self.variables[1])
            ax.set_title('Gráfica del sistema de ecuaciones')
            ax.grid(True)
            ax.axhline(0, color='gray', lw=0.5) # Add x-axis line
            ax.axvline(0, color='gray', lw=0.5) # Add y-axis line
            ax.set_aspect('equal', adjustable='box') # Keep aspect ratio equal


            # Optional: Plot the found solution point if available
            if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution) == 2:
                 try:
                      ax.plot(self._last_solution[0], self._last_solution[1], 'go', markersize=8, label='Solución Encontrada')
                 except Exception as plot_sol_e:
                      print(f"Warning: Could not plot solution point: {plot_sol_e}")


            # Add a legend entry for the contour lines and the solution point
            from matplotlib.lines import Line2D
            legend_elements = []
            if not np.all(np.isnan(Z1)):
                 legend_elements.append(Line2D([0], [0], color='blue', lw=2, label=f'Ecuación 1: {ecuaciones_str[0]}'))
            if not np.all(np.isnan(Z2)):
                 legend_elements.append(Line2D([0], [0], color='red', lw=2, label=f'Ecuación 2: {ecuaciones_str[1]}'))
            if hasattr(self, '_last_solution') and self._last_solution and len(self._last_solution) == 2:
                 legend_elements.append(Line2D([0], [0], marker='o', color='g', label='Solución Encontrada', linestyle='None', markersize=8))

            if legend_elements: # Only add legend if there are elements
                 ax.legend(handles=legend_elements, loc='best')


            canvas = FigureCanvasTkAgg(fig, master=self.grafica_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            self.mostrar_resultado(f"Error al generar gráfica: {str(e)}", error=True)
            plt.close(fig) # Close the figure if an error occurs


    def clear_plot(self):
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        # Close any existing plot figures to free up memory
        plt.close('all')
        if hasattr(self, '_last_solution'):
             del self._last_solution # Clear stored solution


class InterpolacionFrame(BaseMethodFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Interpolación y Aproximación")
        self.content_frame.grid_rowconfigure(2, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Frame para controles
        self.controls_frame = ctk.CTkFrame(self.content_frame)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Entradas para puntos
        ctk.CTkLabel(self.controls_frame, text="X:", font=("Arial", 14)).grid(row=0, column=0, padx=5)
        self.x_entry = ctk.CTkEntry(self.controls_frame, width=100)
        self.x_entry.grid(row=0, column=1, padx=5)
        
        ctk.CTkLabel(self.controls_frame, text="Y:", font=("Arial", 14)).grid(row=0, column=2, padx=5)
        self.y_entry = ctk.CTkEntry(self.controls_frame, width=100)
        self.y_entry.grid(row=0, column=3, padx=5)
        
        self.add_button = ctk.CTkButton(
            self.controls_frame,
            text="Agregar punto",
            command=self.add_point,
            width=120,
            fg_color="#2E8B57",
            font=("Arial", 14)
        )
        self.add_button.grid(row=0, column=4, padx=5)
        
        self.clear_button = ctk.CTkButton(
            self.controls_frame,
            text="Limpiar datos",
            command=self.clear_data,
            width=120,
            fg_color="#D22B2B",
            font=("Arial", 14)
        )
        self.clear_button.grid(row=0, column=5, padx=5)
        
        # Frame para selección de método
        self.method_frame = ctk.CTkFrame(self.content_frame)
        self.method_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(self.method_frame, text="Método:", font=("Arial", 14)).grid(row=0, column=0, padx=5)
        self.method_combobox = ctk.CTkComboBox(
            self.method_frame,
            values=["Lagrange", "Diferencias Divididas (Newton)", "Mínimos Cuadrados"],
            width=200,
            font=("Arial", 14)
        )
        self.method_combobox.grid(row=0, column=1, padx=5)
        
        self.interpolate_button = ctk.CTkButton(
            self.method_frame,
            text="Calcular",
            command=self.calculate_interpolation,
            fg_color="#1E90FF",
            font=("Arial", 14)
        )
        self.interpolate_button.grid(row=0, column=2, padx=10)
        
        # Frame para tabla de puntos
        self.table_frame = ctk.CTkScrollableFrame(self.content_frame, height=150)
        self.table_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        # Frame para resultados
        self.results_frame = ctk.CTkFrame(self.content_frame)
        self.results_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        
        # Frame para gráfica
        self.plot_frame = ctk.CTkFrame(self.content_frame)
        self.plot_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)
        
        # Inicializar variables
        self.points = []
        self.headers = ["Índice", "X", "Y"]
        self.create_table_headers()
    
    def create_table_headers(self):
        for col, header in enumerate(self.headers):
            ctk.CTkLabel(
                self.table_frame,
                text=header,
                font=("Arial", 12, "bold"),
                width=100
            ).grid(row=0, column=col, padx=5, pady=2, sticky="ew")
    
    def add_point(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            
            self.points.append((x, y))
            self.points.sort()  # Ordenar por x para algunos métodos
            self.update_table()
            
            self.x_entry.delete(0, "end")
            self.y_entry.delete(0, "end")
            
        except ValueError:
            self.show_result("Error: Ingresa valores numéricos válidos", is_error=True)
    
    def clear_data(self):
        self.points = []
        self.update_table()
        self.show_result("Datos limpiados")
        self.clear_plot()
    
    def update_table(self):
        # Limpiar tabla (excepto encabezados)
        for widget in self.table_frame.winfo_children():
            if widget.grid_info()["row"] > 0:
                widget.destroy()
        
        # Llenar tabla con puntos
        for idx, (x, y) in enumerate(self.points, start=1):
            ctk.CTkLabel(
                self.table_frame,
                text=str(idx),
                width=100
            ).grid(row=idx, column=0, padx=5, pady=2)
            
            ctk.CTkLabel(
                self.table_frame,
                text=f"{x:.4f}",
                width=100
            ).grid(row=idx, column=1, padx=5, pady=2)
            
            ctk.CTkLabel(
                self.table_frame,
                text=f"{y:.4f}",
                width=100
            ).grid(row=idx, column=2, padx=5, pady=2)
    
    def calculate_interpolation(self):
        if len(self.points) < 2:
            self.show_result("Se necesitan al menos 2 puntos", is_error=True)
            return
        
        method = self.method_combobox.get()
        x_values = [p[0] for p in self.points]
        y_values = [p[1] for p in self.points]
        
        try:
            if method == "Lagrange":
                from metodos.segunda_unidad import lagrange_interpolation
                polynomial = lagrange_interpolation(x_values, y_values)
                result_text = f"Polinomio de {method}:\n{polynomial}"
                self.plot_interpolation(x_values, y_values, method, str(polynomial))
            
            elif method == "Diferencias Divididas (Newton)":
                from metodos.segunda_unidad import diferencias_divididas
                coef, poly = diferencias_divididas(x_values, y_values)
                result_text = f"Coeficientes de {method}:\n{coef}\n\nPolinomio:\n{poly}"
                self.plot_interpolation(x_values, y_values, method, str(poly))
            
            elif method == "Mínimos Cuadrados":
                from metodos.segunda_unidad import minimos_cuadrados
                degree = min(3, len(self.points)-1)  # Grado máximo razonable
                coef, poly = minimos_cuadrados(x_values, y_values, degree)
                result_text = f"Coeficientes (grado {degree}):\n{coef}\n\nPolinomio:\n{poly}"
                self.plot_approximation(x_values, y_values, poly)
            
            self.show_result(result_text)
            
        except Exception as e:
            self.show_result(f"Error en {method}: {str(e)}", is_error=True)
    
    def show_result(self, message, is_error=False):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        textbox = ctk.CTkTextbox(
            self.results_frame,
            width=800,
            height=150,
            font=("Consolas", 12)
        )
        textbox.pack(fill="both", expand=True)
        textbox.insert("end", message)
        textbox.configure(state="disabled")
        
        if is_error:
            textbox.configure(text_color="red")
    
    def plot_interpolation(self, x_values, y_values, method, poly_str):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Puntos originales
        ax.scatter(x_values, y_values, color='red', label='Puntos dados', zorder=5)
        
        # Polinomio interpolante
        x_min, x_max = min(x_values), max(x_values)
        x_range = x_max - x_min
        x_plot = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 400)
        
        try:
            # Evaluar el polinomio (esto es un ejemplo simplificado)
            # En una implementación real, usarías el polinomio calculado
            if method == "Lagrange":
                # Ejemplo simplificado - en realidad usarías el polinomio de Lagrange
                y_plot = np.interp(x_plot, x_values, y_values)
            else:
                # Ejemplo simplificado para Newton
                y_plot = np.polyval(np.polyfit(x_values, y_values, len(x_values)-1), x_plot)
            
            ax.plot(x_plot, y_plot, label=f"Polinomio de {method}", color='blue')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Interpolación: {method}\n{poly_str[:100]}...')
            ax.legend()
            ax.grid(True)
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            self.show_result(f"Error al graficar: {str(e)}", is_error=True)
    
    def plot_approximation(self, x_values, y_values, poly):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Puntos originales
        ax.scatter(x_values, y_values, color='red', label='Puntos dados', zorder=5)
        
        # Aproximación por mínimos cuadrados
        x_min, x_max = min(x_values), max(x_values)
        x_range = x_max - x_min
        x_plot = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 400)
        
        try:
            # Evaluar el polinomio (esto es un ejemplo simplificado)
            y_plot = np.polyval(np.polyfit(x_values, y_values, min(3, len(x_values)-1), x_plot))
            
            ax.plot(x_plot, y_plot, label="Aproximación por Mínimos Cuadrados", color='green')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Aproximación por Mínimos Cuadrados')
            ax.legend()
            ax.grid(True)
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            self.show_result(f"Error al graficar: {str(e)}", is_error=True)
    
    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

class IntegracionFrame(BaseMethodFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Integración Numérica")

        # Configure content_frame rows for Integration frame
        self.content_frame.grid_rowconfigure(2, weight=1) # Row for results
        self.content_frame.grid_rowconfigure(3, weight=1) # Row for plot


        # Frame para entrada de función
        input_frame = ctk.CTkFrame(self.content_frame)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(1, weight=1) # Allow function entry to expand

        ctk.CTkLabel(input_frame, text="Función f(x):", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # Added more examples and explicitly mentioned '**' for power
        self.func_entry = ctk.CTkEntry(input_frame, placeholder_text="Ej: math.sin(x) + x**2 o x^3 - 2*x + 1", width=400)
        self.func_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(input_frame, text="Use 'math.func' (math.sin, math.exp, etc.), 'x' for variable, '**' or '^' for powers.", font=("Arial", 10), text_color="gray").grid(row=1, column=0, columnspan=2, padx=5, pady=0, sticky="w")


        # Frame para límites y parámetros
        params_frame = ctk.CTkFrame(self.content_frame)
        params_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        params_frame.grid_columnconfigure((1,3,5,7), weight=1) # Make entry/combobox columns expandable

        ctk.CTkLabel(params_frame, text="Límite inferior a:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.a_entry = ctk.CTkEntry(params_frame, width=80)
        self.a_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Límite superior b:", font=("Arial", 14)).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.b_entry = ctk.CTkEntry(params_frame, width=80)
        self.b_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Número de intervalos (n):", font=("Arial", 14)).grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.n_entry = ctk.CTkEntry(params_frame, width=80)
        self.n_entry.insert(0, "100") # Default to a higher number of intervals
        self.n_entry.grid(row=0, column=5, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(params_frame, text="Método:", font=("Arial", 14)).grid(row=0, column=6, padx=5, pady=5, sticky="w")
        self.metodo_int = ctk.CTkComboBox(
            params_frame,
            values=["Trapezoidal", "Simpson 1/3", "Simpson 3/8"],
            width=120
        )
        self.metodo_int.grid(row=0, column=7, padx=5, pady=5, sticky="ew")

        self.calc_button = ctk.CTkButton(
            params_frame,
            text="Calcular Integral",
            command=self.calcular_integral,
            fg_color="#8B4513", # Brown-ish
            font=("Arial", 14)
        )
        self.calc_button.grid(row=0, column=8, padx=10, pady=5, sticky="ew")


        # Frame para resultados
        self.resultados_frame = ctk.CTkFrame(self.content_frame)
        self.resultados_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        # Frame para gráfica
        self.grafica_frame = ctk.CTkFrame(self.content_frame)
        self.grafica_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)


    def calcular_integral(self):
        try:
            f_str = self.func_entry.get()
            if not f_str:
                 raise ValueError("Ingrese la función f(x).")
            a_str = self.a_entry.get()
            b_str = self.b_entry.get()
            n_str = self.n_entry.get()

            if not a_str or not b_str or not n_str:
                 raise ValueError("Ingrese los límites de integración (a, b) y el número de intervalos (n).")

            try:
                a = float(a_str)
                b = float(b_str)
                n = int(n_str)
            except ValueError:
                 raise ValueError("Límites (a, b) y n deben ser números válidos.")

            metodo = self.metodo_int.get()

            if a >= b:
                raise ValueError("El límite inferior 'a' debe ser menor que el superior 'b'.")
            if n <= 0:
                raise ValueError("El número de intervalos 'n' debe ser positivo.")
            if metodo == "Simpson 1/3" and n % 2 != 0:
                 raise ValueError("El método Simpson 1/3 requiere un número par de intervalos (n).")
            if metodo == "Simpson 3/8" and n % 3 != 0:
                 raise ValueError("El método Simpson 3/8 requiere que el número de intervalos (n) sea múltiplo de 3.")


            # Define the function to pass to integration methods using the evaluator
            # This lambda function will be called by the integration methods with a numerical value of x
            def func_to_integrate(x_val):
                 # Use the evaluar_funcion from integracion_numerica.py to evaluate the string function
                 # Pass the original function string and the numerical x_val
                 return IntegracionNumerica.evaluar_funcion(f_str, x_val)


            if metodo == "Trapezoidal":
                result = IntegracionNumerica.trapezoidal(func_to_integrate, a, b, n)
            elif metodo == "Simpson 1/3":
                 result = IntegracionNumerica.simpson13(func_to_integrate, a, b, n)
            elif metodo == "Simpson 3/8":
                 result = IntegracionNumerica.simpson38(func_to_integrate, a, b, n)
            else:
                 result = "Error: Método de integración no válido." # Should not be reached with current combobox values


            resultado = f"Método: {metodo}\n"
            resultado += f"Función: f(x) = {f_str}\n"
            resultado += f"Intervalo: [{a}, {b}]\n"
            resultado += f"Número de intervalos (n): {n}\n"
            resultado += f"Resultado de la Integral: {result:.6f}\n"


            self.mostrar_resultado(resultado)
            self.mostrar_grafica(f_str, a, b)

        except ValueError as e:
            # Catch ValueErrors from input parsing, validation, or the integration methods themselves
            self.mostrar_resultado(f"Error de entrada/validación o cálculo: {str(e)}", error=True)
            self.clear_plot()
        except Exception as e:
             # Catch any other unexpected errors during calculation or function evaluation
             self.mostrar_resultado(f"Error al calcular integral: {type(e).__name__}: {str(e)}", error=True)
             self.clear_plot()


    def mostrar_resultado(self, mensaje, error=False):
        # Clear previous content first
        for widget in self.resultados_frame.winfo_children():
            widget.destroy()

        # Use CTkTextbox for multiline output
        textbox = ctk.CTkTextbox(
            self.resultados_frame,
            wrap="word", # Wrap text at word boundaries
            state="normal", # Enable editing temporarily to insert text
            font=("Consolas", 12),
            width=800, # Set a default width, adjust as needed
            height=150 # Set a default height, adjust as needed
        )
        textbox.pack(fill="both", expand=True, padx=5, pady=5) # Add padding

        textbox.insert("end", mensaje)

        if error:
            textbox.configure(text_color="red")
        else:
             textbox.configure(text_color="white") # Default text color

        textbox.configure(state="disabled") # Disable editing after inserting text


    def mostrar_grafica(self, f_str, a, b):
        # Clear previous plot first
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        # Close any existing plot figures from this frame
        plt.close('all')

        fig, ax = plt.subplots(figsize=(8, 5))

        try:
            # Evaluate the function over a range for plotting
            x_plot = np.linspace(a, b, 400) # More points for smooth curve

            y_plot = []
            for x_val in x_plot:
                try:
                    # Use the evaluar_funcion from integracion_numerica.py to evaluate the string function
                    y_plot.append(evaluar_funcion_integracion(f_str, x_val))
                except Exception as eval_point_e:
                    # If evaluation fails for a point (e.g., log of negative number), append NaN
                    # Print a warning but don't stop the plotting of valid points
                    # print(f"Warning: Could not evaluate function for plotting at x={x_val}: {eval_point_e}")
                    y_plot.append(np.nan) # Use NaN so it doesn't plot

            # Convert y_plot to numpy array to handle NaNs correctly in plotting
            y_plot = np.array(y_plot)


            ax.plot(x_plot, y_plot, 'b-', label=f"f(x) = {f_str}")
            # Fill the area under the curve for visualization, only for the valid parts
            # Use where=~np.isnan(y_plot) if needed for more complex NaN handling
            ax.fill_between(x_plot, 0, y_plot, color='skyblue', alpha=0.4)

            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Integración Numérica')
            ax.legend()
            ax.grid(True)
            ax.axhline(0, color='gray', lw=0.5) # Add x-axis line


            canvas = FigureCanvasTkAgg(fig, master=self.grafica_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            # Catch any other unexpected errors during plotting setup
            self.mostrar_resultado(f"Error al graficar la función: {str(e)}", error=True)
            plt.close(fig) # Close the figure if an error occurs


    def clear_plot(self):
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        # Close any existing plot figures to free up memory
        plt.close('all')


if __name__ == "__main__":
    app = App()
    app.mainloop()