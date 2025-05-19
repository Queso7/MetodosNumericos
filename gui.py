import customtkinter as ctk
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        
        for F in (PortadaFrame, MetodosFrame, SistemasFrame, InterpolacionFrame, IntegracionFrame):
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
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
            logo_path = os.path.join(os.path.dirname(__file__), "logo_unam.png")
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((200, 200), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_image)
            logo_label = ctk.CTkLabel(inner_frame, image=self.logo, text="")
            logo_label.pack(pady=20)
        except:
            pass
        
        self.title_label = ctk.CTkLabel(
            inner_frame, 
            text="Facultad de Estudios Superiores Acatlán - UNAM\nMétodos Numéricos Avanzados",
            font=("Arial", 28, "bold"),
            text_color="#2E8B57"
        )
        self.title_label.pack(pady=10)
        
        self.subtitle_label = ctk.CTkLabel(
            inner_frame,
            text="Integrantes:\nCocio Placencia Grecia Paola\nHernández Salcedo Mesías Elohim",
            font=("Arial", 18),
            text_color="#4682B4"
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
            fg_color="#D22B2B", 
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
            hover_color="#F0F0F0"
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
        options_frame.grid_rowconfigure(0, weight=1)
        options_frame.grid_columnconfigure(0, weight=1)
        
        # Botones grandes para selección de método
        btn1 = ctk.CTkButton(
            options_frame,
            text="Sistemas de Ecuaciones No Lineales\n(Unidad 1)",
            command=lambda: controller.show_frame("SistemasFrame"),
            height=100,
            font=("Arial", 18, "bold"),
            fg_color="#4682B4",
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
            fg_color="#2E8B57",
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
            fg_color="#8B4513",
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
            hover_color="#F0F0F0"
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
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

class SistemasFrame(BaseMethodFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Sistemas de Ecuaciones No Lineales")
        
        self.content_frame.grid_rowconfigure(1, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
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
        
        # Frame para parámetros y resultados
        params_frame = ctk.CTkFrame(self.content_frame)
        params_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(params_frame, text="Método:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5)
        self.metodo = ctk.CTkComboBox(
            params_frame,
            values=["Newton-Raphson", "Método de la Secante"],
            width=180
        )
        self.metodo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(params_frame, text="Tolerancia:", font=("Arial", 14)).grid(row=0, column=2, padx=5, pady=5)
        self.tolerancia = ctk.CTkEntry(params_frame, width=80)
        self.tolerancia.insert(0, "1e-6")
        self.tolerancia.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkLabel(params_frame, text="Máx. iteraciones:", font=("Arial", 14)).grid(row=0, column=4, padx=5, pady=5)
        self.max_iter = ctk.CTkEntry(params_frame, width=80)
        self.max_iter.insert(0, "100")
        self.max_iter.grid(row=0, column=5, padx=5, pady=5)
        
        self.solve_btn = ctk.CTkButton(
            params_frame,
            text="Resolver Sistema",
            command=self.resolver_sistema,
            fg_color="#2E8B57"
        )
        self.solve_btn.grid(row=0, column=6, padx=10, pady=5)
        
        # Frame para resultados
        self.resultados_frame = ctk.CTkFrame(self.content_frame)
        self.resultados_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        
        # Frame para gráfica
        self.grafica_frame = ctk.CTkFrame(self.content_frame)
        self.grafica_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)
        
        # Variables para almacenar ecuaciones
        self.ecuaciones_entries = []
        self.variables = []
        
    def configurar_ecuaciones(self):
        try:
            n = int(self.num_ecuaciones.get())
            if n <= 0:
                raise ValueError("Debe ser un número positivo")
            
            # Limpiar frame de ecuaciones
            for widget in self.ecuaciones_frame.winfo_children():
                widget.destroy()
            
            self.ecuaciones_entries = []
            self.variables = ['x', 'y', 'z'][:n]  # Usamos x, y, z como variables
            
            for i in range(n):
                ctk.CTkLabel(self.ecuaciones_frame, text=f"Ecuación {i+1}:", font=("Arial", 14)).grid(row=i, column=0, padx=5, pady=5, sticky="e")
                entry = ctk.CTkEntry(self.ecuaciones_frame, width=300)
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
                self.ecuaciones_entries.append(entry)
                
                # Mostrar ejemplo
                if i == 0:
                    example = "Ej: x^2 + y^2 - 4 = 0"
                elif i == 1:
                    example = "Ej: x^2 - y = 0"
                else:
                    example = "Ej: x + y + z = 0"
                    
                ctk.CTkLabel(self.ecuaciones_frame, text=example, font=("Arial", 12), text_color="gray").grid(row=i, column=2, padx=5, pady=5, sticky="w")
            
        except ValueError as e:
            self.mostrar_resultado(f"Error: {str(e)}", error=True)
    
    def resolver_sistema(self):
        try:
            ecuaciones_str = [entry.get() for entry in self.ecuaciones_entries]
            if not all(ecuaciones_str):
                raise ValueError("Todas las ecuaciones deben estar completas")
            
            metodo = self.metodo.get()
            tol = float(self.tolerancia.get())
            max_iter = int(self.max_iter.get())
            
            # Aquí iría la implementación de los métodos para resolver sistemas no lineales
            # Esto es un placeholder para la demostración
            resultado = f"Sistema resuelto con {metodo}\n"
            resultado += f"Ecuaciones: {ecuaciones_str}\n"
            resultado += f"Solución aproximada: x = 1.0, y = 1.0 (ejemplo)\n"
            resultado += f"Iteraciones: 5 (ejemplo)\n"
            resultado += f"Error: 1e-7 (ejemplo)"
            
            self.mostrar_resultado(resultado)
            self.mostrar_grafica()
            
        except ValueError as e:
            self.mostrar_resultado(f"Error: {str(e)}", error=True)
    
    def mostrar_resultado(self, mensaje, error=False):
        for widget in self.resultados_frame.winfo_children():
            widget.destroy()
        
        textbox = ctk.CTkTextbox(
            self.resultados_frame,
            width=800,
            height=150,
            font=("Consolas", 12)
        )
        textbox.pack(fill="both", expand=True)
        textbox.insert("end", mensaje)
        textbox.configure(state="disabled")
        
        if error:
            textbox.configure(text_color="red")
    
    def mostrar_grafica(self):
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        
        # Ejemplo de gráfica (esto sería reemplazado por la gráfica real del sistema)
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Datos de ejemplo
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z1 = X**2 + Y**2 - 4  # Ejemplo de primera ecuación
        Z2 = X**2 - Y          # Ejemplo de segunda ecuación
        
        ax.contour(X, Y, Z1, levels=[0], colors='blue')
        ax.contour(X, Y, Z2, levels=[0], colors='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gráfica del sistema de ecuaciones')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.grafica_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

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

# Actualización de IntegracionFrame en gui.py
class IntegracionFrame(BaseMethodFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Integración Numérica")
        
        self.content_frame.grid_rowconfigure(2, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Frame para entrada de función
        input_frame = ctk.CTkFrame(self.content_frame)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(input_frame, text="Función f(x):", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5)
        self.func_entry = ctk.CTkEntry(input_frame, width=300, placeholder_text="Ej: x**2 + math.sin(x)")
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame para límites y parámetros
        params_frame = ctk.CTkFrame(self.content_frame)
        params_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(params_frame, text="Límite inferior a:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5)
        self.a_entry = ctk.CTkEntry(params_frame, width=80)
        self.a_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(params_frame, text="Límite superior b:", font=("Arial", 14)).grid(row=0, column=2, padx=5, pady=5)
        self.b_entry = ctk.CTkEntry(params_frame, width=80)
        self.b_entry.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkLabel(params_frame, text="Número de intervalos:", font=("Arial", 14)).grid(row=0, column=4, padx=5, pady=5)
        self.n_entry = ctk.CTkEntry(params_frame, width=80)
        self.n_entry.insert(0, "10")
        self.n_entry.grid(row=0, column=5, padx=5, pady=5)
        
        ctk.CTkLabel(params_frame, text="Método:", font=("Arial", 14)).grid(row=0, column=6, padx=5, pady=5)
        self.metodo_int = ctk.CTkComboBox(
            params_frame,
            values=["Trapezoidal", "Simpson 1/3", "Simpson 3/8"],
            width=120
        )
        self.metodo_int.grid(row=0, column=7, padx=5, pady=5)
        
        self.calc_button = ctk.CTkButton(
            params_frame,
            text="Calcular Integral",
            command=self.calcular_integral,
            fg_color="#8B4513",
            font=("Arial", 14)
        )
        self.calc_button.grid(row=0, column=8, padx=10, pady=5)
        
        # Frame para resultados
        self.resultados_frame = ctk.CTkFrame(self.content_frame)
        self.resultados_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        # Frame para gráfica
        self.grafica_frame = ctk.CTkFrame(self.content_frame)
        self.grafica_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
    
    def calcular_integral(self):
        try:
            from metodos.integracion_numerica import IntegracionNumerica
            
            f_str = self.func_entry.get().strip()  # Añade strip() para eliminar espacios en blanco
            if not f_str:  # Verifica si la cadena está vacía
                raise ValueError("Debe ingresar una función")
                
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            n = int(self.n_entry.get())
            metodo = self.metodo_int.get()
            
            if a >= b:
                raise ValueError("El límite inferior debe ser menor que el superior")
            if n <= 0:
                raise ValueError("El número de intervalos debe ser positivo")
            
            # Definir la función a integrar con manejo de errores mejorado
            def f(x):
                try:
                    return IntegracionNumerica.evaluar_funcion(f_str, x)
                except Exception as e:
                    raise ValueError(f"No se pudo evaluar la función en x={x}: {str(e)}")
            
            # Calcular según el método seleccionado
            if metodo == "Trapezoidal":
                resultado = IntegracionNumerica.trapezoidal(f, a, b, n)
                formula = f"(h/2) * [f(a) + 2Σf(xi) + f(b)]\nDonde h = (b-a)/n"
            elif metodo == "Simpson 1/3":
                if n % 2 != 0:
                    raise ValueError("Simpson 1/3 requiere un número par de intervalos")
                resultado = IntegracionNumerica.simpson13(f, a, b, n)
                formula = f"(h/3) * [f(a) + 4Σf(x_impares) + 2Σf(x_pares) + f(b)]\nDonde h = (b-a)/n"
            elif metodo == "Simpson 3/8":
                if n % 3 != 0:
                    raise ValueError("Simpson 3/8 requiere un número de intervalos múltiplo de 3")
                resultado = IntegracionNumerica.simpson38(f, a, b, n)
                formula = f"(3h/8) * [f(a) + 3Σf(x_no_multiplos_3) + 2Σf(x_multiplos_3) + f(b)]\nDonde h = (b-a)/n"
            else:
                raise ValueError("Método no reconocido")
            
            # Resultado formateado
            resultado_texto = f"Método: {metodo}\n"
            resultado_texto += f"Función: {f_str}\n"
            resultado_texto += f"Intervalo: [{a:.4f}, {b:.4f}]\n"
            resultado_texto += f"Número de intervalos: {n}\n"
            resultado_texto += f"Fórmula usada:\n{formula}\n"
            resultado_texto += f"\nResultado de la integral ≈ {resultado:.8f}"
            
            self.mostrar_resultado(resultado_texto)
            self.mostrar_grafica(f, a, b, metodo, n)
            
        except ValueError as e:
            self.mostrar_resultado(f"Error: {str(e)}", error=True)
        except Exception as e:
            self.mostrar_resultado(f"Error inesperado: {str(e)}", error=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()