# main.py
from gui import App # Importa tu clase App

def main():
    # NO crees un tk.Tk() aquí si App hereda de customtkinter.CTk
    app = App() # Simplemente crea una instancia de tu clase App
    app.mainloop() # La clase App (que hereda de CTk) tiene su propio mainloop

if __name__ == "__main__":
    main()