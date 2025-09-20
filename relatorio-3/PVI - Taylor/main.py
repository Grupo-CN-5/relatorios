# Tkinter – PVI por Série de Taylor (até 5ª ordem)
#
# O app resolve o PVI y' = f(x,y), y(x0)=y0 usando o método de Taylor
# de ordem 1 até 5, derivando simbolicamente f(x,y) com Sympy via o
# operador total D = d/dx + y' d/dy, onde substituímos y' por f(x,y)
# a cada aplicação. Assim obtemos y''(x), y'''(x), ... automaticamente.
# 
# Recursos:
# - Campo para f(x,y) (ex.: y, x+y, x*y, y**2 - x, sin(x)+y )
# - Campos para x0, y0, passo h, número de passos N e ordem (1..5)
# - (Opcional) solução exata g(x) para comparação (ex.: exp(x))
# - Gráfico interativo com matplotlib embutido no Tkinter
# - Tabela simples com os 5 primeiros pontos
# - Botões para Simular e Limpar
#
# Observações:
# - O uso de Sympy permite derivação segura; funções matemáticas
#   suportadas: sin, cos, tan, exp, log, sqrt, sinh, cosh, tanh, etc.
# - Para evitar expressões perigosas, o parser limita os nomes permitidos.
# - Este script é auto-contido; requer: sympy, numpy, matplotlib.

import tkinter as tk
from tkinter import ttk, messagebox
from math import isfinite
import numpy as np
from sympy import symbols, sympify, diff, lambdify, sin, cos, tan, exp, log, sqrt, sinh, cosh, tanh, asin, acos, atan
from sympy import simplify
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ALLOWED_FUNCS = {
    'sin': sin, 'cos': cos, 'tan': tan,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'exp': exp, 'log': log, 'sqrt': sqrt,
    'asin': asin, 'acos': acos, 'atan': atan,
    # constantes comuns
    'e': exp(1), 'E': exp(1)
}

class TaylorPVIApp:
    def __init__(self, master):
        self.master = master
        master.title("PVI por Série de Taylor – Tkinter")
        master.geometry("980x700")

        # --- Top form ---
        form = ttk.Frame(master)
        form.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # Linha 1: f(x,y) e g(x) exata
        ttk.Label(form, text="f(x,y) =").grid(row=0, column=0, sticky=tk.W)
        self.f_entry = ttk.Entry(form, width=40)
        self.f_entry.grid(row=0, column=1, padx=6)
        self.f_entry.insert(0, "y")  # padrão: y' = y (crescimento exponencial)

        ttk.Label(form, text="g(x) (exata, opcional) =").grid(row=0, column=2, sticky=tk.W)
        self.g_entry = ttk.Entry(form, width=30)
        self.g_entry.grid(row=0, column=3, padx=6)
        self.g_entry.insert(0, "exp(x)")

        # Linha 2: x0, y0, h, N, ordem
        ttk.Label(form, text="x0=").grid(row=1, column=0, sticky=tk.W)
        self.x0_entry = ttk.Entry(form, width=10)
        self.x0_entry.grid(row=1, column=1, sticky=tk.W)
        self.x0_entry.insert(0, "0.0")

        ttk.Label(form, text="y0=").grid(row=1, column=2, sticky=tk.E)
        self.y0_entry = ttk.Entry(form, width=10)
        self.y0_entry.grid(row=1, column=3, sticky=tk.W)
        self.y0_entry.insert(0, "1.0")

        ttk.Label(form, text="h=").grid(row=1, column=4, sticky=tk.E)
        self.h_entry = ttk.Entry(form, width=10)
        self.h_entry.grid(row=1, column=5, sticky=tk.W)
        self.h_entry.insert(0, "0.1")

        ttk.Label(form, text="N passos=").grid(row=1, column=6, sticky=tk.E)
        self.n_entry = ttk.Entry(form, width=10)
        self.n_entry.grid(row=1, column=7, sticky=tk.W)
        self.n_entry.insert(0, "40")

        ttk.Label(form, text="Ordem (1-5)=").grid(row=1, column=8, sticky=tk.E)
        self.order_combo = ttk.Combobox(form, values=[1,2,3,4,5], width=3, state="readonly")
        self.order_combo.grid(row=1, column=9, sticky=tk.W)
        self.order_combo.set(3)

        # Botões
        btns = ttk.Frame(form)
        btns.grid(row=0, column=4, columnspan=2, padx=10)
        sim_btn = ttk.Button(btns, text="Simular", command=self.run)
        clr_btn = ttk.Button(btns, text="Limpar", command=self.clear_plot)
        sim_btn.grid(row=0, column=0, padx=4)
        clr_btn.grid(row=0, column=1, padx=4)

        # --- Plot area ---
        self.fig = Figure(figsize=(6.6, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Solução do PVI por Série de Taylor')
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Tabela ---
        right = ttk.Frame(master)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        ttk.Label(right, text="Primeiros pontos (x, y num, y exata, erro)").pack(anchor=tk.W)
        self.table = ttk.Treeview(right, columns=("x","y_num","y_ex","err"), show='headings', height=18)
        for col, w in [("x",100),("y_num",140),("y_ex",140),("err",120)]:
            self.table.heading(col, text=col)
            self.table.column(col, width=w, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)

    # --------- Core math: construir derivadas por D ---------
    def build_derivatives(self, f_expr):
        x, y = symbols('x y')
        f = f_expr
        # operador total: D(phi) = dphi/dx + f * dphi/dy
        def D(phi):
            return simplify(diff(phi, x) + f*diff(phi, y))
        ders = [None]  # 0-index dummy: ders[k] = y^(k)
        y1 = simplify(f)
        ders.append(y1)
        yk = y1
        for k in range(2, 6):  # até 5ª ordem
            yk = simplify(D(yk))
            ders.append(yk)
        return ders  # [_, y', y'', y''', y'''' , y''''']

    # --------- Parser seguro para f(x,y) e g(x) ---------
    def parse_fx(self, s):
        x, y = symbols('x y')
        try:
            expr = sympify(s, locals={**ALLOWED_FUNCS, 'x':x, 'y':y})
            return expr
        except Exception as e:
            raise ValueError(f"Expressão inválida para f(x,y): {e}")

    def parse_gx(self, s):
        x = symbols('x')
        if s.strip() == "":
            return None
        try:
            expr = sympify(s, locals={**ALLOWED_FUNCS, 'x':x})
            return expr
        except Exception as e:
            raise ValueError(f"Expressão inválida para g(x): {e}")

    # --------- Integrador de Taylor ---------
    def taylor_step(self, xk, yk, h, ders_funcs, order):
        # ders_funcs[k] = função para y^(k)(x,y)
        acc = yk
        hpow = h
        fact = 1.0
        for k in range(1, order+1):
            fact *= k
            term = (hpow / fact) * ders_funcs[k](xk, yk)
            acc += term
            hpow *= h
        return acc

    def run(self):
        try:
            f_str = self.f_entry.get().strip()
            g_str = self.g_entry.get().strip()
            x0 = float(self.x0_entry.get())
            y0 = float(self.y0_entry.get())
            h = float(self.h_entry.get())
            N = int(self.n_entry.get())
            order = int(self.order_combo.get())
            if order < 1 or order > 5:
                raise ValueError("Ordem deve estar entre 1 e 5.")
            if h == 0 or N <= 0:
                raise ValueError("Use h != 0 e N > 0.")

            # Parse e derivadas simbólicas
            f_expr = self.parse_fx(f_str)
            ders = self.build_derivatives(f_expr)  # até 5ª

            # lambdify para avaliação numérica
            x, y = symbols('x y')
            ders_funcs = [None]
            for k in range(1, 6):
                ders_funcs.append(lambdify((x,y), ders[k], 'numpy'))

            # solução exata opcional
            g_expr = None
            g_func = None
            if g_str:
                try:
                    g_expr = self.parse_gx(g_str)
                    gx = symbols('x')
                    g_func = lambdify(gx, g_expr, 'numpy')
                except Exception as ge:
                    messagebox.showwarning("Aviso", f"Solução exata ignorada: {ge}")
                    g_func = None

            # Integração
            xs = [x0]
            ys = [y0]
            for i in range(N):
                yn1 = self.taylor_step(xs[-1], ys[-1], h, ders_funcs, order)
                xs.append(xs[-1] + h)
                ys.append(float(yn1))
                if not isfinite(ys[-1]):
                    raise FloatingPointError("A solução divergiu (não finita).")

            xs = np.array(xs)
            ys = np.array(ys)

            # Plot
            self.ax.clear()
            self.ax.grid(True, alpha=0.25)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_title(f"PVI por Série de Taylor (ordem {order})")
            self.ax.plot(xs, ys, marker='o', label=f"Taylor {order}ª")

            # Comparação com exata, se houver
            errs = []
            if g_func is not None:
                try:
                    y_exact = g_func(xs)
                    self.ax.plot(xs, y_exact, linestyle='--', label="Exata")
                    errs = list(np.abs(ys - y_exact))
                except Exception:
                    errs = []

            self.ax.legend(loc='best')
            self.canvas.draw()

            # Tabela (primeiros 10 pontos, incluindo inicial)
            for item in self.table.get_children():
                self.table.delete(item)
            max_rows = min(10, len(xs))
            for i in range(max_rows):
                xe = xs[i]
                yn = ys[i]
                if errs:
                    ye = g_func(xe)
                    er = abs(yn - ye)
                    self.table.insert('', 'end', values=(f"{xe:.4f}", f"{yn:.6g}", f"{float(ye):.6g}", f"{er:.3e}"))
                else:
                    self.table.insert('', 'end', values=(f"{xe:.4f}", f"{yn:.6g}", "-", "-"))

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def clear_plot(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Solução do PVI por Série de Taylor')
        self.canvas.draw()
        for item in self.table.get_children():
            self.table.delete(item)

if __name__ == '__main__':
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = TaylorPVIApp(root)
    root.mainloop()
