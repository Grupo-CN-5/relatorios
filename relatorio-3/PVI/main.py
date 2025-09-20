"""
Método de Taylor (ordem N) para PVI: y' = f(x,y), y(a)=y0 em [a,b].

• Gera y', y'', y''', ... simbolicamente com o operador D = d/dx + f*d/dy.
• Faz passos de Taylor: y_{n+1} = y_n + sum_{k=1..p} (h^k/k!) * y^{(k)}(x_n,y_n)
• Inclui exemplo: y' = x + y, y(0)=1 em [0,2], compara com solução exata.
"""

from __future__ import annotations
import math
from typing import Callable, List, Tuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Símbolos globais para facilitar a definição de f(x,y)
x, y = sp.symbols('x y')


# ----------------------- Núcleo do método de Taylor ------------------------ #
def build_taylor_derivatives(f_expr: sp.Expr, order: int) -> List[sp.Expr]:
    """
    Constrói a lista [f, D f, D^2 f, ..., D^(order-1) f] via SymPy.
    D = d/dx + f*d/dy é a derivada total ao longo da solução.
    """
    if order < 1:
        raise ValueError("order deve ser >= 1")

    derivs = [sp.simplify(f_expr)]
    D = lambda g: sp.diff(g, x) + f_expr * sp.diff(g, y)

    g = f_expr
    for _ in range(order - 1):
        g = sp.simplify(D(g))
        derivs.append(g)

    return derivs  # len = order


def lambdify_derivs(derivs: List[sp.Expr]) -> List[Callable[[float, float], float]]:
    """Converte as expressões simbólicas em funções numéricas (x,y) -> float."""
    return [sp.lambdify((x, y), d, "numpy") for d in derivs]


def taylor_step(xn: float, yn: float, h: float,
                deriv_funcs: List[Callable[[float, float], float]]) -> float:
    """
    Um passo do método de Taylor de ordem p = len(deriv_funcs).
    y_{n+1} = y_n + sum_{k=1..p} (h^k/k!) * y^{(k)}(x_n,y_n)
    onde y^{(1)} = f, y^{(2)} = D f, ...
    """
    y_next = yn
    for k, fk in enumerate(deriv_funcs, start=1):
        yk = fk(xn, yn)
        y_next += (h ** k) / math.factorial(k) * yk
    return y_next


def taylor_ivp(f_expr: sp.Expr,
               a: float, b: float,
               y0: float,
               h: float,
               order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve y' = f(x,y), y(a)=y0 em [a,b] por Taylor de ordem 'order'.
    Retorna (xs, ys) com xs uniformemente espaçado.

    Observação: o passo é levemente ajustado para cair exatamente em b.
    """
    if order < 1:
        raise ValueError("order deve ser >= 1")
    if h <= 0:
        raise ValueError("h deve ser > 0")
    if b <= a:
        raise ValueError("exige b > a")

    # Ajusta h para terminar exatamente em b:
    n_steps = int(math.ceil((b - a) / h))
    h = (b - a) / n_steps

    derivs = build_taylor_derivatives(f_expr, order)
    deriv_funcs = lambdify_derivs(derivs)

    xs = np.linspace(a, b, n_steps + 1)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i in range(n_steps):
        ys[i + 1] = taylor_step(xs[i], ys[i], h, deriv_funcs)

    return xs, ys


# ------------------------------ Exemplo pronto ----------------------------- #
def exemplo():
    """
    Exemplo clássico: y' = x + y, y(0)=1 em [0,2].
    Solução exata: y(x) = -x - 1 + 2 e^x.
    Mostra curvas e gráfico de erro vs ordem.
    """
    f_example = x + y
    a, b, y0 = 0.0, 2.0, 1.0
    h = 0.1
    orders = [1, 2, 3, 4, 5]

    # Solução exata no mesmo grid do exemplo
    xs_ref = np.arange(a, b + 1e-12, h)
    y_exact = -xs_ref - 1 + 2 * np.exp(xs_ref)

    resultados = {}
    for p in orders:
        xs_p, ys_p = taylor_ivp(f_example, a, b, y0, h, p)
        resultados[p] = (xs_p, ys_p)

    # --- gráfico solução exata vs aproximações ---
    plt.figure()
    plt.plot(xs_ref, y_exact, label="Solução exata")
    for p in orders:
        xs_p, ys_p = resultados[p]
        plt.plot(xs_p, ys_p, linestyle="--", marker="o", label=f"Taylor ordem {p}")
    plt.title("PVI via Taylor: y' = x + y, y(0)=1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- gráfico erro máximo por ordem ---
    erros = []
    for p in orders:
        xs_p, ys_p = resultados[p]
        y_ex_on_grid = -xs_p - 1 + 2 * np.exp(xs_p)
        err_inf = np.max(np.abs(ys_p - y_ex_on_grid))
        erros.append(err_inf)

    plt.figure()
    plt.plot(orders, erros, marker="o")
    plt.title("Erro máximo (norma ∞) vs Ordem de Taylor")
    plt.xlabel("Ordem p")
    plt.ylabel("Erro máximo no intervalo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- gráfico erro ponto a ponto para cada ordem ---
    plt.figure()
    for p in orders:
        xs_p, ys_p = resultados[p]
        y_ex_on_grid = -xs_p - 1 + 2 * np.exp(xs_p)
        erro_ponto = np.abs(ys_p - y_ex_on_grid)
        plt.plot(xs_p, erro_ponto, marker="o", linestyle="--", label=f"Ordem {p}")
    plt.title("Erro absoluto ponto a ponto vs x")
    plt.xlabel("x")
    plt.ylabel("Erro absoluto |y_aprox - y_exato|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------ Uso básico -------------------------------- #
if __name__ == "__main__":
    # Exemplo rápido de uso programático (troque f, a, b, y0, h, ordem):
    f = x + y           # defina f(x,y) como expressão SymPy
    a, b = 0.0, 2.0
    y0 = 1.0
    h = 0.1
    ordem = 4

    xs, ys = taylor_ivp(f, a, b, y0, h, ordem)
    for xi, yi in zip(xs[:5], ys[:5]):  # imprime só os 5 primeiros
        print(f"x={xi:.2f}, y≈{yi:.6f}")

    exemplo()
