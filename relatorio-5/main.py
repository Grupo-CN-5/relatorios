"""
PVC por Diferenças Finitas (segunda ordem, esquema central)
Resolve: y'' + p(x) y' + q(x) y = r(x) em [a,b], com y(a)=alpha, y(b)=beta.

Gera:
- CSV com x, y_num, (opcional) y_exata, erro_abs, erro_rel
- PNG com gráfico de comparação (se y_exata fornecida)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# ------------------------------------------------------------
# Núcleo: montador/solucionador por diferenças finitas
# ------------------------------------------------------------
def solve_bvp_fdm_linear(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float, b: float,
    alpha: float, beta: float,
    N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve y'' + p(x) y' + q(x) y = r(x),  x in [a,b]
    CC: y(a)=alpha, y(b)=beta
    Malha uniforme com N subintervalos (N+1 pontos).
    Retorna x (N+1,), y (N+1,).
    """
    if N < 2:
        raise ValueError("N deve ser >= 2.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)

    m = N - 1  # número de pontos internos
    A = np.zeros((m, m), dtype=float)
    rhs = np.zeros(m, dtype=float)

    # Esquema central:
    # y''(x_i) ~ (y_{i-1} - 2 y_i + y_{i+1}) / h^2
    # y'(x_i)  ~ (y_{i+1} - y_{i-1}) / (2 h)
    # => (1/h^2 - p_i/(2h)) * y_{i-1} + (-2/h^2 + q_i) * y_i + (1/h^2 + p_i/(2h)) * y_{i+1} = r_i
    for k, i in enumerate(range(1, N)):  # i = 1..N-1; k = 0..m-1
        xi = x[i]
        pi = p(xi)
        qi = q(xi)
        ri = r(xi)

        a_im1 = (1.0 / h**2) - (pi / (2.0 * h))
        a_i   = (-2.0 / h**2) + qi
        a_ip1 = (1.0 / h**2) + (pi / (2.0 * h))

        # Diagonal
        A[k, k] = a_i

        # Sub e superdiagonais
        if k - 1 >= 0:
            A[k, k - 1] = a_im1
        else:
            rhs[k] -= a_im1 * alpha  # contribuição de y0

        if k + 1 <= m - 1:
            A[k, k + 1] = a_ip1
        else:
            rhs[k] -= a_ip1 * beta   # contribuição de yN

        rhs[k] += ri

    # Resolve sistema
    y_interior = np.linalg.solve(A, rhs)

    # Remonta vetor completo com contornos
    y = np.empty(N + 1, dtype=float)
    y[0] = alpha
    y[-1] = beta
    y[1:-1] = y_interior
    return x, y

# ------------------------------------------------------------
# Utilitário: erro relativo ponto a ponto (NaN quando y_exata=0)
# ------------------------------------------------------------
def relative_error_pointwise(y_num: np.ndarray, y_exact: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    abs_exact = np.abs(y_exact)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(abs_exact > tol, np.abs(y_num - y_exact) / abs_exact, np.nan)
    return rel

# ------------------------------------------------------------
# Exemplo: y'' + y = 0, y(0)=0, y(pi/2)=1  -> y(x)=sin(x)
# ------------------------------------------------------------
def exemplo_seno(
    N: int = 80,
    a: float = 0.0,
    b: float = math.pi / 2.0,
    alpha: float = 0.0,
    beta: float = 1.0,
    csv_path: str = "pvc_fdm_results.csv",
    png_path: str = "pvc_fdm_plot.png",
    mostrar_plot: bool = True
):
    # p(x)=0, q(x)=1, r(x)=0
    p = lambda x: 0.0
    q = lambda x: 1.0
    rfun = lambda x: 0.0

    x, y_fd = solve_bvp_fdm_linear(p, q, rfun, a, b, alpha, beta, N)

    # Solução exata para comparação
    y_exact = np.sin(x)

    # Erros
    err_abs = np.abs(y_fd - y_exact)
    err_rel = relative_error_pointwise(y_fd, y_exact)  # NaN quando y_exata ~ 0

    # Métricas globais
    linf_abs = float(np.max(err_abs))
    l2_abs = float(np.sqrt(np.mean(err_abs**2)))

    # Para métricas relativas globais, ignore NaNs (pontos onde y_exata=0)
    linf_rel = float(np.nanmax(err_rel))
    # Para L2_rel, substitui NaN por 0 (não afeta a média quadrática)
    l2_rel = float(np.sqrt(np.nanmean(np.nan_to_num(err_rel, nan=0.0)**2)))

    h = (b - a) / N

    print(f"[INFO] h={h:.6f}, N={N}")
    print(f"[INFO] ||erro_abs||_inf={linf_abs:.3e}, ||erro_abs||_2={l2_abs:.3e}")
    print(f"[INFO] ||erro_rel||_inf={linf_rel:.3e}, ||erro_rel||_2={l2_rel:.3e}  (NaN onde y_exata=0)")

    # Salva CSV com erro relativo também
    df = pd.DataFrame({
        "x": x,
        "y_fd": y_fd,
        "y_exata": y_exact,
        "erro_abs": err_abs,
        "erro_rel": err_rel
    })
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV salvo em: {csv_path}")

    # Gráfico
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(x, y_exact, label="Solução exata  y(x)=sin(x)")
    plt.plot(x, y_fd, marker="o", linestyle="--", label=f"Diferenças Finitas (N={N})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PVC: y'' + y = 0,  y(0)=0,  y(π/2)=1 — Comparação")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    print(f"[OK] Figura salva em: {png_path}")
    if mostrar_plot:
        plt.show()
    else:
        plt.close()

# ------------------------------------------------------------
# Ponto de entrada
# ------------------------------------------------------------
if __name__ == "__main__":
    exemplo_seno(N=40)
