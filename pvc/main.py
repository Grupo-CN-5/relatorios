# Taylor-series-based shooting method for 2nd-order BVPs
# Author: ChatGPT
#
# Resolve y'' = g(x, y, y'),  com condições de contorno y(a)=alpha, y(b)=beta
# usando método de disparo e integrador por Série de Taylor (ordem 1 ou 2).
# - order=1  -> Euler (Taylor de 1ª ordem)
# - order=2  -> Taylor de 2ª ordem (usa derivadas numéricas Fx e Fz)
#
# As derivadas necessárias do campo F(x,z) são estimadas por diferenças finitas,
# então você só precisa fornecer g(x,y,y').
#
# Exemplo ao final: y'' = -y,  y(0)=0, y(pi/2)=1 -> solução exata: y(x)=sin(x).

from dataclasses import dataclass
import numpy as np
import math
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt

Vector = np.ndarray

@dataclass
class BVPResult:
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    slope_initial: float
    iters: int
    converged: bool
    residual: float


def _jacobian_wrt_z(F: Callable[[float, Vector], Vector], x: float, z: Vector, eps: float = 1e-6) -> np.ndarray:
    """Jacobiano numérico dF/dz por diferenças centrais."""
    m = len(z)
    f0 = F(x, z)
    J = np.zeros((len(f0), m))
    for j in range(m):
        dz = np.zeros_like(z)
        dz[j] = eps
        fp = F(x, z + dz)
        fm = F(x, z - dz)
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def _partial_wrt_x(F: Callable[[float, Vector], Vector], x: float, z: Vector, eps: float = 1e-6) -> Vector:
    """Parcial dF/dx (z fixo) por diferenças centrais."""
    fp = F(x + eps, z)
    fm = F(x - eps, z)
    return (fp - fm) / (2 * eps)


def taylor_step(F: Callable[[float, Vector], Vector], x: float, z: Vector, h: float, order: int) -> Vector:
    """
    Um passo do método de Taylor para z' = F(x, z).
    Suporta ordem 1 (Euler) e ordem 2 (Taylor-2).
    """
    f = F(x, z)

    if order == 1:
        # Euler (Taylor de 1ª ordem)
        return z + h * f

    elif order == 2:
        # Taylor de 2ª ordem: z_{n+1} = z + h z' + (h^2/2) z''
        # z'' = d/dx F = F_x + F_z * z'
        Fx = _partial_wrt_x(F, x, z)
        Jz = _jacobian_wrt_z(F, x, z)
        z2 = Fx + Jz @ f
        return z + h * f + 0.5 * (h ** 2) * z2

    else:
        raise NotImplementedError("A implementação atual suporta somente ordem=1 ou ordem=2.")


def integrate_ivp_taylor(F: Callable[[float, Vector], Vector],
                         a: float, b: float, z0: Vector, N: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integra z' = F(x,z) de x=a até x=b com N passos uniformes usando Taylor (ordem dada).
    Retorna a malha x e a matriz Z (N+1, m).
    """
    x = np.linspace(a, b, N + 1)
    h = (b - a) / N
    Z = np.zeros((N + 1, len(z0)))
    Z[0] = z0.copy()

    for n in range(N):
        Z[n + 1] = taylor_step(F, x[n], Z[n], h, order)

    return x, Z


def shoot_bvp_taylor(g: Callable[[float, float, float], float],
                     a: float, b: float, alpha: float, beta: float,
                     N: int = 200, order: int = 2,
                     tol: float = 1e-8, maxit: int = 30,
                     s0: float = 0.0, s1: float = 1.0) -> BVPResult:
    """
    Método de disparo para y'' = g(x,y,y'), com y(a)=alpha e y(b)=beta.
    Integra o PVI interno com Taylor (ordem 1 ou 2) e ajusta y'(a) por secante.

    Parâmetros principais:
      - g(x,y,y')   : função da EDO de 2ª ordem (forma de estado explícita)
      - a, b        : intervalo
      - alpha, beta : condições de contorno
      - N           : número de passos
      - order       : 1 (Euler) ou 2 (Taylor-2)
      - tol         : tolerância no resíduo em x=b
      - maxit       : máx. iterações da secante
      - s0, s1      : chutes iniciais para y'(a)
    """
    def F(x: float, z: Vector) -> Vector:
        y, v = z[0], z[1]
        return np.array([v, g(x, y, v)], dtype=float)

    def solve_with_s(slope: float):
        z0 = np.array([alpha, slope], dtype=float)
        xs, Zs = integrate_ivp_taylor(F, a, b, z0, N, order)
        yb = Zs[-1, 0]
        return yb - beta, xs, Zs

    r0, _, _ = solve_with_s(s0)
    r1, xs, Zs = solve_with_s(s1)

    it = 1
    while it <= maxit and abs(r1) > tol:
        # Atualização por secante
        if abs(r1 - r0) < 1e-15:
            s1 = s1 + 1e-6
            r0, _, _ = solve_with_s(s0)
            r1, xs, Zs = solve_with_s(s1)
            it += 1
            continue

        s2 = s1 - r1 * (s1 - s0) / (r1 - r0)
        s0, r0 = s1, r1
        s1 = s2
        r1, xs, Zs = solve_with_s(s1)
        it += 1

    y = Zs[:, 0]
    v = Zs[:, 1]
    converged = abs(r1) <= tol

    return BVPResult(x=xs, y=y, v=v, slope_initial=s1, iters=it, converged=converged, residual=float(r1))


# ========== FEATURE ADICIONADA ==========
def plot_taylor_order_comparison(
    g: Callable[[float, float, float], float],
    a: float, b: float, alpha: float, beta: float,
    N: int = 100,
    orders: Optional[List[int]] = None,
    y_exact: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    s0: float = 0.0, s1: float = 1.0,
):
    """
    Gera gráficos mostrando como ordens maiores da Série de Taylor
    se aproximam da solução exata (se fornecida).

    - orders: lista de ordens para comparar (ex: [1,2]).
    - y_exact: função vetorizada da solução exata y(x). Se None, plota só as aproximações.
    """
    if orders is None:
        orders = [1, 2]

    # Resolver para cada ordem
    results = []
    for ordk in orders:
        res = shoot_bvp_taylor(g, a, b, alpha, beta, N=N, order=ordk, s0=s0, s1=s1)
        results.append((ordk, res))

    x = results[0][1].x
    yex = y_exact(x) if y_exact is not None else None

    # Figura 1: soluções
    plt.figure()
    if yex is not None:
        plt.plot(x, yex, '--', label="Solução exata")
    for ordk, res in results:
        plt.plot(x, res.y, label=f"Taylor {ordk}ª ordem (N={N})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparação de ordens da Série de Taylor (método de disparo)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figura 2: erro absoluto (se exata disponível)
    if yex is not None:
        plt.figure()
        for ordk, res in results:
            err = np.abs(res.y - yex)
            plt.plot(x, err, label=f"Erro absoluto - ordem {ordk}")
        plt.xlabel("x")
        plt.ylabel("erro absoluto")
        plt.title("Erro absoluto vs ordem da Série de Taylor")
        plt.legend()
        plt.tight_layout()
        plt.show()
# ========== FIM DA FEATURE ==========


# --------------------- Exemplo de uso ---------------------
# BVP: y'' = -y,  y(0) = 0,  y(pi/2) = 1  -> solução exata y(x) = sin(x)
def g_example(x: float, y: float, v: float) -> float:
    return -y

if __name__ == "__main__":
    a = 0.0
    b = math.pi / 2
    alpha = 0.0
    beta = 1.0

    # Experimente "order = 1" e "order = 2"
    order = 2
    N = 200

    res = shoot_bvp_taylor(g_example, a, b, alpha, beta, N=N, order=order, s0=0.0, s1=2.0)

    # Mostrar resultados e comparar com a solução exata
    x = res.x
    y_num = res.y
    y_ex = np.sin(x)

    print(f"Convergiu: {res.converged}, iterações: {res.iters}, slope inicial encontrado: {res.slope_initial:.8f}, residual: {res.residual:.2e}")
    max_err = np.max(np.abs(y_num - y_ex))
    print(f"Erro máximo |y_num - y_ex| = {max_err:.3e}")

    # Gráfico básico da solução
    plt.figure()
    plt.plot(x, y_num, label=f"Taylor ordem {order} (N={N})")
    plt.plot(x, y_ex, '--', label="Solução exata sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BVP por Disparo com Método de Taylor")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======= USO DA FEATURE ADICIONADA =======
    # Compara ordens 1 e 2 e mostra aproximação vs exata + erro absoluto
    plot_taylor_order_comparison(
        g=g_example, a=a, b=b, alpha=alpha, beta=beta,
        N=80, orders=[1, 2],
        y_exact=np.sin,  # função exata
        s0=0.0, s1=2.0
    )
