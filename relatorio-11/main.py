# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Dados experimentais
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.8, 3.2, 4.9, 7.1, 8.9])

# Numero de pontos
n = len(x)

# Calculo das somas necessarias
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x * y)

print("=" * 50)
print("METODO DOS QUADRADOS MINIMOS")
print("=" * 50)
print(f"\nDados experimentais:")
print(f"x = {x}")
print(f"y = {y}")
print(f"\nNumero de pontos: n = {n}")
print(f"\nSomas calculadas:")
print(f"Somatorio de x = {sum_x}")
print(f"Somatorio de y = {sum_y}")
print(f"Somatorio de x² = {sum_x2}")
print(f"Somatorio de xy = {sum_xy}")

# Montagem das equacoes normais
# Sistema: na0 + (soma x)a1 = soma y
#          (soma x)a0 + (soma x^2)a1 = soma xy

# Resolucao usando matrizes
A = np.array([[n, sum_x],
              [sum_x, sum_x2]])
b = np.array([sum_y, sum_xy])

# Solucao do sistema
coeficientes = np.linalg.solve(A, b)
a0, a1 = coeficientes

print(f"\n" + "=" * 50)
print("EQUACOES NORMAIS")
print("=" * 50)
print(f"\n{n}a0 + {sum_x}a1 = {sum_y}")
print(f"{sum_x}a0 + {sum_x2}a1 = {sum_xy}")

print(f"\n" + "=" * 50)
print("SOLUCAO")
print("=" * 50)
print(f"\na0 = {a0:.4f}")
print(f"a1 = {a1:.4f}")
print(f"\nReta ajustada: y = {a0:.4f} + {a1:.4f}x")

# Calculo dos residuos
y_ajustado = a0 + a1 * x
residuos = y - y_ajustado
S = np.sum(residuos**2)

print(f"\n" + "=" * 50)
print("RESIDUOS")
print("=" * 50)
print(f"\n{'i':>3} {'x':>6} {'y':>8} {'y_ajust':>8} {'residuo':>8} {'residuo²':>10}")
print("-" * 50)
for i in range(n):
    print(f"{i+1:3d} {x[i]:6.1f} {y[i]:8.2f} {y_ajustado[i]:8.2f} "
          f"{residuos[i]:8.4f} {residuos[i]**2:10.6f}")
print("-" * 50)
print(f"Soma dos quadrados dos residuos: S = {S:.6f}")

# Calculo do coeficiente de determinacao R^2
y_media = np.mean(y)
SQT = np.sum((y - y_media)**2)  # Soma total dos quadrados
SQE = S  # Soma dos quadrados dos erros
R2 = 1 - (SQE / SQT)

print(f"Coeficiente de determinacao: R² = {R2:.6f}")

# Criacao do grafico
plt.figure(figsize=(10, 6))

# Plotar os pontos experimentais
plt.scatter(x, y, color='red', s=100, zorder=3, label='Dados experimentais')

# Plotar a reta ajustada
x_linha = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
y_linha = a0 + a1 * x_linha
plt.plot(x_linha, y_linha, color='blue', linewidth=2, label=f'y = {a0:.4f} + {a1:.4f}x')

# Plotar as linhas dos residuos
for i in range(n):
    plt.plot([x[i], x[i]], [y[i], y_ajustado[i]],
             color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Configuracoes do grafico
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Método dos Quadrados Mínimos\nS = {S:.6f} | R² = {R2:.6f}',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)

# Adicionar texto com informacoes
textstr = f'Equacao: y = {a0:.4f} + {a1:.4f}x\n'
textstr += f'Erro quadratico: S = {S:.6f}\n'
textstr += f'R² = {R2:.6f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./quadrados_minimos.png', dpi=300, bbox_inches='tight')
print(f"\n" + "=" * 50)
print("Grafico salvo como 'quadrados_minimos.png'")
print("=" * 50)

plt.show()
