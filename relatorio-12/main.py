# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CONFIGURACAO - ALTERE O GRAU AQUI
# ============================================
GRAU_POLINOMIO = 4

# Dados experimentais com mais dispersao
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2.1, 5.8, 6.2, 10.5, 12.1, 16.8, 18.3])

# Numero de pontos
n = len(x)

print("=" * 70)
print(f"METODO DOS QUADRADOS MINIMOS - POLINOMIO DE GRAU {GRAU_POLINOMIO}")
print("=" * 70)
print(f"\nDados experimentais:")
print(f"x = {x}")
print(f"y = {y}")
print(f"\nNumero de pontos: n = {n}")

# Ajuste polinomial usando numpy.polyfit
coeficientes = np.polyfit(x, y, GRAU_POLINOMIO)
polinomio = np.poly1d(coeficientes)

# Calculo dos valores ajustados
y_ajustado = polinomio(x)

# Calculo dos residuos
residuos = y - y_ajustado
S = np.sum(residuos**2)

# Calculo do R²
y_media = np.mean(y)
SQT = np.sum((y - y_media)**2)
SQE = S
R2 = 1 - (SQE / SQT)

print(f"\n" + "=" * 70)
print(f"POLINOMIO DE GRAU {GRAU_POLINOMIO}")
print("=" * 70)
print(f"Coeficientes (do maior para o menor grau):")
for i, coef in enumerate(coeficientes):
    potencia = GRAU_POLINOMIO - i
    if potencia > 1:
        print(f"  a{potencia} (x^{potencia}) = {coef:.8f}")
    elif potencia == 1:
        print(f"  a{potencia} (x) = {coef:.8f}")
    else:
        print(f"  a{potencia} (constante) = {coef:.8f}")

print(f"\nEquacao: {polinomio}")
print(f"Soma dos quadrados dos residuos: S = {S:.8f}")
print(f"Coeficiente de determinacao: R² = {R2:.8f}")

print(f"\n{'i':>3} {'x':>6} {'y':>8} {'y_ajust':>12} {'residuo':>12} {'residuo²':>14}")
print("-" * 70)
for i in range(n):
    print(f"{i+1:3d} {x[i]:6.1f} {y[i]:8.2f} {y_ajustado[i]:12.6f} "
          f"{residuos[i]:12.8f} {residuos[i]**2:14.10f}")
print("-" * 70)

# Criar pontos para plotagem suave das curvas
x_linha = np.linspace(min(x) - 0.5, max(x) + 0.5, 1000)
y_linha = polinomio(x_linha)

# Criacao do grafico
plt.figure(figsize=(12, 8))

# Plotar a curva ajustada
plt.plot(x_linha, y_linha, color='blue', linewidth=2.5,
         label=f'Polinômio grau {GRAU_POLINOMIO}', alpha=0.9)

# Plotar os pontos experimentais
plt.scatter(x, y, color='red', s=150, zorder=5, label='Dados experimentais',
            edgecolors='black', linewidths=2)

# Plotar as linhas dos residuos
for i in range(n):
    plt.plot([x[i], x[i]], [y[i], y_ajustado[i]],
             color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Destacar as regioes entre os pontos onde pode ocorrer overfitting
for i in range(len(x) - 1):
    x_meio = (x[i] + x[i+1]) / 2
    plt.axvline(x=x_meio, color='lightgray', linestyle=':', linewidth=1, alpha=0.4)

# Configuracoes do grafico
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)

# Manter escala fixa para comparacao entre diferentes graus
plt.xlim(min(x) - 0.5, max(x) + 0.5)
plt.ylim(-5, 25)  # Escala fixa para comparacao

# Titulo simples
titulo = f'Ajuste Polinomial de Grau {GRAU_POLINOMIO}'

plt.title(titulo, fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')

plt.tight_layout()

# Nome do arquivo baseado no grau
nome_arquivo = f'./relatorio-12/polinomio_grau_{GRAU_POLINOMIO}.png'
plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')

print(f"\n" + "=" * 70)
print(f"Grafico salvo como '{nome_arquivo}'")
print("=" * 70)

# Analise de overfitting
print(f"\n" + "=" * 70)
print("ANALISE")
print("=" * 70)

if GRAU_POLINOMIO >= n - 1:
    print(f"\n⚠ OVERFITTING SEVERO!")
    print(f"O grau {GRAU_POLINOMIO} é >= n-1 ({n-1}).")
    print(f"O polinômio passa por todos (ou quase todos) os pontos,")
    print(f"gerando oscilações (ruídos) entre os intervalos.")
    print(f"O modelo perde capacidade de generalização.")
elif GRAU_POLINOMIO >= 4:
    print(f"\n⚠ Possível overfitting!")
    print(f"Grau {GRAU_POLINOMIO} pode gerar oscilações entre pontos.")
    print(f"Observe o gráfico cuidadosamente.")
elif GRAU_POLINOMIO == 2 or GRAU_POLINOMIO == 3:
    print(f"\n✓ Grau {GRAU_POLINOMIO} geralmente oferece bom equilíbrio.")
    print(f"Flexibilidade moderada sem overfitting severo.")
else:
    print(f"\n✓ Grau {GRAU_POLINOMIO} é simples.")
    print(f"Pode haver subajuste se os dados não forem lineares.")

print("=" * 70)
print(f"\nPara testar outro grau, altere a variável GRAU_POLINOMIO no topo do código.")
print("=" * 70)

plt.show()
