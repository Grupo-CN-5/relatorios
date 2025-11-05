# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CONFIGURACAO - ALTERE O GRAU AQUI
# ============================================
GRAU_POLINOMIO = 6

# Dados experimentais com mais dispersao
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2.1, 5.8, 6.2, 10.5, 12.1, 16.8, 18.3])

# Numero de pontos
n = len(x)

print("=" * 70)
print(f"ANALISE DE EXTRAPOLACAO - POLINOMIO DE GRAU {GRAU_POLINOMIO}")
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

# Criar pontos para extrapolacao (fora do intervalo dos dados)
x_extrapolacao = np.linspace(-2, max(x) + 5, 1000)
y_extrapolacao = polinomio(x_extrapolacao)

# Criacao do grafico de extrapolacao
plt.figure(figsize=(14, 8))

# Plotar a curva completa incluindo extrapolacao
plt.plot(x_extrapolacao, y_extrapolacao, color='blue', linewidth=2.5,
         label=f'Polinômio grau {GRAU_POLINOMIO}', alpha=0.9, zorder=2)

# Destacar a regiao dos dados experimentais (interpolacao)
plt.axvspan(min(x), max(x), color='lightgreen', alpha=0.25,
            label='Região de interpolação (dados)', zorder=1)

# Marcar as regioes de extrapolacao
plt.axvspan(-2, min(x), color='orange', alpha=0.15, zorder=1)
plt.axvspan(max(x), max(x) + 5, color='orange', alpha=0.15, zorder=1)

# Plotar os pontos experimentais
plt.scatter(x, y, color='red', s=150, zorder=5, label='Dados experimentais',
            edgecolors='black', linewidths=2)

# Adicionar linhas verticais marcando os limites dos dados
plt.axvline(x=min(x), color='green', linestyle='--', linewidth=2, alpha=0.6)
plt.axvline(x=max(x), color='green', linestyle='--', linewidth=2, alpha=0.6)

# Adicionar texto indicando as regioes
# Calcular uma posicao adequada para o texto baseada nos limites do grafico
y_min_grafico, y_max_grafico = plt.ylim()
y_texto = y_max_grafico * 0.9

plt.text(-0.5, y_texto, 'Extrapolação', fontsize=11, color='darkorange',
         fontweight='bold', ha='center', bbox=dict(boxstyle='round,pad=0.5',
         facecolor='orange', alpha=0.3))

plt.text((min(x) + max(x)) / 2, y_texto, 'Interpolação', fontsize=11,
         color='darkgreen', fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

plt.text(max(x) + 2.5, y_texto, 'Extrapolação', fontsize=11, color='darkorange',
         fontweight='bold', ha='center', bbox=dict(boxstyle='round,pad=0.5',
         facecolor='orange', alpha=0.3))

# Configuracoes do grafico
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.title(f'Análise de Extrapolação - Polinômio de Grau {GRAU_POLINOMIO}',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')

# Ajustar limites para visualizar todo o comportamento
plt.xlim(-2, max(x) + 5)

plt.tight_layout()

# Nome do arquivo baseado no grau
nome_arquivo = f'./relatorio-12/extrapolacao_grau_{GRAU_POLINOMIO}.png'
plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')

print(f"\n" + "=" * 70)
print(f"Grafico salvo como '{nome_arquivo}'")
print("=" * 70)

# Analise de extrapolacao
print(f"\n" + "=" * 70)
print("ANALISE DE EXTRAPOLACAO")
print("=" * 70)

# Calcular valores em pontos de extrapolacao
x_extrap_esq = -1
x_extrap_dir = max(x) + 3
y_extrap_esq = polinomio(x_extrap_esq)
y_extrap_dir = polinomio(x_extrap_dir)

print(f"\nExtrapolação à esquerda (x = {x_extrap_esq}):")
print(f"  y = {y_extrap_esq:.2f}")

print(f"\nExtrapolação à direita (x = {x_extrap_dir}):")
print(f"  y = {y_extrap_dir:.2f}")

print(f"\nIntervalo de y nos dados experimentais: [{min(y):.2f}, {max(y):.2f}]")

if GRAU_POLINOMIO >= n - 1:
    print(f"\n⚠ ALTO RISCO DE EXTRAPOLACAO INSTAVEL!")
    print(f"Grau {GRAU_POLINOMIO} >= n-1 causa oscilações extremas fora dos dados.")
    print(f"NÃO use este modelo para prever valores fora do intervalo [{min(x)}, {max(x)}]!")
elif GRAU_POLINOMIO >= 4:
    print(f"\n⚠ CUIDADO com extrapolação!")
    print(f"Grau {GRAU_POLINOMIO} pode ter comportamento instável fora dos dados.")
    print(f"Use com cautela para prever valores fora de [{min(x)}, {max(x)}].")
elif GRAU_POLINOMIO <= 2:
    print(f"\n✓ Extrapolação relativamente segura para grau {GRAU_POLINOMIO}.")
    print(f"Modelos lineares/quadráticos tendem a ter comportamento mais estável.")
else:
    print(f"\n⚠ Extrapolação moderadamente arriscada para grau {GRAU_POLINOMIO}.")
    print(f"Verifique se os valores extrapolados fazem sentido fisicamente.")

print("=" * 70)
print(f"\nPara testar outro grau, altere a variável GRAU_POLINOMIO no topo do código.")
print("=" * 70)

plt.show()
