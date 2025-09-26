# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt


t = np.array([0, 10, 20, 30, 40, 50, 60, 70])
y = np.array([5.1, 6.3, 6.9, 8.1, 9.2, 10.1, 11.2, 11.8]) 


X = np.c_[np.ones(len(t)), t]


theta_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

print(f"Matriz de Design (X) criada:\n{X}")
print(f"\nParâmetros estimados (theta_hat):")
print(f"  θ1 (intercepto): {theta_hat[0]:.4f}")
print(f"  θ2 (inclinação): {theta_hat[1]:.4f}")
print(f"\nNosso modelo de degradação é: z = {theta_hat[0]:.4f} + {theta_hat[1]:.4f} * t")


print("\n--- Passo 2: Visualização ---")

plt.figure(figsize=(10, 6))
plt.plot(t, y, 'o', label='Dados Medidos (y)')
plt.plot(t, X @ theta_hat, 'r-', label='Linha de Regressão (z_hat)')

plt.title('Regressão Linear com Método dos Mínimos Quadrados')
plt.xlabel('Tempo (ciclos)')
plt.ylabel('Tamanho da Trinca (mm)')
plt.legend()
plt.grid(True)
plt.show()

print("\nO gráfico mostra como a linha vermelha (nosso modelo) se ajusta")
print("aos pontos azuis (nossas medições reais).")

