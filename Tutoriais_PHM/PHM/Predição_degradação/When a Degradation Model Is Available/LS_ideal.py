import numpy as np
import matplotlib.pyplot as plt

print("--- ANÁLISE DO CASO 1: DADOS PERFEITOS ---")

t_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([5.0, 5.3, 6.6, 9.5, 14.6])
L = 1.0

X = np.c_[np.ones_like(t_data), L * t_data**2, t_data**3]
print("Matriz de Design (X) gerada:")
print(X)

theta_hat, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)

t_plot = np.arange(0, 15.5, 0.5) 
X_plot = np.c_[np.ones_like(t_plot), L * t_plot**2, t_plot**3]

z_hat = X_plot @ theta_hat

theta_true = np.array([5.0, 0.2, 0.1])
z_true = X_plot @ theta_true

plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')
plt.plot(t_plot, z_hat, 'r--', linewidth=2, label='Prediction')
plt.title('Fig 2.2: Prediction with Perfect Data')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 1 ---")
print(f"Parâmetros estimados (theta_hat): {np.round(theta_hat, 4)}")
print(f"Parâmetros verdadeiros (theta_true): {theta_true}")
print("Resultado: Os parâmetros estimados são IDÊNTICOS aos verdadeiros.")
print("Isso ocorre porque os dados de entrada eram perfeitos (sem ruído).")