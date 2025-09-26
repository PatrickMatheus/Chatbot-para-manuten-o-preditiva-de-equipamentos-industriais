import numpy as np
import matplotlib.pyplot as plt

print("\n\n--- ANÁLISE DO CASO 2(a): DADOS COM RUÍDO ---")

t_data = np.array([0, 1, 2, 3, 4])
y_data_noisy = np.array([6.99, 2.28, 1.91, 11.94, 14.60])
L = 1.0

X_noisy = np.c_[np.ones_like(t_data), L * t_data**2, t_data**3]
print(X_noisy)

theta_hat_noisy, _, _, _ = np.linalg.lstsq(X_noisy, y_data_noisy, rcond=None)

t_plot = np.arange(0, 15.5, 0.5)
X_plot = np.c_[np.ones_like(t_plot), L * t_plot**2, t_plot**3]
z_hat_noisy = X_plot @ theta_hat_noisy # Predição com params "ruidosos"

theta_true = np.array([5.0, 0.2, 0.1])
z_true = X_plot @ theta_true 

plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data_noisy, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')
plt.plot(t_plot, z_hat_noisy, 'r--', linewidth=2, label='Prediction')
plt.title('Fig 2.3(a): Prediction with Noisy Data')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 2(a) ---")
print(f"Parâmetros estimados (theta_hat_noisy): {[round(p, 2) for p in theta_hat_noisy]}")
print(f"Parâmetros verdadeiros (theta_true):     {[round(p, 2) for p in theta_true]}")
print("Resultado: Os parâmetros estimados agora são DIFERENTES dos verdadeiros.")
print("O ruído nos dados de medição 'enganou' o algoritmo, que encontrou uma curva ligeiramente diferente para se ajustar aos pontos.")

#=================================================================#

print("\n\n--- ANÁLISE DO CASO 2(b): DADOS COM RUÍDO ---")

t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
y_data_noisy = np.array([6.99, 2.28, 1.91, 11.94, 14.60, 22.30, 37.85,50.20,70.18])
L = 1.0

X_noisy = np.c_[np.ones_like(t_data), L * t_data**2, t_data**3]

theta_hat_noisy, _, _, _ = np.linalg.lstsq(X_noisy, y_data_noisy, rcond=None)

t_plot = np.arange(0, 15.5, 0.5)
X_plot = np.c_[np.ones_like(t_plot), L * t_plot**2, t_plot**3]
z_hat_noisy = X_plot @ theta_hat_noisy 

theta_true = np.array([5.0, 0.2, 0.1])
z_true = X_plot @ theta_true 

plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data_noisy, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')
plt.plot(t_plot, z_hat_noisy, 'r--', linewidth=2, label='Prediction')
plt.title('Fig 2.3(b): Prediction with Noisy Data')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 2(b) ---")
print(f"Parâmetros estimados (theta_hat_noisy): {[round(p, 2) for p in theta_hat_noisy]}")
print(f"Parâmetros verdadeiros (theta_true):     {[round(p, 2) for p in theta_true]}")
print("Os parâmetros ficaram bem mais próximos dos reais")
print("A medida que eu aumento os dados, a interferência do ruído vai se tornando mínima, e a curva se aproxima")