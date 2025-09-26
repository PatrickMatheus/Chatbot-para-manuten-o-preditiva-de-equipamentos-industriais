import numpy as np
import matplotlib.pyplot as plt

print("\n\n--- ANÁLISE DO CASO 2: MAIS DADOS (9 PONTOS) ---")

t_data_9 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
y_data_9 = np.array([6.99, 2.28, 1.91, 11.94, 14.60, 22.30, 37.85, 50.20, 70.18])
ny = len(t_data_9)

theta_true = np.array([5.0, 0.2, 0.1])
L_true = 1.0
t_plot = np.arange(0, 12.5, 0.5)
X_true_plot = np.c_[np.ones_like(t_plot), L_true * t_plot**2, t_plot**3]
z_true = X_true_plot @ theta_true

plt.figure(figsize=(8, 6))
plt.plot(t_data_9, y_data_9, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')

print("Resultados do Ajuste com 9 pontos:")
print("-" * 40)
print(f"{'Ordem':<8} | {'R²':<12} | {'R² Ajustado':<15}")
print("-" * 40)

for order in [2, 3]:
    X = np.vander(t_data_9, order + 1, increasing=True)
    np_params = order + 1
    theta_hat, _, _, _ = np.linalg.lstsq(X, y_data_9, rcond=None)
    
    y_mean = np.mean(y_data_9)
    z_hat_data = X @ theta_hat
    sse = np.sum((y_data_9 - z_hat_data)**2)
    sst = np.sum((y_data_9 - y_mean)**2)
    r_squared = 1 - (sse / sst)
    adj_r_squared = 1 - (1 - r_squared) * (ny - 1) / (ny - np_params)
    
    print(f"{order:<8} | {r_squared:<12.4f} | {adj_r_squared:<15.4f}")

    X_plot_poly = np.vander(t_plot, order + 1, increasing=True)
    z_hat_plot = X_plot_poly @ theta_hat
    plt.plot(t_plot, z_hat_plot, '--', label=f'Pred. {order}ª Ord')

print("-" * 40)

plt.title('Fig 2.6: Prediction with 9 Data Points')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 2 ---")
print("Com mais dados, os valores de R² Ajustado para ambos os modelos são altos e muito próximos.")
print("O gráfico mostra que as predições são muito melhores e mais parecidas entre si.")
print("Mais dados ajudaram os modelos a 'enxergar' a tendência real através do ruído.")