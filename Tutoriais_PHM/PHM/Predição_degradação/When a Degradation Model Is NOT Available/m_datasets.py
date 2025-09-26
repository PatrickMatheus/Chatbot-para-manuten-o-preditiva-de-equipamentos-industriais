import numpy as np
import matplotlib.pyplot as plt

print("\n\n--- ANÁLISE DO CASO 3: MÚLTIPLOS CONJUNTOS DE DADOS (31 PONTOS) ---")

t_L1 = np.arange(0, 9)
y_L1 = np.array([6.99, 2.28, 1.91, 11.94, 14.60, 22.30, 37.85, 50.20, 70.18])

t_L05 = np.arange(0, 11)
y_L05 = np.array([0.46, 1.17, 9.43, 10.55, 11.17, 24.50, 25.54, 43.59, 61.42, 88.66, 117.95])

t_L2 = np.arange(0, 11)
y_L2 = np.array([1.87, 5.40, 6.86, 12.76, 19.89, 30.05, 38.76, 60.70, 83.35, 106.93, 141.19])

t_data_31 = np.concatenate([t_L1, t_L05, t_L2])
y_data_31 = np.concatenate([y_L1, y_L05, y_L2])
ny = len(t_data_31)

theta_true = np.array([5.0, 0.2, 0.1])
L_true = 1.0
t_plot = np.arange(0, 12.5, 0.5)
X_true_plot = np.c_[np.ones_like(t_plot), L_true * t_plot**2, t_plot**3]
z_true = X_true_plot @ theta_true

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t_L1, y_L1, 'o', label='L=1 (Atual)')
plt.plot(t_L05, y_L05, 's', label='L=0.5 (Histórico)')
plt.plot(t_L2, y_L2, '^', label='L=2 (Histórico)')
plt.title('Fig 2.7(a): Training Data')
plt.xlabel('Cycles'); plt.ylabel('Degradation level')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_L1, y_L1, 'ok', markersize=7, label='Data (L=1)')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')

print("Resultados do Ajuste com 31 pontos:")
print("-" * 40)
print(f"{'Ordem':<8} | {'R²':<12} | {'R² Ajustado':<15}")
print("-" * 40)

for order in [2, 3]:
    X = np.vander(t_data_31, order + 1, increasing=True)
    np_params = order + 1
    theta_hat, _, _, _ = np.linalg.lstsq(X, y_data_31, rcond=None)
    
    y_mean = np.mean(y_data_31)
    z_hat_data = X @ theta_hat
    sse = np.sum((y_data_31 - z_hat_data)**2)
    sst = np.sum((y_data_31 - y_mean)**2)
    r_squared = 1 - (sse / sst)
    adj_r_squared = 1 - (1 - r_squared) * (ny - 1) / (ny - np_params)
    
    print(f"{order:<8} | {r_squared:<12.4f} | {adj_r_squared:<15.4f}")

    X_plot_poly = np.vander(t_plot, order + 1, increasing=True)
    z_hat_plot = X_plot_poly @ theta_hat
    plt.plot(t_plot, z_hat_plot, '--', label=f'Pred. {order}ª Ord')

print("-" * 40)

plt.title('Fig 2.7(b): Predictions')
plt.xlabel('Cycles'); plt.ylabel('Degradation level')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- CONCLUSÃO DO CASO 3 ---")
print("Usando um grande conjunto de dados de múltiplos sistemas, a predição se torna muito precisa.")
print("A abordagem orientada a dados, quando alimentada com dados suficientes e variados, pode compensar a falta de um modelo físico.")