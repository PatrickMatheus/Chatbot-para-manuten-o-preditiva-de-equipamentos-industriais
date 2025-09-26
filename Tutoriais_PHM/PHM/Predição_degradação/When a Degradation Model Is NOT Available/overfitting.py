import numpy as np
import matplotlib.pyplot as plt

print("--- ANÁLISE DO CASO 1: POUCOS DADOS (5 PONTOS) ---")

t_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([6.99, 2.28, 1.91, 11.94, 14.60])
ny = len(t_data)


theta_true = np.array([5.0, 0.2, 0.1])
L_true = 1.0
t_plot = np.arange(0, 10.5, 0.5)
X_true_plot = np.c_[np.ones_like(t_plot), L_true * t_plot**2, t_plot**3]
z_true = X_true_plot @ theta_true

plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='True')

print("Resultados do Ajuste (Tabela 2.3):")
print("-" * 40)
print(f"{'Ordem':<8} | {'R²':<12} | {'R² Ajustado':<15}")
print("-" * 40)

for order in [1, 2, 3]:
    X = np.vander(t_data, order + 1, increasing=True)
    np_params = order + 1 # Número de parâmetros do modelo

    theta_hat, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)
    
    y_mean = np.mean(y_data)
    z_hat_data = X @ theta_hat
    sse = np.sum((y_data - z_hat_data)**2)
    sst = np.sum((y_data - y_mean)**2)
    r_squared = 1 - (sse / sst)
    
    adj_r_squared = 1 - (1 - r_squared) * (ny - 1) / (ny - np_params)
    
    print(f"{order:<8} | {r_squared:<12.4f} | {adj_r_squared:<15.4f}")

    X_plot_poly = np.vander(t_plot, order + 1, increasing=True)
    z_hat_plot = X_plot_poly @ theta_hat
    plt.plot(t_plot, z_hat_plot, '--', label=f'Pred. {order}ª Ord')

print("-" * 40)

plt.title('Fig 2.5: Prediction with 5 Data Points (Data-Driven)')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.ylim(-60, 160)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 1 ---")
print("O modelo de 3ª ordem tem o maior R² Ajustado, sugerindo o melhor *ajuste aos dados existentes*.")
print("PORÉM, o gráfico mostra que ele é o PIOR para *prever o futuro*, com um comportamento fisicamente implausível.")
print("Isso é OVERFITTING: o modelo se ajustou ao ruído dos dados, não à tendência real.")