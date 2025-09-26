import numpy as np
import matplotlib.pyplot as plt

print("\n\n--- ANÁLISE DO CASO 3: SUPOSIÇÃO INCORRETA (L=2) ---")

t_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([5.0, 5.3, 6.6, 9.5, 14.6])
L_wrong = 2.0 


X_wrong = np.c_[np.ones_like(t_data), L_wrong * t_data**2, t_data**3]
print("Matriz de Design (X) gerada com L=2:")
print(X_wrong)


theta_hat_wrong, _, _, _ = np.linalg.lstsq(X_wrong, y_data, rcond=None)

t_plot = np.arange(0, 15.5, 0.5)
X_plot_wrong = np.c_[np.ones_like(t_plot), L_wrong * t_plot**2, t_plot**3]
z_hat_wrong = X_plot_wrong @ theta_hat_wrong

theta_true = np.array([5.0, 0.2, 0.1])
L_true = 1.0
X_plot_true = np.c_[np.ones_like(t_plot), L_true * t_plot**2, t_plot**3]
z_true = X_plot_true @ theta_true

plt.figure(figsize=(8, 6))
plt.plot(t_data, y_data, 'ok', markersize=9, label='Data')
plt.plot(t_plot, z_true, 'k-', linewidth=2, label='θ_true with L=1')
plt.plot(t_plot, z_hat_wrong, 'r--', linewidth=2, label='θ_pred with L=2')
plt.title('Fig E2.1: Prediction with Incorrect Loading Condition')
plt.xlabel('Cycles')
plt.ylabel('Degradation level')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DO CASO 3 ---")
print(f"Parâmetros estimados (theta_hat_wrong): {[round(p, 4) for p in theta_hat_wrong]}")
print(f"Parâmetros verdadeiros (theta_true):     {[round(p, 4) for p in theta_true]}")
print("Resultado: Os parâmetros estimados são diferentes dos verdadeiros.")
print("\nMesmo com parâmetros errados, a predição final foi PERFEITA.")
print("Isso acontece por causa da CORRELAÇÃO entre L e θ2.")
print(f"  - No modelo verdadeiro: L*θ2 = {L_true} * {theta_true[1]} = {L_true * theta_true[1]}")
print(f"  - No modelo estimado:   L*θ2 = {L_wrong} * {round(theta_hat_wrong[1], 1)} = {L_wrong * round(theta_hat_wrong[1], 1)}")
print("O efeito combinado do termo (L*θ2) permaneceu o mesmo, então o modelo se auto-corrigiu, 'escondendo' nosso erro de suposição.")