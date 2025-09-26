import numpy as np
import matplotlib.pyplot as plt

print("--- ANÁLISE DA SEÇÃO 2.3.1: PREVISÃO DE RUL ---")

L = 2.0
threshold = 150.0

t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([1.87, 5.40, 6.86, 12.76, 19.89, 30.05, 38.76, 60.70, 83.35, 106.93, 141.19])

rul_predictions = np.full_like(t_data, np.nan, dtype=float)

for i in range(4, len(t_data)):
    current_time = t_data[i]
    num_points = i + 1
    
    t_current_slice = t_data[:num_points]
    y_current_slice = y_data[:num_points]

    X = np.c_[np.ones(num_points), L * t_current_slice**2, t_current_slice**3]
    theta_hat, _, _, _ = np.linalg.lstsq(X, y_current_slice, rcond=None)

    c3 = theta_hat[2]
    c2 = theta_hat[1] * L 
    c1 = 0
    c0 = theta_hat[0] - threshold
    
    solutions = np.roots([c3, c2, c1, c0])

    real_positive_solutions = [sol.real for sol in solutions if np.isreal(sol) and sol.real > current_time]
    if real_positive_solutions:
        eol = min(real_positive_solutions)

        rul = eol - current_time
        rul_predictions[i] = rul

plt.figure(figsize=(8, 6))

print(rul_predictions)

plt.plot(t_data, rul_predictions, '.--r', markersize=12, linewidth=2, label='Prediction')

true_eol = 10.27 
true_rul = true_eol - t_data
plt.plot(t_data, true_rul, 'k-', linewidth=2, label='True')

plt.title('Fig E2.3: RUL Prediction')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.axis([0, 10, 0, 14])
plt.legend()
plt.grid(True)
plt.show()

print("\n--- CONCLUSÃO DA SEÇÃO 2.3.1 ---")
print("O gráfico mostra que a previsão da RUL (vermelho) começa com um erro grande.")
print("À medida que mais dados são coletados (avançamos nos ciclos), a previsão converge e se aproxima da RUL verdadeira (preto).")
print("Isso demonstra que a confiança na previsão aumenta com a quantidade de informação disponível.")