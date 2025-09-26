# Arquivo: exemplo_2_5.py
import numpy as np

y_data = np.array([6.99, 2.28, 1.91, 11.94, 14.60])
t_data = np.array([0, 1, 2, 3, 4])


X = np.c_[np.ones(len(t_data)), t_data]


theta_hat, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)

ny = len(y_data)
np_params = X.shape[1] 


sse = np.sum((y_data - X @ theta_hat)**2)

sigma_hat_2 = sse / (ny - np_params)

theta_cov = sigma_hat_2 * np.linalg.inv(X.T @ X)

print("--- Resultados do Exemplo 2.5 ---")
print(f"Melhor Estimativa (theta_hat):")
print(f"  θ1 (intercepto) = {theta_hat[0]:.2f}")
print(f"  θ2 (inclinação) = {theta_hat[1]:.2f}\n")

print(f"Soma dos Erros Quadrados (SSE) = {sse:.2f}")
print(f"Variância do Ruído Estimada (sigma^2) = {sigma_hat_2:.2f}\n")

print("Matriz de Covariância dos Parâmetros (Incerteza):")
print(np.round(theta_cov, 2))