import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, multivariate_t

y_data_raw = np.array([6.99, 2.28, 1.91, 11.94, 14.60])
t_data = np.array([0, 1, 2, 3, 4])
h1_known = 5.0

y_data = y_data_raw - h1_known
X = np.c_[t_data**2, t_data**3]

ny = len(y_data)
np_params = X.shape[1]

theta_hat, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)

sse = np.sum((y_data - X @ theta_hat)**2)
sigma_hat_2 = sse / (ny - np_params)

theta_cov = sigma_hat_2 * np.linalg.inv(X.T @ X)


print(f"Melhor Estimativa (theta_hat):")
print(f"  h2 = {theta_hat[0]:.2f}")
print(f"  h3 = {theta_hat[1]:.2f}\n")
print(f"Variância do Ruído Estimada (sigma^2) = {sigma_hat_2:.2f}\n")
print("Matriz de Covariância dos Parâmetros (Incerteza):")
print(np.round(theta_cov, 2))

ns = 5000 
dof = ny - np_params 

np.random.seed(42)
param_samples = multivariate_t.rvs(loc=theta_hat, shape=theta_cov, df=dof, size=ns)


percentiles = [5, 50, 95]
param_ci = np.percentile(param_samples, percentiles, axis=0)

print("Intervalo de Confiança de 90% para os parâmetros (usando distribuição t):")
print(f"h2: Mediana={param_ci[1, 0]:.2f}, IC=[{param_ci[0, 0]:.2f}, {param_ci[2, 0]:.2f}]")
print(f"h3: Mediana={param_ci[1, 1]:.2f}, IC=[{param_ci[0, 1]:.2f}, {param_ci[2, 1]:.2f}]")

fig1, axs = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Parte (b): Distribuição dos Parâmetros Amostrados')
axs[0].hist(param_samples[:, 0], bins=30, edgecolor='k')
axs[0].set_title('Distribuição de h2 (t-Student)')
axs[0].set_xlabel('Valor de h2')
axs[1].hist(param_samples[:, 1], bins=30, edgecolor='k')
axs[1].set_title('Distribuição de h3 (t-Student)')
axs[1].set_xlabel('Valor de h3')

fig2 = plt.figure(figsize=(6, 6))
plt.plot(param_samples[:, 0], param_samples[:, 1], '.', color='gray', alpha=0.3)
plt.axhline(0, color='k', linestyle=':')
plt.axvline(0, color='k', linestyle=':')
plt.title('Parte (b): Nuvem de Probabilidade (h2 vs h3)')
plt.xlabel('Valores de h2')
plt.ylabel('Valores de h3')
plt.grid(True)


t_new = np.arange(4, 16)

X_new = np.c_[t_new**2, t_new**3]

z_hat_samples = h1_known + X_new @ param_samples.T

z_ci = np.percentile(z_hat_samples, percentiles, axis=1)

future_noise = t.rvs(df=dof, size=z_hat_samples.shape) * np.sqrt(sigma_hat_2)
z_pred_samples = h1_known + (X_new @ param_samples.T) + future_noise 

z_pi = np.percentile(z_pred_samples, percentiles, axis=1)

theta_true = np.array([0.2, 0.1])
z_true = h1_known + X_new @ theta_true

fig3 = plt.figure(figsize=(10, 7))
plt.plot(t_data, y_data_raw, 'sk', markersize=8, label='Dados (Data)')
plt.plot(t_new, z_true, 'k-', linewidth=2, label='Verdadeiro (True)')
plt.plot(t_new, z_ci[1, :], '--r', linewidth=2, label='Mediana (Median)')
plt.plot(t_new, z_ci[0, :], ':r', label='90% I.C. (C.I.)')
plt.plot(t_new, z_ci[2, :], ':r')
plt.plot(t_new, z_pi[0, :], '-.b', label='90% I.P. (P.I.)')
plt.plot(t_new, z_pi[2, :], '-.b')
plt.axhline(150, color='g', linestyle='-', label='Limiar (Threshold)')
plt.title('Parte (c): Previsão de Degradação com Incerteza')
plt.xlabel('Ciclos'); plt.ylabel('Degradação')
plt.legend(); plt.grid(True); plt.ylim(-50, 250)
 
threshold = 150.0
rul_samples = []

for i in range(ns):
    current_curve = z_pred_samples[:, i]
    cross_indices = np.where(current_curve >= threshold)[0]
    
    if len(cross_indices) > 0:
        first_cross_idx = cross_indices[0]
        if first_cross_idx > 0:
            p1_t, p1_z = t_new[first_cross_idx-1], current_curve[first_cross_idx-1]
            p2_t, p2_z = t_new[first_cross_idx], current_curve[first_cross_idx]
            eol = np.interp(threshold, [p1_z, p2_z], [p1_t, p2_t])
            rul = eol - t_data[-1]
            rul_samples.append(rul)


fig4 = plt.figure(figsize=(8, 5))
plt.hist(rul_samples, bins=30, edgecolor='k', alpha=0.7)
plt.title(f'Parte (d): Distribuição da RUL (a partir do ciclo {t_data[-1]})')
plt.xlabel('RUL Prevista (ciclos)')
plt.ylabel('Frequência (Nº de Ocorrências)')
plt.grid(True)

plt.show()