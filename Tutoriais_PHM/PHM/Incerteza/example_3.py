# Arquivo: exemplo_2_7_completo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, multivariate_t

y_data_full = np.array([6.99, 2.28, 1.91, 11.94, 14.60, 22.30, 37.85, 50.20, 70.18, 105.3, 142.6])
t_data_full = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
h1_known = 5.0
ns = 5000
percentiles = [5, 50, 95]
threshold = 150.0

all_degradation_results = {}
all_rul_percentiles = np.full((len(t_data_full), 3), np.nan) # [ciclo, (5, 50, 95)]

for k in range(3, len(t_data_full) + 1):
    current_time_idx = k - 1
    current_time = t_data_full[current_time_idx]
    print(f"Analisando no ciclo k={current_time} (usando {k} pontos de dados)...")
    
    t_data = t_data_full[:k]
    y_data_raw = y_data_full[:k]
    
    y_data = y_data_raw - h1_known
    X = np.c_[t_data**2, t_data**3]
    ny = len(y_data)
    np_params = X.shape[1]
    dof = ny - np_params

    if dof > 0:
        theta_hat, _, _, _ = np.linalg.lstsq(X, y_data, rcond=None)
        sse = np.sum((y_data - X @ theta_hat)**2)
        sigma_hat_2 = sse / dof
        theta_cov = sigma_hat_2 * np.linalg.inv(X.T @ X)
        
        param_samples = multivariate_t.rvs(loc=theta_hat, shape=theta_cov, df=dof, size=ns)
        
        t_new = np.arange(current_time, 20)
        X_new = np.c_[t_new**2, t_new**3]
        
        z_hat_samples = h1_known + X_new @ param_samples.T
        future_noise = t.rvs(df=dof, size=z_hat_samples.shape) * np.sqrt(sigma_hat_2)
        z_pred_samples = z_hat_samples + future_noise
        
        if current_time in [6, 10]: # O livro usa k=7 e k=11, que são os ciclos 6 e 10
            all_degradation_results[current_time] = {
                't_new': t_new,
                'z_ci': np.percentile(z_hat_samples, percentiles, axis=1),
                'z_pi': np.percentile(z_pred_samples, percentiles, axis=1)
            }
        
        rul_samples = []
        for i in range(ns):
            cross_indices = np.where(z_pred_samples[:, i] >= threshold)[0]
            if len(cross_indices) > 0:
                first_cross_idx = cross_indices[0]
                if first_cross_idx > 0:
                    p1_t, p1_z = t_new[first_cross_idx-1], z_pred_samples[first_cross_idx-1, i]
                    p2_t, p2_z = t_new[first_cross_idx], z_pred_samples[first_cross_idx, i]
                    eol = np.interp(threshold, [p1_z, p2_z], [p1_t, p2_t])
                    rul = eol - current_time
                    rul_samples.append(rul)
        
        if rul_samples:
            all_rul_percentiles[current_time_idx, :] = np.percentile(rul_samples, percentiles)


print("\n--- (a) Gerando Gráficos de Degradação ---")
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for i, cycle_to_plot in enumerate([6, 10]):
    res = all_degradation_results[cycle_to_plot]
    ax = axs[i]
    ax.plot(t_data_full[:cycle_to_plot+1], y_data_full[:cycle_to_plot+1], 'sk', markersize=8, label='Dados Usados')
    ax.plot(res['t_new'], res['z_ci'][1, :], '--r', label='Mediana')
    ax.plot(res['t_new'], res['z_ci'][0, :], ':r', label='90% I.C.')
    ax.plot(res['t_new'], res['z_ci'][2, :], ':r')
    ax.plot(res['t_new'], res['z_pi'][0, :], '-.b', label='90% I.P.')
    ax.plot(res['t_new'], res['z_pi'][2, :], '-.b')
    ax.axhline(threshold, color='g', linestyle='-', label='Limiar')
    ax.set_title(f'Previsão no Ciclo {cycle_to_plot}')
    ax.set_xlabel('Ciclos'); ax.set_ylabel('Degradação')
    ax.legend(); ax.grid(True); ax.set_ylim(-50, 250)
plt.show()

print("\n--- (b) Gerando Gráfico da Evolução da RUL ---")
plt.figure(figsize=(8, 8))

eol_true_approx = 10.69
plt.plot([0, eol_true_approx], [eol_true_approx, 0], 'k-', linewidth=3, label='RUL Verdadeira')

plt.plot(t_data_full, all_rul_percentiles[:, 1], 'o--r', linewidth=2, label='RUL Mediana Prevista')

plt.plot(t_data_full, all_rul_percentiles[:, 0], ':r', linewidth=2, label='90% I.P. da RUL')
plt.plot(t_data_full, all_rul_percentiles[:, 2], ':r', linewidth=2)

plt.title('Evolução da Previsão de RUL com Incerteza')
plt.xlabel('Ciclos'); plt.ylabel('RUL')
plt.axis([0, eol_true_approx, 0, eol_true_approx])
plt.legend(); plt.grid(True)
plt.show()

