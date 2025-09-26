import numpy as np
import matplotlib.pyplot as plt


rul_pred = np.array([np.nan, np.nan, np.nan, np.nan, 12.2072, 6.5588, 11.4634, 3.4997, 2.2000, 1.3270, 0.2674])
ciclos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
fim_vida_real = 10.27
ciclo_inicial = 4
alfa = 0.05
lambda_val = 0.5


idx_validos = np.where(ciclos >= ciclo_inicial)[0]
rul_prev_aval = rul_pred[idx_validos]
ciclos_aval = ciclos[idx_validos]
rul_real_aval = fim_vida_real - ciclos_aval

t_lambda = ciclo_inicial + (fim_vida_real - ciclo_inicial) * lambda_val
idx_lambda = np.argmin(np.abs(ciclos_aval - t_lambda))
ciclo_teste = ciclos_aval[idx_lambda]


margem_ala = rul_real_aval[idx_lambda] * alfa
erro_no_ponto = np.abs(rul_prev_aval[idx_lambda] - rul_real_aval[idx_lambda])
ala_ok = erro_no_ponto <= margem_ala

print("--- Métrica ALA ---")
print(f"No ciclo {ciclo_teste}: ALA = {ala_ok}")


plt.figure(figsize=(10, 5))
plt.plot(ciclos_aval, rul_real_aval, 'k-', linewidth=2, label='RUL Real')
plt.plot(ciclos_aval, rul_prev_aval, '.--r', label='RUL Prevista')

lim_sup = rul_real_aval * (1 + alfa)
lim_inf = rul_real_aval * (1 - alfa)
plt.fill_between(ciclos_aval, lim_inf, lim_sup, color='orange', alpha=0.3, label=f'Zona de Aceitação (α={alfa*100}%)')


plt.axvline(x=ciclo_teste, color='b', linestyle=':', linewidth=2, label=f'Ciclo {ciclo_teste}')
plt.plot(ciclo_teste, rul_prev_aval[idx_lambda], 'ro', markersize=10, label='Previsão')
plt.plot(ciclo_teste, rul_real_aval[idx_lambda], 'ko', markersize=10, label='Real')

plt.title('Métrica ALA (α-λ)')
plt.xlabel('Ciclos')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()
