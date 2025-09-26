import numpy as np
import matplotlib.pyplot as plt

rul_pred = np.array([np.nan, np.nan, np.nan, np.nan, 12.2072, 6.5588, 11.4634, 3.4997, 2.2000, 1.3270, 0.2674])
cycle = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
eol_true = 10.27
t_s = 4

valid_indices = np.where(cycle >= t_s)[0]
rul_pred_eval = rul_pred[valid_indices]
cycle_eval = cycle[valid_indices]
rul_true_eval = eol_true - cycle_eval
relative_errors = np.abs(rul_pred_eval - rul_true_eval) / rul_true_eval
all_ras = 1 - relative_errors
cra = np.mean(all_ras)

print("--- Métrica: Cumulative Relative Accuracy (CRA) ---")
print(f"A precisão relativa acumulada (CRA) é: {cra:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(cycle_eval, all_ras, 'o-b', label='Precisão Relativa (RA) a cada ciclo')
plt.axhline(y=cra, color='r', linestyle='--', linewidth=2, label=f'Média Geral (CRA = {cra:.2f})')
plt.axhline(y=1.0, color='k', linestyle=':', linewidth=2, label='100% de Precisão')

plt.title('Visualização da Cumulative Relative Accuracy (CRA)')
plt.xlabel('Ciclos')
plt.ylabel('Nível de Precisão (RA)')
plt.ylim(bottom=-1) 
plt.legend()
plt.grid(True)
plt.show()