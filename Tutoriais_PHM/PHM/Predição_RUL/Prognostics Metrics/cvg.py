import numpy as np
import matplotlib.pyplot as plt

rul_prevista = np.array([np.nan, np.nan, np.nan, np.nan, 12.2072, 6.5588, 11.4634, 3.4997, 2.2000, 1.3270, 0.2674])
ciclos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
eol_real = 10.27
t_inicio = 4  

indices_validos = np.where(ciclos >= t_inicio)[0]
rul_prev = rul_prevista[indices_validos]
ciclos_eval = ciclos[indices_validos]
rul_real = eol_real - ciclos_eval

erro_relativo = np.abs(rul_prev - rul_real) / rul_real

numerador_tc = 0
numerador_ec = 0
denominador = 0

for k in range(len(ciclos_eval)-1):
    tk = ciclos_eval[k]
    tk1 = ciclos_eval[k+1]
    Ek = erro_relativo[k]
    delta_t = tk1 - tk
    numerador_tc += (tk + tk1) * Ek * delta_t
    numerador_ec += (Ek**2) * delta_t
    denominador += Ek * delta_t

tc = 0.5*(numerador_tc / denominador)
ec = 0.5*(numerador_ec / denominador)

cvg = np.sqrt((tc - t_inicio)**2 + ec**2)

print("--- Métrica: Convergência (Cvg) ---")
print(f"Ciclo médio ponderado pelo erro (tc): {tc:.4f}")
print(f"Erro médio ponderado (ec): {ec:.4f}")
print(f"Convergência (Cvg): {cvg:.4f}")

plt.figure(figsize=(10,6))
plt.plot(ciclos_eval, erro_relativo, 'o-b', label='Erro Relativo')
plt.fill_between(ciclos_eval, erro_relativo, color='blue', alpha=0.2, label='Área do Erro')
plt.plot(tc, ec, 'r*', markersize=15, label=f'Centro de Massa ({tc:.2f}, {ec:.2f})')
plt.title('Visualização da Convergência (Cvg)')
plt.xlabel('Ciclos')
plt.ylabel('Erro Relativo')
plt.legend()
plt.grid(True)
plt.show()
