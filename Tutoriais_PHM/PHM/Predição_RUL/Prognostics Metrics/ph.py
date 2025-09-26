import numpy as np
import matplotlib.pyplot as plt

rul_prevista = np.array([np.nan, np.nan, np.nan, np.nan, 12.2072, 6.5588, 11.4634, 3.4997, 2.2000, 1.3270, 0.2674])
ciclos = np.arange(len(rul_prevista)) 
vida_total = 10.27                     
alpha = 0.05                             
ciclo_inicial_valido = 4                


mascara_valida = ciclos >= ciclo_inicial_valido
ciclos_aval = ciclos[mascara_valida]
rul_prevista_aval = rul_prevista[mascara_valida]
rul_verdadeira = vida_total - ciclos_aval
margem = vida_total * alpha

na_zona = np.abs(rul_prevista_aval - rul_verdadeira) <= margem
indices_na_zona = np.where(na_zona)[0]

if len(indices_na_zona) == 0:
    ciclo_inicio_ph = np.nan
    ph = 0
else:
    pulos = np.where(np.diff(indices_na_zona) > 1)[0]
    indice_inicio = indices_na_zona[pulos[-1] + 1] if len(pulos) > 0 else indices_na_zona[0]
    ciclo_inicio_ph = ciclos_aval[indice_inicio]
    ph = vida_total - ciclo_inicio_ph

print(f"--- Métrica: Horizonte Prognóstico (PH) ---")
print(f"PH = {ph:.4f}, início do PH no ciclo = {ciclo_inicio_ph}")


plt.figure(figsize=(10,6))

plt.plot(ciclos_aval, rul_verdadeira, 'k-', linewidth=2, label='RUL Verdadeira')

plt.plot(ciclos_aval, rul_prevista_aval, '.--r', label='RUL Prevista')

plt.fill_between(ciclos_aval, rul_verdadeira - margem, rul_verdadeira + margem, 
                 color='gray', alpha=0.3, label=f'Zona de Acerto (α={alpha*100:.0f}%)')

if not np.isnan(ciclo_inicio_ph):
    plt.axvline(ciclo_inicio_ph, color='g', linestyle='--', linewidth=2, 
                label=f'Início do PH (Ciclo {ciclo_inicio_ph})')

plt.title('Horizonte Prognóstico (PH)')
plt.xlabel('Ciclos')
plt.ylabel('RUL (Vida Útil Remanescente)')
plt.legend()
plt.grid(True)
plt.show()
