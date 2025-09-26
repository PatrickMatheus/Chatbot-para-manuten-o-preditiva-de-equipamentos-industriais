import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    df = pd.read_csv("arquivos/Bearing2_2.csv")
    pc1 = df["PC1"].values
except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Usando dados de exemplo.")
    pc1 = np.linspace(1, 0, 200)

y_data_completo = pc1[0] - pc1
t_data_completo = np.arange(len(y_data_completo))

threshold = np.percentile(y_data_completo, 95)

ponto_de_corte = int(len(y_data_completo) * 0.7)
dados_de_treino = y_data_completo[:ponto_de_corte]

p = 5
X_list, y_list = [], []
for i in range(p, len(dados_de_treino)):
    y_list.append(dados_de_treino[i])
    historico = dados_de_treino[i-p:i]
    X_list.append(historico[::-1])

y_ar = np.array(y_list)
X_ar_final = np.c_[np.ones(len(X_list)), np.array(X_list)]
theta_hat, _, _, _ = np.linalg.lstsq(X_ar_final, y_ar, rcond=None)

historico_recente = dados_de_treino[-p:].tolist()
previsao_futura = []

for _ in range(len(y_data_completo) - ponto_de_corte):
    X_pred = np.array([1] + historico_recente[::-1])
    proximo_ponto_dano = X_pred @ theta_hat
    previsao_futura.append(float(proximo_ponto_dano))
    historico_recente.pop(0)
    historico_recente.append(proximo_ponto_dano)

historico_completo = np.concatenate([dados_de_treino, previsao_futura])
t_historico = np.arange(len(historico_completo))

plt.figure(figsize=(12, 6))
plt.plot(t_data_completo, y_data_completo, 'k-', label='Dados Reais')
plt.plot(t_historico, historico_completo, 'r-', label='Dados Previstos')
plt.axhline(threshold, color='g', linestyle='--', label='Threshold')
plt.title('Dano Acumulado: Dados Reais x Previstos')
plt.xlabel('Ciclos')
plt.ylabel('Dano Acumulado')
plt.legend()
plt.grid(True)

indices_falha_prev = np.where(historico_completo >= threshold)[0]
t_falha_prev = indices_falha_prev[0] if len(indices_falha_prev) > 0 else len(historico_completo)

indices_falha_real = np.where(y_data_completo >= threshold)[0]
t_falha_real = indices_falha_real[0] if len(indices_falha_real) > 0 else len(y_data_completo)

rul_completo = t_falha_prev - np.arange(len(historico_completo))
rul_real = t_falha_real - np.arange(len(y_data_completo))

alpha = 0.05
l = 0.5
t_lambda = t_data_completo[0] + (t_falha_real - t_data_completo[0]) * l
idx_lambda = np.argmin(np.abs(t_data_completo - t_lambda))

margem_alfa = rul_real[idx_lambda] * alpha
erro_ponto = np.abs(rul_real[idx_lambda] - rul_completo[idx_lambda])
ala_ok = erro_ponto <= margem_alfa
print(f"ALA OK: {ala_ok}")

plt.figure(figsize=(12, 6))
plt.plot(t_data_completo, rul_real, marker='.', linestyle='-', color='blue', label='RUL Real')
plt.plot(t_historico, rul_completo, marker='.', linestyle='--', color='red', label='RUL Previsto')


lim_sup = rul_real * (1 + alpha)
lim_inf = rul_real * (1 - alpha)
plt.fill_between(t_data_completo, lim_inf, lim_sup, color='orange', alpha=0.3, label=f'Zona Aceitação (α={alpha*100:.1f}%)')

plt.axvline(x=t_lambda, color='b', linestyle=':', linewidth=2, label=f'Ciclo λ={t_lambda:.1f}')
plt.plot(t_lambda, rul_completo[idx_lambda], 'ro', markersize=10, label='Previsão λ')
plt.plot(t_lambda, rul_real[idx_lambda], 'ko', markersize=10, label='Real λ')

plt.axvline(t_falha_prev, color='r', linestyle='--', label=f'Falha Prevista (t={t_falha_prev})')

plt.title('Métrica ALA (α-λ)')
plt.xlabel('Ciclos')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)

plt.show()
