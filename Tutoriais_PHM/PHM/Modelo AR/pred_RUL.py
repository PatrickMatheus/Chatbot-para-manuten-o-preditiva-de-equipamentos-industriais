import numpy as np
import dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import json
import re

def carregar_api_key():
    dotenv.load_dotenv()
    return os.environ["API_KEY"]

df = pd.read_csv("arquivos/Bearing1_2.csv")
pc1 = df["PC1"].values
t_data = np.array(range(len(pc1)))
y_data = pc1[0] - pc1

threshold = np.percentile(y_data, 95)

falha_idx = np.where(y_data >= threshold)[0][0]
rul_real = falha_idx - t_data

api_key = carregar_api_key()
client = OpenAI(api_key=api_key)

tamanho_analise = len(y_data)*0.5

historico = y_data[:int(tamanho_analise)]

prompt = f"""
Aqui está o histórico dos últimos {len(historico)} pontos.
Os pontos são: {historico}.
O limiar de falha (threshold) é {threshold:.3f}.
Os dados representam a degradação de um rolamento ao longo do tempo.
Usando técnicas de análise de séries temporais, prediga os próximos valores (APENAS NÚMEROS, SEM EXPLICAÇÕES).
Prediga os próximos {int(tamanho_analise)} pontos levando em conta aceleração da degradação conforme o tempo passa.
As suas predições devem acelerar a degradação e DEVEM atingir o threshold antes do fim dos pontos previstos.
Retorne SOMENTE um  JSON no formato: {{"predicoes": [v1, v2, v3, ..., vN]}} onde v1, v2, ... são os próximos valores previstos. 
"""


resposta = client.chat.completions.create(
    model= "gpt-5",
    messages=[
        {"role": "system", "content": "Você é um engenheiro de manutenção preditiva especializado em analise de vibrações em rolamentos. Sua tarefa é analisar séries temporais que representam a degradação de rolamentos e prever quando eles falharão."},
        {"role": "user", "content": prompt}]
)

try:
    conteudo = resposta.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', conteudo, re.DOTALL)
    if match:
        conteudo = match.group(0)
    else:
        raise ValueError("Nenhum JSON encontrado na resposta.")
    dados_pred = json.loads(conteudo)
    predicoes = dados_pred["predicoes"]
    historico_completo = np.concatenate([historico, predicoes])

except Exception as e:
    print("Erro ao decodificar a resposta do LLM:", e)

t_data_pred = np.array(range(len(historico_completo)))
idx_fail_prev = np.where(historico_completo >= threshold)[0][0]
rul_pred = idx_fail_prev - t_data_pred

alpha = 0.05
l = 0.5

t_lambda = t_data[0] + (falha_idx - t_data[0]) * l
idx_lambda = np.argmin(np.abs(t_data - t_lambda))
t_teste = t_data[idx_lambda]

margem_ala = rul_real[idx_lambda] * alpha
erro_ponto = np.abs(rul_real[idx_lambda] - rul_pred[idx_lambda])
ala_ok = erro_ponto <= margem_ala

print(ala_ok)

plt.figure(figsize=(12, 6))
plt.plot(t_data, rul_real, marker='.', linestyle='-', color='blue', label='RUL Verdadeiro')
plt.plot(t_data_pred, rul_pred, marker='.', linestyle='--', color='red', label='RUL Previsto')

lim_sup = rul_real * (1 + alpha)
lim_inf = rul_real * (1 - alpha)
plt.fill_between(t_data, lim_inf, lim_sup, color='orange', alpha=0.3, label=f'Zona de Aceitação (α={alpha*100}%)')

plt.axvline(x=t_teste, color='b', linestyle=':', linewidth=2, label=f'Ciclo {t_teste}')
plt.plot(t_teste, rul_pred[idx_lambda], 'ro', markersize=10, label='Previsão')
plt.plot(t_teste, rul_real[idx_lambda], 'ko', markersize=10, label='Real')


plt.title('Métrica ALA (α-λ)')
plt.xlabel('Ciclos')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()