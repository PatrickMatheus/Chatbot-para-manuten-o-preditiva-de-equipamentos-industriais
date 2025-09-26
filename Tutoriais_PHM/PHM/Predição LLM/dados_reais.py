import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('arquivos/Bearing1_5.csv')
    sinal_degradacao_real = df['PC1'].values
    y_data = sinal_degradacao_real[0] - sinal_degradacao_real
    print("Arquivo CSV carregado com sucesso!")
    print(f"Total de pontos de dados: {len(sinal_degradacao_real)}")
    print("Primeiros 5 pontos:", y_data[:5])

    plt.figure(figsize=(10, 6))
    plt.plot(y_data, label='Índice de Saúde Real (PC1)')
    plt.title('Sinal de Degradação do Rolamento')
    plt.xlabel('Ciclos (amostras de tempo)')
    plt.ylabel('Índice de Saúde')
    plt.grid(True)
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Erro: Arquivo Bearing1_2.csv não encontrado. Certifique-se de que ele está na mesma pasta que o script.")