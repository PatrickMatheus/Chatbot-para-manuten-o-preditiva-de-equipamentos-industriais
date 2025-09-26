import os
import asyncio
import dotenv
import pandas as pd
from openai import OpenAI
from fastmcp import FastMCP
from typing import List

server_ts = FastMCP("TimeSeries_MCP")

def carregar_api_key():
    dotenv.load_dotenv()
    return os.environ["API_KEY"]

api_key = carregar_api_key()

client = OpenAI(api_key=api_key)

def carregar_arquivo(arquivo: str = "arquivos/Bearing1_2.csv") -> list[float]:
    """Função auxiliar para ler os arquivos e retornar a série."""
    try:
        df = pd.read_csv(arquivo)
        serie = (1-df["PC1"]).values.tolist()
        return serie
    except FileNotFoundError:
        raise ValueError(f"Arquivo não encontrado em: {arquivo}")
    except KeyError:
        raise ValueError("A coluna 'PC1' não foi encontrada no arquivo CSV.")

def _gerar_previsao_com_llm(serie_historica: list[float], passos_a_prever: int) -> list[float]:
    """
    Função central que formata o prompt, chama a API do LLM e trata a resposta.
    """
    historico_str = ", ".join(map(str, serie_historica))
    
    prompt_sistema = (
        "Você é um especialista em previsão de séries temporais. "
        "Sua tarefa é analisar a sequência de dados fornecida e prever os próximos valores. "
        "Responda APENAS com os valores numéricos previstos, separados por vírgula. "
        "Não inclua texto explicativo, apenas os números."
    )
    
    prompt_usuario = (
        f"Dada a seguinte série temporal: {historico_str}. "
        f"Preveja os próximos {passos_a_prever} valores da sequência."
    )

    try:
        resposta =  client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.2,
        )
        
        resultado_texto = resposta.choices[0].message.content
        previsoes_str = resultado_texto.strip().split(',')
        previsoes_float = [float(p.strip()) for p in previsoes_str]
        
        return previsoes_float

    except Exception as e:
        raise ValueError(f"Falha na comunicação com o LLM: {e}")

@server_ts.resource("serie://{pontos}")
def get_slice_serie(pontos: int) -> list[float]:
    """Retorna uma fatia da série temporal com a quantidade de pontos especificada."""
    serie_completa = carregar_arquivo()
    if pontos <= 0:
        return []
    return serie_completa[:pontos]

@server_ts.tool
async def prever_oneShot(dados: list[float], passos_futuros: int) -> list[float]:
    """
    Ao receber um conjunto de dados, retorna a previsão dos próximos 'passos_futuros' valores usando um LLM.
    """
    if not dados:
        raise ValueError("A lista de dados históricos não pode estar vazia.")
    
    previsao =  _gerar_previsao_com_llm(serie_historica=dados, passos_a_prever=passos_futuros)
    return previsao

@server_ts.tool
def prever_zeroShot(dados: List[float], passos_futuros: int) -> List[float]:
    """
    Tenta prever o próximo ponto com o mínimo de instruções (abordagem 'zero-shot').
    Testa a capacidade do LLM de continuar uma sequência numérica de forma crua.
    """
    if not dados:
        raise ValueError("A lista de dados históricos não pode estar vazia.")
    
    
    historico_str = ", ".join(map(str, dados))
    prompt_sistema = (
        "Você é um assistente de analise de dados preciso. Sua função é prever os próximos números de uma série temporal com base nos dados fornecidos."
        "Responda APENAS com os valores numéricos previstos, separados por vírgula. "
        "Não inclua texto explicativo, apenas os números.")
    prompt_usuario = f" Dada a série temporal {historico_str}, preveja os proximos {passos_futuros} valores futuros dessa sequência"

    try:
        resposta =  client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.0, 
        )
        
        
        resultado_texto = resposta.choices[0].message.content
        previsoes_str = resultado_texto.strip().split(',')
        previsoes_float = [float(p.strip()) for p in previsoes_str]
        
        return previsoes_float

    except Exception as e:
        raise ValueError(f"Falha na previsão zero-shot: {e}")

def _criar_exemplos_internos(dados: List[float], tamanho_janela: int, num_exemplos: int) -> str:
    """Função auxiliar para gerar exemplos de dentro da própria série."""
    if len(dados) < tamanho_janela + 1:
        return "" 
    
    exemplos_formatados = ""
    inicio = max(0, len(dados) - num_exemplos - tamanho_janela - 1)
    
    for i in range(inicio, len(dados) - tamanho_janela):
        entrada = dados[i : i + tamanho_janela]
        saida = dados[i + tamanho_janela]
        entrada_str = ", ".join(f"{x:.4f}" for x in entrada)
        saida_str = f"{saida:.4f}"
        exemplos_formatados += f"Entrada: [{entrada_str}] -> Saída: [{saida_str}]\n"
        
    return exemplos_formatados

@server_ts.tool
def prever_fewShot(dados: List[float], passos_futuros: int) -> List[float]:
    """
    Prevê usando a técnica de few-shot, criando os exemplos a partir da própria
    série de dados de entrada para ensinar o padrão interno ao LLM.
    """
    tamanho_janela = 10
    num_exemplos = 3

    if len(dados) < tamanho_janela + num_exemplos + 1:
        raise ValueError("Não há dados suficientes para criar exemplos e fazer uma previsão few-shot.")

    prompt_sistema = (
        "Você é um assistente de analise de dados preciso. Sua função é prever os próximos números de uma série temporal com base nos dados e exemplos fornecidos."
        "Responda APENAS com os valores numéricos previstos, separados por vírgula. "
        "Não inclua texto explicativo, apenas os números.")
    
    exemplos_formatados = _criar_exemplos_internos(dados[:-tamanho_janela], tamanho_janela, num_exemplos)
    
    dados_finais_para_prever = dados[-tamanho_janela:]
    entrada_final_str = ", ".join(f"{x:.4f}" for x in dados_finais_para_prever)

    prompt_usuario = (
        f"Aprenda com os seguintes exemplos extraídos da própria série:\n{exemplos_formatados}"
        f"Agora, preveja o(s) próximo(s) {passos_futuros} valor(es) para a entrada final:\n"
        f"Entrada: [{entrada_final_str}] -> Saída:"
    )
    
    try:
        resposta =  client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}],
            temperature=0.0,
        )
        resultado_texto = resposta.choices[0].message.content
        previsoes = [float(p.strip()) for p in resultado_texto.strip().split(',') if p.strip()]
        return previsoes[:passos_futuros]
    except Exception as e:
        raise ValueError(f"Falha na previsão few-shot: {e}")

@server_ts.tool
async def prever_reprogramming(dados: List[float], passos_futuros: int) -> List[float]:
    """
    'Reprograma' o LLM para atuar em uma atividade que ele tem domínio, como programação
    """
    historico_str = ", ".join(f"{x:.4f}" for x in dados)

    prompt_usuario = (
        "#O script python a seguir analisa uma lista de dados de degradação (dano acumulado) de rolamentos."
        f"#E usa uma função de machine learning para prever os próximos {passos_futuros} valores futuros da sequência."
        "#Complete os valores que seriam retornados pela função."
        "#Responda APENAS com os valores numéricos previsto."
        "#Não inclua texto explicativo, apenas os valores."
        "#Script python: "
        f"sensor_data = [{historico_str}]"
        "def predict_next_degradation_value(data):"
            "#...modelo machine learning complexo..."
            "return #valores previstos"
        "predicted_value = predict_next_degradation_value(sensor_data)"
    )

    try:
        resposta =  client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", "content": prompt_usuario}],
            temperature=0.0,
        )
        resultado_texto = resposta.choices[0].message.content
        previsoes = [float(p.strip()) for p in resultado_texto.strip().split(',') if p.strip()]
        return previsoes[:passos_futuros]
    except Exception as e:
        raise ValueError(f"Falha na previsão com reprogramming: {e}")

@server_ts.tool
def prever_promptLearning(dados: List[float], passos_futuros: int) -> List[float]:
    """Preve usando a técnica de prompt learning. Dados ao LLM instruções específicas de como pensar"""
    historico_str = ", ".join(f"{x:.4f}" for x in dados)
    prompt_usuario = (
        f"Analise os seguintes dados e preveja od próximos {passos_futuros} valores futuros da sequência." 
        "Responda apenas com os valores previstos."
        "Não inclua texto explicativo, apenas os valores."
        f"Dados: [{historico_str}]"
        "Próximos valores: "
    )
    prompt_sistema = "Você é um engenheiro de manutenção preditiva especializado em análise de vibração. Sua tarefa é analisar séries temporais que representam a degradação (dano acumulado) de rolamentos industriais e prever o próximo valor com a maior precisão possível."
    try:
        resposta = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt_sistema},
             {"role": "user", "content": prompt_usuario}],
            temperature=0.2,
        )
        resultado_texto = resposta.choices[0].message.content
        return [float(p.strip()) for p in resultado_texto.strip().split(',')]
    except Exception as e:
        raise ValueError(f"Falha na previsão com prompt: {e}")


if __name__ == "__main__":
    print("Servidor TimeSeries MCP rodando...")
    server_ts.run(transport="sse")