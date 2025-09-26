import gradio as gr
import plotly.graph_objects as go
import asyncio
import json
from contextlib import suppress

# Importa a sua classe utilitária
from utils.mcp_util import mcpUtil

mcp_util = mcpUtil()
AVAILABLE_TOOLS = []
SERVER_HOST = "http://127.0.0.1:8000/sse"

async def initialize_client():
    """Função assíncrona para inicializar a conexão com o servidor."""
    global AVAILABLE_TOOLS
    try:
        print(f"Conectando ao servidor em {SERVER_HOST}...")
        await mcp_util.initialize_with_sse(SERVER_HOST)
        print("Conexão estabelecida com sucesso.")
        
        # Busca dinamicamente as ferramentas disponíveis no servidor
        tools_list = await mcp_util.get_tools()
        AVAILABLE_TOOLS = [tool.name for tool in tools_list]
        print(f"Ferramentas encontradas: {AVAILABLE_TOOLS}")
        return True
    except Exception as e:
        print(f"Erro ao conectar ou inicializar o cliente MCP: {e}")
        print("Verifique se o servidor 'server_mcp.py' está em execução.")
        return False

def plotar_serie_e_previsao(historico: list, previsao: list = None, titulo: str = "Série Temporal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=historico, mode='lines', name='Dados Históricos', line=dict(color='blue')))
    if previsao:
        indices_previsao = list(range(len(historico), len(historico) + len(previsao)))
        linha_conexao_y = [historico[-1]] + previsao
        indices_conexao = [len(historico) - 1] + indices_previsao
        fig.add_trace(go.Scatter(x=indices_conexao, y=linha_conexao_y, mode='lines+markers', name='Previsão', line=dict(color='red', dash='dot'), marker=dict(size=8)))
    fig.update_layout(title=titulo, xaxis_title="Passos no Tempo", yaxis_title="Valor (PC1)", legend_title="Legenda", template="plotly_white")
    return fig

async def carregar_dados_do_servidor(pontos: int):
    if not mcp_util.session:
        raise gr.Error("Cliente não conectado ao servidor.")
    
    try:
        uri = f"serie://{int(pontos)}"
        print(f"Buscando recurso: {uri}")
        result_object = await mcp_util.get_resource(uri)
        data_as_string = result_object.contents[0].text
        serie_historica = json.loads(data_as_string)
        
        plot_inicial = plotar_serie_e_previsao(serie_historica, titulo=f"Série Histórica - {pontos} pontos")
        return serie_historica, plot_inicial
    except Exception as e:
        raise gr.Error(f"Falha ao carregar dados do servidor: {e}")

async def executar_previsao(dados_historicos: list, metodo_selecionado: str, passos_futuros: int):
    if not mcp_util.session:
        raise gr.Error("Cliente não conectado ao servidor.")
    if not dados_historicos:
        raise gr.Error("Primeiro, carregue os dados históricos.")
    if not metodo_selecionado:
        raise gr.Error("Selecione um método de previsão.")

    try:
        args = {
            "dados": dados_historicos,
            "passos_futuros": int(passos_futuros)
        }
        
        print(f"Chamando a ferramenta '{metodo_selecionado}' com {len(dados_historicos)} pontos...")
        

        result_object = await mcp_util.call_tool(metodo_selecionado, args=args)
        
        previsoes = result_object.structuredContent['result']
        # ---------------------
        
        print(f"Previsão recebida: {previsoes}")
        plot_resultado = plotar_serie_e_previsao(dados_historicos, previsoes, f"Resultado com '{metodo_selecionado}'")
        saida_texto = ", ".join(map(str, previsoes))
        
        return plot_resultado, saida_texto
    except Exception as e:
        raise gr.Error(f"Falha na execução da previsão: {e}")

with gr.Blocks(theme=gr.themes.Soft(), title="Cliente de Previsão de Séries Temporais") as demo:
    gr.Markdown("# 📈 Cliente para Servidor de Previsão")
    gr.Markdown("Interface interagindo com o servidor através da classe `mcpUtil`.")
    dados_historicos_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configurações")
            pontos_input = gr.Slider(minimum=20, maximum=200, value=50, step=1, label="Pontos Históricos")
            load_button = gr.Button("Carregar Dados", variant="secondary")
            metodo_dropdown = gr.Dropdown(choices=AVAILABLE_TOOLS, label="Método de Previsão", info="Selecione a técnica.")
            passos_futuros_input = gr.Number(value=5, label="Passos a Prever", minimum=1, step=1)
            predict_button = gr.Button("Realizar Previsão", variant="primary")
        with gr.Column(scale=3):
            gr.Markdown("### 2. Resultados")
            output_plot = gr.Plot(label="Visualização da Série Temporal")
            output_text = gr.Textbox(label="Valores Previstos")
    load_button.click(fn=carregar_dados_do_servidor, inputs=[pontos_input], outputs=[dados_historicos_state, output_plot])
    predict_button.click(fn=executar_previsao, inputs=[dados_historicos_state, metodo_dropdown, passos_futuros_input], outputs=[output_plot, output_text])
    demo.load(lambda: gr.Dropdown(choices=AVAILABLE_TOOLS), inputs=None, outputs=metodo_dropdown)

# --- 6. Iniciar a Aplicação (sem alterações) ---
async def main():
    is_connected = await initialize_client()
    if is_connected:
        print("Cliente Gradio pronto para ser lançado.")
        try:
            app, local_url, share_url = demo.launch(prevent_thread_lock=True)
            print(f"Gradio rodando em: {local_url}")
            while True:
                await asyncio.sleep(0.1)
        finally:
            print("\nFechando conexão com o servidor...")
            await mcp_util.clearup()
            print("Conexão fechada.")
    else:
        print("Aplicação Gradio não iniciada devido a falha de conexão.")

if __name__ == "__main__":
    with suppress(asyncio.CancelledError):
        asyncio.run(main())