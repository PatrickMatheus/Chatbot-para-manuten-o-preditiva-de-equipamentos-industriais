# Chatbot para Manutenção Preditiva de Equipamentos Industriais

Este projeto faz parte de um **Programa Institucional de Bolsas de Iniciação Científica (PIBIC)** e tem como objetivo explorar o uso de **modelos de linguagem (LLMs)** aliados a técnicas de **PHM (Prognostics and Health Management)** para auxiliar na **manutenção preditiva** de equipamentos industriais.  

A proposta é desenvolver um chatbot capaz de interpretar dados coletados de equipamentos, sejam eles dados de degradação (dano acumulado) ou índice de saúde, realizar predições de falhas e comunicar os resultados de forma clara para operadores e engenheiros.  

---

## Motivação

A manutenção corretiva em equipamentos industriais gera **custos elevados** e **paradas inesperadas**.  
Já a manutenção preventiva pode resultar em gastos desnecessários.  

A **manutenção preditiva**, apoiada por **PHM** e **Inteligência Artificial**, surge como alternativa para:  

- Prever falhas antes que aconteçam.  
- Estimar a **vida útil remanescente (RUL)** dos equipamentos.  
- Apoiar decisões de manutenção com base em dados reais.  

O uso de **LLMs** e **MCP (Model Context Protocol)** permite criar interfaces conversacionais que tornam os resultados técnicos mais acessíveis e compreensíveis.  

---

## Objetivos

- **Objetivo geral**  
  Desenvolver um chatbot baseado em modelos de linguagem que auxilie na manutenção preditiva de equipamentos industriais.  

- **Objetivos específicos (fase atual)**  
  - Explorar técnicas de **PHM** para análise, prognóstico de falhas e estimativas de **RUL**.  
  - Avaliar o uso de **LLMs (GPT, Gemini, LLaMA)** juntamente com as técnicas de PHM para interpretar um conjunto de dados, predizer falhas e a **RUL**
  - Integrar o fluxo de predição a servidores **MCP (Model Context Protocol)** para comunicação entre módulos.  

---

## Tecnologias Utilizadas

- **Linguagem:** Python  
- **Modelos de Linguagem (LLMs):** GPT, Gemini, LLaMA  
- **Protocolos/Servidores:** MCP (*Model Context Protocol*)  
- **PHM (Prognostics and Health Management):** prognóstico de falhas e predição de vida útil  

---

## Status do Projeto

  **Em andamento (2025–2026):**  
- Estudo e experimentação com algoritmos de **PHM**.  
- Predições de falhas e de **RUL** em séries temporais.  
- Integração inicial com **MCP** para comunicação entre módulos.  
---

# Como testar os arquivos
Para fazer o teste da fase atual do projeto siga os passos a seguir:
  1. Clone o repositório para o seu computador:
   ```bash
   git clone https://github.com/PatrickMatheus/Chatbot-para-manuten-o-preditiva-de-equipamentos-industriais.git
   ```
  2. Navegue até o diretório do projeto.
  3. Crie um ambiente virtual de desenvolvimento (evita erros)
   ```bash
   python -m venv .venv
  ```
  4. Instale as dependências necessárias
     ``` bash
     pip install -r .\requirements.txt
     ```
  5. pesquise e instale o node.js
     ``` bash
     https://nodejs.org/pt/download
     ```
  6. Execute os exemplos que quiser.
  
