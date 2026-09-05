# Minicurso SMC — Assistente de Controle PID com IA

Minicurso prático de engenharia de controle que constrói, passo a passo, um
assistente de IA para uma planta industrial real (sistema de distribuição de
água), combinando LLM (Groq), controle PID clássico, RAG sobre documentos
técnicos e um agente autônomo (LangGraph) com painel interativo.

## O que o assistente faz

- **Bloco I — Fundamentos da LLM**: valida a conexão com a Groq.
- **Bloco II — Sistema de Controle**: simula a planta (uma RNA treinada com
  dados reais de campo) controlada por um PID em malha fechada, com
  proteção contra overshoot e otimização automática dos ganhos (grid
  search).
- **Bloco III — RAG**: responde perguntas de teoria com base em PDFs
  técnicos (controle PID, métricas de desempenho, discretização, etc.),
  citando a fonte.
- **Bloco IV — Agente de IA (LangGraph)**: recebe uma pergunta em linguagem
  natural e decide sozinho se deve explicar teoria, simular ou otimizar o
  controlador.
- **Bloco V — Interface**: painel industrial em Gradio, publicado com link
  público próprio (`share=True`), sem necessidade de túnel externo.

## Como rodar (Google Colab)

1. Abra o [Google Colab](https://colab.research.google.com/) e faça upload
   de [`versao_colab/Minicurso_SCP01.ipynb`](versao_colab/Minicurso_SCP01.ipynb)
   (ou `Arquivo > Abrir notebook > GitHub`, colando a URL deste repositório).
2. Configure os segredos do notebook (ícone de chave 🔑 na barra lateral
   esquerda):
   - `GROQ_API_KEY` — **obrigatório**. Crie uma chave gratuita em
     [console.groq.com](https://console.groq.com/keys).
   - `GITHUB_TOKEN` — opcional. Com um Personal Access Token do GitHub
     (escopo `repo` se o repositório estiver privado; não é necessário
     enquanto ele for público), o próprio notebook clona este repositório e
     copia os arquivos de apoio para `/content/` automaticamente.
3. Sem `GITHUB_TOKEN`, baixe manualmente os arquivos de apoio (veja abaixo)
   e envie pela aba de arquivos do Colab (`/content/`) antes de rodar as
   células do Bloco II.
4. Rode as células em ordem, de cima para baixo. O Bloco V, ao final, abre o
   painel e gera um link público temporário.

## Arquivos de apoio necessários

Todos disponíveis em [`versao_colab/`](versao_colab/):

| Arquivo | Descrição |
|---|---|
| `Modelo_AI_v1.h5` | Rede Neural Artificial já treinada (a "planta") |
| `DadosTratados.xlsx` | Dados reais de campo, usados como contexto fixo |
| `01_sistema_distribuicao.pdf` | Base de conhecimento do RAG |
| `02_sistema_controle.pdf` | Base de conhecimento do RAG |
| `03_dados.pdf` | Base de conhecimento do RAG |

## Estrutura do repositório

- `versao_colab/Minicurso_SCP01.ipynb` — notebook do minicurso, pronto para
  importar no Colab.
- `versao_colab/minicurso_scp01_colab.py` — mesmo conteúdo do notebook, em
  formato `.py` (célula por célula, no padrão de exportação do Colab).
- `versao_colab/` — arquivos de apoio (RNA, dataset, PDFs) usados pelo
  notebook.

## Pré-requisitos

- Uma chave de API da [Groq](https://console.groq.com/keys) (gratuita).
- Conta Google para rodar o notebook no Colab (não é necessário instalar
  nada localmente).
