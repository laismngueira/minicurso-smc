# 🧠 Minicurso CBA 2026: Imersão em Modelos LLMs — Aplicação a Sistemas de Controle

<p align="center">
  <b>Eng. Laís Araújo Mangueira</b><br/>
  <i>Engenheira Eletricista</i>
</p>

<p align="center">
  <b>Prof. Dr. Juan Moises Maurício Villanueva</b><br/>
  <i>Professor da UFPB</i>
</p>

📅 **Data:** 06 de outubro de 2026, das 13h às 17h

📍 **Evento:** Congresso Brasileiro de Automática (CBA 2026)

🏛️ **Apoio institucional:** UFPB, PPGEE/UFPB, LAII, FAPESQ-PB, IEEE SMC

## 📌 Visão geral

Minicurso prático de engenharia de controle que constrói, passo a passo, um
assistente de IA para uma planta industrial real (sistema de distribuição de
água), combinando LLM (Groq), controle PID clássico, RAG sobre documentos
técnicos e um agente autônomo (LangGraph) com painel interativo.

## 🎯 Objetivos

### Objetivo geral

Capacitar os participantes a construir, de ponta a ponta, um assistente de
IA que integra modelos de linguagem (LLMs) a um sistema de controle
industrial real.

### Objetivos específicos

- Compreender os fundamentos de uma LLM e como consumi-la via API (Groq).
- Simular uma planta de controle (RNA) e ajustar um controlador PID em
  malha fechada.
- Aplicar RAG (Retrieval-Augmented Generation) para responder perguntas de
  teoria com base em documentos técnicos.
- Implementar um agente autônomo (LangGraph) com engenharia de prompt, capaz
  de decidir entre explicar, simular ou otimizar o controlador.
- Publicar uma interface interativa (Gradio) para operar o assistente.

## 🧑‍💻 Público-alvo e pré-requisitos

### Público-alvo

Estudantes, professores, pesquisadores e profissionais de Engenharia,
Computação ou áreas afins com interesse em IA aplicada a sistemas de
controle.

### Pré-requisitos

- Conhecimento básico de **Python**.
- Uma chave de API da [Groq](https://console.groq.com/keys) (gratuita).
- Conta Google para rodar o notebook no Colab (não é necessário instalar
  nada localmente).

## 🗂️ Plano de aula

### ⏱️ Duração: 4 horas (13h–17h)

1. **Bloco I — Fundamentos da LLM** (60 min)
   - Primeiro contato com o modelo de linguagem
   - Validação da conexão com a Groq
2. **Bloco II — Sistema de Controle** (45 min)
   - A planta: uma RNA treinada com dados reais de campo
   - O controlador PID em malha fechada, com proteção contra overshoot
   - Otimização automática dos ganhos (grid search)

**COFFEE-BREAK (14:45 -15:15)**

3. **Bloco III — RAG** (45 min)
   - Mecanismos de consulta e fluxograma de recuperação da informação
   - Respostas de teoria com base em PDFs técnicos, citando a fonte
4. **Bloco IV — Agente de IA (LangGraph)** (45 min)
   - Engenharia de prompt e construção dos prompts de instrução
   - O agente decide sozinho entre explicar teoria, simular ou otimizar

## ▶️ Como rodar (Google Colab)

### Opção 1 — link direto (mais rápido)

Clique para abrir o notebook direto no Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laismngueira/minicurso-smc/blob/main/versao_colab/Minicurso_SCP01.ipynb)

### Opção 2 — clonando o repositório

```bash
git clone https://github.com/laismngueira/minicurso-smc.git
```

Depois, no Colab, vá em **Arquivo > Fazer upload de notebook** e selecione
o arquivo `versao_colab/Minicurso_SCP01.ipynb` da pasta clonada (o repo é
público, então o clone não pede usuário nem senha).

### Rodando o notebook

1. Configure o segredo `GROQ_API_KEY` (ícone de chave 🔑 na barra lateral
   esquerda do Colab) — crie uma chave gratuita em
   [console.groq.com](https://console.groq.com/keys). É o único segredo
   necessário.
2. Rode as células em ordem, de cima para baixo. O Bloco II baixa sozinho
   os arquivos de apoio (RNA, dataset, PDFs) direto deste repositório —
   nenhuma configuração extra é necessária.
3. O Bloco V, ao final, abre o painel e gera um link público temporário.

## 📁 Arquivos de apoio necessários

Todos disponíveis em [`versao_colab/`](versao_colab/):

| Arquivo | Descrição |
|---|---|
| `Modelo_AI_v1.h5` | Rede Neural Artificial já treinada (a "planta") |
| `DadosTratados.xlsx` | Dados reais de campo, usados como contexto fixo |
| `01_sistema_distribuicao.pdf` | Base de conhecimento do RAG |
| `02_sistema_controle.pdf` | Base de conhecimento do RAG |
| `03_dados.pdf` | Base de conhecimento do RAG |

## 📂 Estrutura do repositório

```plaintext
📦 minicurso-smc
├── 📂 versao_colab
│   ├── Minicurso_SCP01.ipynb        # notebook do minicurso (Colab)
│   ├── minicurso_scp01_colab.py     # mesmo conteúdo, exportado em .py
│   ├── Modelo_AI_v1.h5              # RNA já treinada (a "planta")
│   ├── DadosTratados.xlsx           # dados reais de campo
│   ├── 01_sistema_distribuicao.pdf  # base de conhecimento do RAG
│   ├── 02_sistema_controle.pdf      # base de conhecimento do RAG
│   └── 03_dados.pdf                 # base de conhecimento do RAG
├── 📜 .gitignore
└── 📜 README.md
```

## 🎞️ Slides

Apresentação do minicurso (CBA 2026): [baixar PDF](https://github.com/laismngueira/minicurso-smc/releases/download/slides-cba-sp/V4.Minicurso.CBA.SP.pdf).
