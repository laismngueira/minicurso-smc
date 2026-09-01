# -*- coding: utf-8 -*-
"""
Lançador do painel SCP-01 (Streamlit) dentro do Google Colab.

Uso:
  1) Envie para a sessão do Colab (aba de arquivos, /content/):
       - app_streamlit.py
       - os PDFs de conhecimento (ex.: 01_sistema_distribuicao.pdf,
         02_sistema_controle.pdf, 03_dados.pdf)
       - os arquivos da planta RNA: Modelo_AI_v1.h5, DadosTratados.xlsx
  2) Configure o segredo GROQ_API_KEY no Colab (ícone de chave 🔑 na
     barra lateral esquerda), com acesso liberado para este notebook.
  3) Cole os blocos abaixo em células separadas e rode em ordem.
"""

# ----------------------------------------------------------------------
# Célula 1 — instalação das dependências
# ----------------------------------------------------------------------
!pip install -q streamlit langchain langchain-groq langgraph \
    tensorflow scikit-learn pandas openpyxl \
    langchain_community faiss-cpu langchain-text-splitters pymupdf \
    sentence-transformers python-dotenv

!npm install -g localtunnel

# ----------------------------------------------------------------------
# Célula 2 — subir o Streamlit em background
# ----------------------------------------------------------------------
!streamlit run /content/app_streamlit.py --server.port 8501 &>/content/logs_streamlit.txt &

# ----------------------------------------------------------------------
# Célula 3 — senha do túnel (é o IP público desta máquina do Colab)
# ----------------------------------------------------------------------
!wget -q -O - https://loca.lt/mytunnelpassword

# ----------------------------------------------------------------------
# Célula 4 — abrir o túnel público
# (clique no link impresso; cole a senha da Célula 3 quando pedido)
# ----------------------------------------------------------------------
!npx localtunnel --port 8501

# ----------------------------------------------------------------------
# Solução de problemas
# ----------------------------------------------------------------------
# - Se a página não carregar: rode `!cat /content/logs_streamlit.txt`
#   para ver o erro real do Streamlit.
# - Se o painel mostrar "SISTEMA OFFLINE": o segredo GROQ_API_KEY não
#   está acessível — confira o ícone de chave 🔑 no Colab.
# - Se o LED de RAG aparecer em amarelo ("sem PDFs"): confirme que os
#   PDFs foram enviados para /content/ antes de rodar a Célula 2.
# - Se o LED de RNA aparecer vermelho ("arquivos ausentes"): confirme que
#   Modelo_AI_v1.h5 e DadosTratados.xlsx foram enviados para /content/.
# - Se o processo do Streamlit cair sem erro visível (segfault): geralmente
#   é conflito TensorFlow/PyTorch quando a ordem de import muda — não mova
#   o import de tensorflow para o topo do arquivo (fica local, dentro de
#   carregar_planta_rna(), de propósito).
