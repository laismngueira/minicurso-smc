# -*- coding: utf-8 -*-
"""
Lançador do painel SCP-01 (Streamlit) dentro do Google Colab.

Uso:
  1) Envie para a sessão do Colab (aba de arquivos, /content/):
       - app_streamlit.py
       - scp01_core.py (planta, PID, RAG e agente — app_streamlit.py importa
         daqui em vez de reimplementar tudo)
       - os PDFs de conhecimento (ex.: 01_sistema_distribuicao.pdf,
         02_sistema_controle.pdf, 03_dados.pdf)
       - os arquivos da planta RNA: Modelo_AI_v1.h5, DadosTratados.xlsx
  2) Configure o segredo GROQ_API_KEY no Colab (ícone de chave 🔑 na
     barra lateral esquerda), com acesso liberado para este notebook.
  3) Cole os blocos abaixo em células separadas e rode em ordem.

Publicamos com o proxy de porta nativo do Colab (`google.colab.output`),
não com um túnel de terceiros (localtunnel/ngrok): não precisa de
`npm install`, não pede senha, e evita as falhas de carregamento dos
módulos JS do Streamlit que o localtunnel costuma apresentar.
"""

# ----------------------------------------------------------------------
# Célula 1 — instalação das dependências
# ----------------------------------------------------------------------
!pip install -q streamlit langchain langchain-groq langgraph \
    tensorflow scikit-learn pandas openpyxl \
    langchain_community faiss-cpu langchain-text-splitters pymupdf \
    sentence-transformers python-dotenv

# ----------------------------------------------------------------------
# Célula 2 — repassar a chave para o processo do Streamlit
# (userdata.get() só funciona dentro do kernel do notebook; como o
# Streamlit sobe como um processo separado, a chave precisa virar
# variável de ambiente ANTES de subir o processo, senão o painel aparece
# como "SISTEMA OFFLINE" mesmo com o segredo configurado)
# ----------------------------------------------------------------------
import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

# ----------------------------------------------------------------------
# Célula 3 — subir o Streamlit em background
# ----------------------------------------------------------------------
!streamlit run /content/app_streamlit.py --server.port 8501 &>/content/logs_streamlit.txt &

# ----------------------------------------------------------------------
# Célula 4 — abrir o painel via proxy do Colab
# (dá alguns segundos para o Streamlit subir antes de gerar o link)
# ----------------------------------------------------------------------
import time

time.sleep(8)

from google.colab.output import eval_js

print(eval_js("google.colab.kernel.proxyPort(8501)"))

# ----------------------------------------------------------------------
# Solução de problemas
# ----------------------------------------------------------------------
# - Se a página não carregar: rode `!cat /content/logs_streamlit.txt`
#   para ver o erro real do Streamlit, ou aumente o time.sleep da Célula 4
#   e rode de novo (o Streamlit pode levar mais que 8s para subir).
# - Se o painel mostrar "SISTEMA OFFLINE": confirme que a Célula 2 rodou
#   sem erro — se o secret GROQ_API_KEY não estiver liberado para este
#   notebook, userdata.get() lança exceção antes mesmo de chegar no
#   Streamlit.
# - Se o LED de RAG aparecer em amarelo ("sem PDFs"): confirme que os
#   PDFs foram enviados para /content/ antes de rodar a Célula 3.
# - Se o LED de RNA aparecer vermelho ("arquivos ausentes"): confirme que
#   Modelo_AI_v1.h5 e DadosTratados.xlsx foram enviados para /content/.
# - Se o processo do Streamlit cair sem erro visível (segfault): geralmente
#   é conflito TensorFlow/PyTorch quando a ordem de import muda — não mova
#   o import de tensorflow para o topo do arquivo (fica local, dentro de
#   carregar_planta_rna(), em scp01_core.py, de propósito).
