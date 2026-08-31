# -*- coding: utf-8 -*-
"""
SCP-01 — Painel Industrial de Controle PID
============================================

Interface Streamlit para o assistente de engenharia de controle do minicurso.
Mantém a planta e o agente originais (LLM de triagem + RAG sobre PDFs +
LangGraph), apenas trocando a interface Gradio por um painel de estilo
SCADA/HMI.

Planta: função de transferência de 2ª ordem G(s) = 0.4 / (s² + 10s + 25),
discretizada por ZOH — não depende de RNA nem de arquivos de dados externos.

Como rodar (Colab, com túnel):
  1) Envie este arquivo e os PDFs de conhecimento para a sessão do Colab
     (aba de arquivos, /content/).
  2) Rode o notebook de lançamento (colab_launch_streamlit.py) célula a
     célula, ou manualmente:
       !pip install -q streamlit langchain langchain-groq langgraph control \
           langchain_community faiss-cpu langchain-text-splitters pymupdf \
           sentence-transformers
       !streamlit run /content/app_streamlit.py &>/content/logs.txt &
       !npx localtunnel --port 8501

Como rodar localmente:
  export GROQ_API_KEY=...
  streamlit run app_streamlit.py
"""

import base64
import io
import os
import pathlib
import re
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuração da página e paleta industrial
# ============================================================================

st.set_page_config(
    page_title="SCP-01 · Controle PID",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONTENT_DIR = pathlib.Path(os.environ.get("MINICURSO_CONTENT_DIR", "/content"))

# Paleta validada (dataviz skill): superfícies e tinta em modo escuro,
# duas séries categóricas (slot 1 azul / slot 2 laranja) e paleta de status
# fixa (nunca reaproveitada para séries).
COR = {
    "pagina": "#0d0d0d",
    "painel": "#1a1a19",
    "painel_alt": "#212120",
    "borda": "#383835",
    "grade": "#2c2c2a",
    "texto": "#ffffff",
    "texto_sec": "#c3c2b7",
    "texto_mudo": "#898781",
    "serie_medida": "#3987e5",   # categórico slot 1 (azul) — variável controlada
    "serie_setpoint": "#d95926",  # categórico slot 2 (laranja) — setpoint
    "bom": "#0ca30c",
    "alerta": "#fab219",
    "serio": "#ec835a",
    "critico": "#d03b3b",
}

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {{ color-scheme: dark; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {COR["pagina"]} !important;
    color: {COR["texto"]};
    font-family: 'Inter', sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: {COR["painel"]};
    border-right: 1px solid {COR["borda"]};
}}

.scp-banner {{
    background: linear-gradient(180deg, {COR["painel_alt"]} 0%, {COR["painel"]} 100%);
    border: 1px solid {COR["borda"]};
    border-left: 4px solid {COR["serie_medida"]};
    border-radius: 6px;
    padding: 14px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.scp-banner h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    letter-spacing: 0.04em;
    margin: 0;
    color: {COR["texto"]};
}}
.scp-banner p {{
    margin: 2px 0 0 0;
    color: {COR["texto_mudo"]};
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}

.led {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {COR["texto_sec"]};
    padding: 4px 10px;
    border: 1px solid {COR["borda"]};
    border-radius: 999px;
    background: {COR["painel_alt"]};
}}
.led .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    box-shadow: 0 0 6px currentColor;
}}
.led.on .dot {{ background: {COR["bom"]}; color: {COR["bom"]}; }}
.led.off .dot {{ background: {COR["critico"]}; color: {COR["critico"]}; }}
.led.warn .dot {{ background: {COR["alerta"]}; color: {COR["alerta"]}; }}

.scp-panel {{
    background: {COR["painel"]};
    border: 1px solid {COR["borda"]};
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 14px;
}}

.scp-section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COR["texto_sec"]};
    border-bottom: 1px solid {COR["borda"]};
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
}}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {{
    color: {COR["texto"]} !important;
}}

.scp-console {{
    background: #0a0a0a;
    border: 1px solid {COR["borda"]};
    border-radius: 6px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    color: {COR["texto_sec"]};
    white-space: pre-wrap;
    max-height: 420px;
    overflow-y: auto;
}}

.scp-tile {{
    background: {COR["painel"]};
    border: 1px solid {COR["borda"]};
    border-radius: 6px;
    padding: 12px 14px;
    text-align: left;
}}
.scp-tile .label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COR["texto_mudo"]};
    margin-bottom: 4px;
}}
.scp-tile .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: {COR["texto"]};
}}
.scp-tile .unit {{
    font-size: 0.85rem;
    color: {COR["texto_mudo"]};
    margin-left: 4px;
}}
.scp-tile .badge {{
    display: inline-block;
    margin-top: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 999px;
}}

.scp-cite {{
    border-left: 3px solid {COR["serie_medida"]};
    background: {COR["painel_alt"]};
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
}}
.scp-cite .fonte {{
    font-family: 'JetBrains Mono', monospace;
    color: {COR["texto_mudo"]};
    font-size: 0.75rem;
    text-transform: uppercase;
}}

[data-testid="stChatInputTextArea"] {{
    color: #0b0b0b !important;
    caret-color: #0b0b0b;
}}
[data-testid="stChatInputTextArea"]::placeholder {{
    color: #52514e !important;
}}

.stButton > button {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 4px;
    border: 1px solid {COR["borda"]};
}}
.stButton > button[kind="primary"] {{
    background: {COR["serie_medida"]};
    border-color: {COR["serie_medida"]};
}}

.scp-footer {{
    color: {COR["texto_mudo"]};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-align: right;
    margin-top: 24px;
    border-top: 1px solid {COR["borda"]};
    padding-top: 8px;
}}
</style>
"""
st.html(CSS)


def led(texto: str, estado: str) -> str:
    """estado: 'on' | 'off' | 'warn'"""
    return f'<span class="led {estado}"><span class="dot"></span>{texto}</span>'


def metric_tile(label: str, value, unit: str = "", status: Optional[str] = None, status_label: str = "") -> str:
    cor_badge = {
        "bom": COR["bom"],
        "alerta": COR["alerta"],
        "critico": COR["critico"],
    }.get(status)
    badge_html = ""
    if status and cor_badge:
        badge_html = (
            f'<div class="badge" style="background:{cor_badge}22;color:{cor_badge};'
            f'border:1px solid {cor_badge}66;">{status_label}</div>'
        )
    return f"""
    <div class="scp-tile">
        <div class="label">{label}</div>
        <div><span class="value">{value}</span><span class="unit">{unit}</span></div>
        {badge_html}
    </div>
    """


# ============================================================================
# Credenciais (Colab ou ambiente local)
# ============================================================================

def _get_groq_key() -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
        chave = userdata.get("GROQ_API_KEY")
        if chave:
            return chave
    except Exception:
        pass
    if hasattr(st, "secrets"):
        try:
            chave = st.secrets.get("GROQ_API_KEY")
            if chave:
                return chave
        except Exception:
            pass
    return os.environ.get("GROQ_API_KEY")


GROQ_API_KEY = _get_groq_key()


@st.cache_resource(show_spinner=False)
def get_llm() -> Optional[ChatGroq]:
    if not GROQ_API_KEY:
        return None
    # "llama-3.3-70b-versatile" (usado no notebook original) foi descontinuado
    # pela Groq e não está mais disponível no catálogo desta chave — testado
    # em 31/08/2026, ver `curl https://api.groq.com/openai/v1/models`.
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, api_key=GROQ_API_KEY)


# ============================================================================
# Bloco II — Planta (função de transferência discretizada por ZOH) + PID
# ============================================================================

PLANTA = ct.tf([0.4], [1, 10, 25])


def simular_planta(
    Kp: float = 10.0,
    Ti: float = 0.1,
    Td: float = 0.0,
    T_amostragem: float = 0.1,
    N_amostras: int = 150,
    plot: bool = True,
) -> Dict:
    sistema_discreto = PLANTA.sample(T_amostragem, method="zoh")
    NumD = sistema_discreto.num[0][0]
    DenD = sistema_discreto.den[0][0]

    yr = np.zeros(N_amostras)
    y_med = np.zeros(N_amostras)
    yr[0:50] = 1.0
    yr[50:100] = 2.0
    yr[100:150] = 0.75

    y = np.zeros(N_amostras)
    u = np.zeros(N_amostras)
    erro = np.zeros(N_amostras)
    s_int = np.zeros(N_amostras)

    kp = Kp
    ki = Kp * T_amostragem / Ti if Ti > 1e-3 else 0
    kd = Kp * Td / T_amostragem

    for k in range(2, N_amostras):
        y[k] = (
            -DenD[1] * y[k - 1]
            - DenD[2] * y[k - 2]
            + NumD[0] * u[k - 1]
            + NumD[1] * u[k - 2]
        )
        y_med[k] = y[k]  # ruído desligado (ver versão original para reativar)
        erro[k] = yr[k] - y_med[k]
        s_int[k] = s_int[k - 1] + erro[k]
        u[k] = kp * erro[k] + ki * s_int[k] + kd * (erro[k] - erro[k - 1])

    tempo = np.arange(0, N_amostras) * T_amostragem

    ise = np.sum(erro**2) * T_amostragem
    iae = np.sum(np.abs(erro)) * T_amostragem
    itae = np.sum(tempo * np.abs(erro)) * T_amostragem

    y_segment = y_med[:50]
    ref = 1.0
    overshoot = (np.max(y_segment) - ref) / ref * 100 if np.max(y_segment) > ref else 0
    erro_final = abs(ref - y_segment[-1])

    sup, inf = ref * 1.02, ref * 0.98
    indices_fora = np.where((y_segment < inf) | (y_segment > sup))[0]
    tempo_acomod = tempo[indices_fora[-1] + 1] if len(indices_fora) else tempo[0]

    img64 = None
    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(COR["painel"])
        ax.set_facecolor(COR["painel"])

        ax.plot(tempo, y, color=COR["serie_medida"], linewidth=2.2, label="Variável Controlada")
        ax.plot(tempo, yr, "--", color=COR["serie_setpoint"], linewidth=1.8, label="Setpoint")

        ax.set_title(f"PID  |  Kp={Kp:.2f}  Ti={Ti:.2f}  Td={Td:.2f}",
                     color=COR["texto"], fontfamily="monospace", fontsize=11)
        ax.set_xlabel("Tempo (s)", color=COR["texto_sec"])
        ax.set_ylabel("Amplitude", color=COR["texto_sec"])
        ax.tick_params(colors=COR["texto_mudo"])
        ax.grid(True, color=COR["grade"], linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color(COR["borda"])
        legend = ax.legend(facecolor=COR["painel"], edgecolor=COR["borda"], labelcolor=COR["texto_sec"])
        legend.get_frame().set_alpha(0.95)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=140, bbox_inches="tight")
        buf.seek(0)
        img64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

    return {
        "Kp": Kp, "Ti": Ti, "Td": Td,
        "overshoot": round(overshoot, 2),
        "tempo_acomod": round(float(tempo_acomod), 2),
        "erro_final": round(erro_final, 4),
        "ise": round(ise, 2),
        "iae": round(iae, 2),
        "itae": round(itae, 2),
        "grafico": img64,
    }


def buscar_melhores_parametros() -> Optional[Dict]:
    """Otimização por busca em grade (grid search) minimizando ISE + 2*overshoot.

    A penalidade em overshoot evita que a busca empurre Kp/Td para os limites
    da grade em troca de um ISE menor à custa de uma resposta com sobressinal
    alto (o que acontecia minimizando só o ISE)."""
    melhor, melhor_custo = None, 1e9
    for kp in np.linspace(1, 30, 10):
        for ti in np.linspace(0.05, 1, 10):
            for td in np.linspace(0, 0.5, 6):
                try:
                    r = simular_planta(kp, ti, td, plot=False)
                    custo = r["ise"] + 2 * r["overshoot"]
                    if np.isnan(custo) or np.isinf(custo):
                        continue
                    if custo < melhor_custo:
                        melhor_custo, melhor = custo, r
                except Exception:
                    continue
    if melhor is None:
        return None
    return simular_planta(melhor["Kp"], melhor["Ti"], melhor["Td"], plot=True)


def simular_com_protecao_overshoot(kp: float, ti: float, td: float) -> Dict:
    """Reduz Kp/Td iterativamente se o overshoot passar de 30%."""
    kp = max(0.0, min(kp, 50.0))
    ti = max(0.01, min(ti, 5.0))
    td = max(0.0, min(td, 2.0))

    fator_reducao = 0.7
    melhor = None
    for _ in range(5):
        try:
            resultado = simular_planta(Kp=kp, Ti=ti, Td=td, plot=False)
            overshoot = resultado["overshoot"]
            if np.isnan(overshoot) or np.isinf(overshoot):
                raise ValueError("Overshoot inválido")
            melhor = resultado
            if overshoot <= 30:
                break
            kp *= fator_reducao
            td *= fator_reducao
        except Exception:
            kp *= fator_reducao
            td *= fator_reducao

    if melhor is None:
        return {
            "Kp": kp, "Ti": ti, "Td": td,
            "overshoot": None, "tempo_acomod": None, "erro_final": None,
            "ise": None, "iae": None, "itae": None, "grafico": None,
            "erro": "Falha na simulação",
        }
    return simular_planta(Kp=melhor["Kp"], Ti=melhor["Ti"], Td=melhor["Td"], plot=True)


# ============================================================================
# Bloco III — RAG (PDFs técnicos)
# ============================================================================

TRIAGEM_PROMPT = (
    "Você é um agente de triagem para um assistente de engenharia de controle "
    "utilizado em uma planta industrial em malha fechada.\n\n"
    "O sistema possui três funcionalidades principais:\n"
    "1) Responder perguntas teóricas sobre controle e PID\n"
    "2) Simular o comportamento da planta com parâmetros fornecidos\n"
    "3) Otimizar os parâmetros do controlador para melhorar o desempenho\n\n"
    "Dada a mensagem do usuário, retorne SOMENTE um JSON no formato:\n"
    "{\n"
    '  "decisao": "TEORIA" | "SIMULAR" | "OTIMIZAR",\n'
    '  "kp": float | null,\n'
    '  "ti": float | null,\n'
    '  "td": float | null,\n'
    '  "ganho_alvo": float | null,\n'
    '  "erro_alvo": float | null\n'
    "}\n\n"
    "Regras de classificação:\n"
    "- **TEORIA**: perguntas conceituais sobre controle, PID ou comportamento de sistemas.\n"
    "- **SIMULAR**: quando o usuário fornece valores de controlador e deseja ver o comportamento da planta. "
    "Valores podem ser ZERO (0) e devem ser mantidos.\n"
    "- **OTIMIZAR**: quando o usuário quer encontrar automaticamente os parâmetros do controlador.\n"
    "Se um parâmetro não estiver presente na mensagem, retorne null."
)


class TriagemOut(BaseModel):
    decisao: Literal["TEORIA", "SIMULAR", "OTIMIZAR"]
    kp: Optional[float] = None
    ti: Optional[float] = None
    td: Optional[float] = None
    ganho_alvo: Optional[float] = None
    erro_alvo: Optional[float] = None


@st.cache_resource(show_spinner=False)
def build_retriever(_llm):
    """Carrega PDFs de CONTENT_DIR, faz chunking e indexa em FAISS.
    Retorna (retriever, n_paginas, n_chunks, arquivos) ou (None, 0, 0, []).
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    arquivos = []
    for n in CONTENT_DIR.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            arquivos.append(n.name)
        except Exception:
            continue

    if not docs:
        return None, 0, 0, []

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.15, "k": 4},
    )
    return retriever, len(docs), len(chunks), arquivos


PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente especialista em sistemas de controle em malha fechada "
     "aplicados a processos industriais.\n\n"
     "Seu conhecimento baseia-se em documentos técnicos sobre:\n"
     "- Controle PID\n- Métricas de desempenho (ISE, IAE, ITAE, overshoot)\n"
     "- Modelagem de plantas\n- Discretização de sistemas (ZOH)\n"
     "- Controle de pressão em sistemas de distribuição de água\n\n"
     "A saída do sistema corresponde à pressão da rede hidráulica.\n\n"
     "Responda SOMENTE com base no contexto fornecido.\n"
     "Se não houver informação suficiente no contexto, responda apenas: 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto técnico:\n{context}"),
])


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    ini = max(0, pos - janela // 2)
    fim = min(len(txt), pos + janela // 2)
    return txt[ini:fim]


def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source", "")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]


def perguntar_controle_rag(pergunta: str, llm, retriever) -> Dict:
    if retriever is None:
        return {"answer": None, "citacoes": [], "contexto_encontrado": False}

    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    document_chain = (
        {"context": lambda x: x["context"], "input": RunnablePassthrough()}
        | PROMPT_RAG | llm | StrOutputParser()
    )
    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    return {
        "answer": txt,
        "citacoes": formatar_citacoes(docs_relacionados, pergunta),
        "contexto_encontrado": True,
    }


# ============================================================================
# Bloco IV — Agente (LangGraph)
# ============================================================================

class AgentState(TypedDict, total=False):
    pergunta: str
    classificacao: str
    parametros: dict
    resultado: dict
    resposta_tecnica: str
    resposta: str
    citacoes: list


@st.cache_resource(show_spinner=False)
def build_workflow(_llm, _retriever):
    triagem_chain = _llm.with_structured_output(TriagemOut)

    def triagem(mensagem: str) -> dict:
        saida: TriagemOut = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT),
            HumanMessage(content=mensagem),
        ])
        return saida.model_dump()

    def node_triagem(state: AgentState) -> AgentState:
        saida = triagem(state["pergunta"])
        classificacao = saida["decisao"].upper().strip()
        parametros = {
            "kp": saida.get("kp"), "ti": saida.get("ti"), "td": saida.get("td"),
            "ganho_alvo": saida.get("ganho_alvo"), "erro_alvo": saida.get("erro_alvo"),
        }
        return {**state, "classificacao": classificacao, "parametros": parametros}

    def node_teoria(state: AgentState) -> AgentState:
        pergunta = state["pergunta"]
        rag_resp = perguntar_controle_rag(pergunta, _llm, _retriever)

        if rag_resp["contexto_encontrado"]:
            resposta_final, citacoes = rag_resp["answer"], rag_resp["citacoes"]
        else:
            prompt = f"""
Você é um especialista em sistemas de controle em malha fechada.

Explique a pergunta abaixo de forma clara para um estudante de engenharia elétrica.

Pergunta:
{pergunta}

Observação:
Não foram encontrados documentos na base de conhecimento.
Responda com base apenas no conhecimento geral de engenharia de controle.
"""
            resposta_llm = _llm.invoke(prompt)
            resposta_final, citacoes = resposta_llm.content, []

        return {**state, "resposta": resposta_final, "citacoes": citacoes}

    def node_simular(state: AgentState) -> AgentState:
        p = state["parametros"]
        kp = p.get("kp") if p.get("kp") is not None else 10.0
        ti = p.get("ti") if p.get("ti") is not None else 0.1
        td = p.get("td") if p.get("td") is not None else 0.0
        resultado_final = simular_com_protecao_overshoot(kp, ti, td)
        return {**state, "resultado": resultado_final}

    def node_otimizar(state: AgentState) -> AgentState:
        melhor_final = buscar_melhores_parametros()
        return {**state, "resultado": melhor_final}

    def node_resposta_final(state: AgentState) -> AgentState:
        r = state["resultado"]
        texto = f"""
Simulação PID realizada.

Parâmetros do controlador:
Kp = {r["Kp"]}
Ti = {r["Ti"]}
Td = {r["Td"]}

Métricas:
Overshoot: {r["overshoot"]} %
Tempo de acomodação: {r["tempo_acomod"]} s
Erro final: {r["erro_final"]}

ISE: {r["ise"]}
IAE: {r["iae"]}
ITAE: {r["itae"]}
"""
        return {**state, "resposta_tecnica": texto}

    def node_llm(state: AgentState) -> AgentState:
        classificacao = state.get("classificacao")

        if classificacao in ("SIMULAR", "OTIMIZAR"):
            # Resultado de simulação/otimização: reporta os números tal como
            # estão, sem reabrir uma aula de teoria de PID por cima.
            texto = state.get("resposta_tecnica", "")
            acao = "SIMULAÇÃO" if classificacao == "SIMULAR" else "OTIMIZAÇÃO"
            prompt = f"""
Você é um especialista em sistemas de controle industrial.

Abaixo está o resultado de uma {acao} de um controlador PID que já foi
executada. Use exatamente essa palavra ({acao.lower()}) ao se referir ao que
foi feito — não troque por outro termo.

Apresente esse resultado de forma limpa, mantendo EXATAMENTE os valores
numéricos informados (não invente nem arredonde diferente do que está aqui).

Adicione no máximo 2-3 frases de comentário técnico ESPECÍFICO sobre este
resultado (por exemplo, se o overshoot está alto, se o erro final é
satisfatório, se a sintonia parece adequada).

NÃO explique conceitos gerais de controle PID (o que é Kp, Ti, Td, overshoot,
ISE, IAE, ITAE etc.) — o usuário já pediu uma {acao.lower()}, não uma aula de
teoria. Seja direto e objetivo.

Resultado:
{texto}
"""
        else:
            texto = state.get("resposta_tecnica", state.get("resposta", ""))
            prompt = f"""
Você é um especialista em sistemas de controle.

Explique o conteúdo abaixo de forma clara e didática:

{texto}
"""
        resposta = _llm.invoke(prompt)
        return {**state, "resposta": resposta.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("triagem", node_triagem)
    workflow.add_node("simular", node_simular)
    workflow.add_node("teoria", node_teoria)
    workflow.add_node("otimizar", node_otimizar)
    workflow.add_node("resposta_final", node_resposta_final)
    workflow.add_node("llm", node_llm)

    workflow.add_edge(START, "triagem")
    workflow.add_conditional_edges(
        "triagem", lambda x: x["classificacao"],
        {"SIMULAR": "simular", "TEORIA": "teoria", "OTIMIZAR": "otimizar"},
    )
    workflow.add_edge("simular", "resposta_final")
    workflow.add_edge("otimizar", "resposta_final")
    workflow.add_edge("resposta_final", "llm")
    workflow.add_edge("teoria", "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()


def caminho_da_classificacao(classificacao: Optional[str]) -> List[str]:
    """Reconstrói a sequência de nós percorrida no grafo a partir da decisão
    da triagem — o workflow é determinístico (sem loops), então a
    classificação final identifica o caminho inteiro sem precisar de
    streaming de eventos do LangGraph."""
    base = ["__start__", "triagem"]
    if classificacao == "SIMULAR":
        return base + ["simular", "resposta_final", "llm", "__end__"]
    if classificacao == "OTIMIZAR":
        return base + ["otimizar", "resposta_final", "llm", "__end__"]
    if classificacao == "TEORIA":
        return base + ["teoria", "llm", "__end__"]
    return []


@st.cache_data(show_spinner=False)
def render_diagrama_destacado(_app, classificacao: Optional[str]) -> Optional[bytes]:
    """Renderiza o fluxograma do agente destacando os nós que realmente
    executaram na última interação (azul), o início/fim (verde) e os nós
    não utilizados (cinza)."""
    try:
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png

        graph = _app.get_graph()
        mermaid_txt = graph.draw_mermaid(with_styles=False)
        caminho = set(caminho_da_classificacao(classificacao))

        estilos = []
        for no in graph.nodes:
            if no in ("__start__", "__end__"):
                cor = COR["bom"] if no in caminho else COR["borda"]
                estilos.append(f"style {no} fill:{cor},stroke:{cor},color:#ffffff")
            elif no in caminho:
                estilos.append(
                    f"style {no} fill:{COR['serie_medida']},stroke:{COR['serie_medida']},"
                    f"color:#ffffff,stroke-width:3px"
                )
            else:
                estilos.append(
                    f"style {no} fill:{COR['painel_alt']},stroke:{COR['borda']},"
                    f"color:{COR['texto_mudo']}"
                )

        mermaid_final = mermaid_txt + "\n" + "\n".join(estilos)
        return draw_mermaid_png(mermaid_final, background_color=COR["pagina"])
    except Exception:
        return None


# ============================================================================
# Estado da sessão
# ============================================================================

if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ultima_classificacao" not in st.session_state:
    st.session_state.ultima_classificacao = None

llm = get_llm()
retriever, n_paginas, n_chunks, arquivos_pdf = (None, 0, 0, [])
if llm is not None:
    retriever, n_paginas, n_chunks, arquivos_pdf = build_retriever(llm)

app_workflow = build_workflow(llm, retriever) if llm is not None else None

# ============================================================================
# Cabeçalho (banner HMI)
# ============================================================================

status_geral = "on" if llm is not None else "off"
st.markdown(f"""
<div class="scp-banner">
    <div>
        <h1>SCP-01 · SISTEMA DE CONTROLE PID — MALHA FECHADA</h1>
        <p>Planta: G(s) = 0.4 / (s² + 10s + 25) · discretizada por ZOH</p>
    </div>
    <div>{led("SISTEMA " + ("ONLINE" if llm is not None else "OFFLINE"), status_geral)}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar — Painel de controle
# ============================================================================

with st.sidebar:
    st.markdown("### 🏭 PAINEL DE CONTROLE")

    st.markdown(led("LLM (Groq)", "on" if llm is not None else "off"), unsafe_allow_html=True)
    st.markdown(
        led(f"RAG · {n_paginas} pág / {n_chunks} chunks" if retriever else "RAG · sem PDFs",
            "on" if retriever else "warn"),
        unsafe_allow_html=True,
    )
    st.markdown(led("Workflow", "on" if app_workflow is not None else "off"), unsafe_allow_html=True)

    if llm is None:
        st.error("GROQ_API_KEY não configurada. Defina em Colab (userdata), "
                 "st.secrets ou variável de ambiente.")

    st.divider()

    modo = st.radio(
        "MODO DE OPERAÇÃO",
        ["🤖 Assistente (IA)", "🎛️ Manual (Operador)"],
        label_visibility="visible",
    )

    st.divider()

    if modo == "🤖 Assistente (IA)":
        st.caption("Digite sua pergunta ou comando no chat, na aba **📟 Console** "
                   "→ campo na parte de baixo da tela.")
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("**PARÂMETROS DO CONTROLADOR**")
        kp_manual = st.slider("Kp", 0.0, 50.0, 10.0, 0.1)
        ti_manual = st.slider("Ti", 0.01, 5.0, 0.1, 0.01)
        td_manual = st.slider("Td", 0.0, 2.0, 0.0, 0.01)
        col_a, col_b = st.columns(2)
        simular_manual = col_a.button("▶ SIMULAR", type="primary", use_container_width=True)
        otimizar_manual = col_b.button("🔍 OTIMIZAR", use_container_width=True)

    st.markdown(
        f'<div class="scp-footer">{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}<br>'
        f'LangGraph · Groq GPT-OSS-120B</div>',
        unsafe_allow_html=True,
    )

# ============================================================================
# Execução (modo Manual — disparada pelos botões da sidebar)
# ============================================================================

if modo == "🎛️ Manual (Operador)" and simular_manual:
    with st.spinner("SIMULANDO..."):
        st.session_state.resultado = simular_planta(kp_manual, ti_manual, td_manual, plot=True)

elif modo == "🎛️ Manual (Operador)" and otimizar_manual:
    with st.spinner("BUSCANDO PARÂMETROS ÓTIMOS..."):
        st.session_state.resultado = buscar_melhores_parametros()

# ============================================================================
# Corpo principal — tudo em uma única tela (chat + painel de simulação)
# ============================================================================

def render_citacoes(citacoes: List[Dict]) -> None:
    if not citacoes:
        return
    st.markdown("**Citações técnicas:**")
    for c in citacoes:
        st.markdown(
            f'<div class="scp-cite"><div class="fonte">{c["documento"]} · pág. {c["pagina"]}</div>'
            f'{c["trecho"]}</div>',
            unsafe_allow_html=True,
        )


def status_overshoot(v: float):
    if v <= 10:
        return "bom", "OK"
    if v <= 30:
        return "alerta", "ATENÇÃO"
    return "critico", "CRÍTICO"


def status_erro(v: float):
    if v <= 0.05:
        return "bom", "OK"
    if v <= 0.2:
        return "alerta", "ATENÇÃO"
    return "critico", "CRÍTICO"


col_chat, col_painel = st.columns([3, 2], gap="large")

with col_chat:
    st.markdown(f'<div class="scp-section-title">💬 Assistente</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="scp-console">Aguardando comando. Digite uma pergunta ou pedido no campo '
            'abaixo — ex.: "o que é overshoot?", "simule a planta com kp=10 ti=0.5", '
            '"otimize os parâmetros".</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🏭"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_citacoes(msg.get("citacoes", []))

    pergunta_chat = st.chat_input(
        "Digite sua pergunta ou comando (ex.: simule a planta com kp=10 ti=0.5 td=0.05)...",
        disabled=(llm is None),
    )

    if pergunta_chat:
        st.session_state.chat_history.append({"role": "user", "content": pergunta_chat})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(pergunta_chat)

        with st.chat_message("assistant", avatar="🏭"):
            with st.spinner("PROCESSANDO..."):
                try:
                    resultado = app_workflow.invoke(
                        {"pergunta": pergunta_chat}, config={"recursion_limit": 10}
                    )
                    resposta = resultado.get("resposta") or "Sem resposta."
                    citacoes = resultado.get("citacoes", [])
                    if resultado.get("resultado"):
                        st.session_state.resultado = resultado["resultado"]
                    st.session_state.ultima_classificacao = resultado.get("classificacao")
                    st.markdown(resposta)
                    render_citacoes(citacoes)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": resposta, "citacoes": citacoes}
                    )
                except Exception as e:
                    erro_msg = f"⚠️ ALARME · Falha na execução: {e}"
                    st.markdown(erro_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": erro_msg})

with col_painel:
    st.markdown(f'<div class="scp-section-title">📊 Métricas</div>', unsafe_allow_html=True)

    r = st.session_state.resultado
    if r and r.get("overshoot") is not None:
        s_over, l_over = status_overshoot(r["overshoot"])
        s_err, l_err = status_erro(r["erro_final"])

        c1, c2 = st.columns(2)
        c1.markdown(metric_tile("Overshoot", f'{r["overshoot"]:.2f}', "%", s_over, l_over), unsafe_allow_html=True)
        c2.markdown(metric_tile("Erro final", f'{r["erro_final"]:.4f}', "", s_err, l_err), unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        c3.markdown(metric_tile("Tempo de acomodação", f'{r["tempo_acomod"]:.2f}', "s"), unsafe_allow_html=True)
        c4.markdown(metric_tile("ISE", f'{r["ise"]:.2f}'), unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        c5.markdown(metric_tile("IAE", f'{r["iae"]:.2f}'), unsafe_allow_html=True)
        c6.markdown(metric_tile("ITAE", f'{r["itae"]:.2f}'), unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        c7.markdown(metric_tile("Kp", f'{r["Kp"]:.3f}'), unsafe_allow_html=True)
        c8.markdown(metric_tile("Ti", f'{r["Ti"]:.3f}'), unsafe_allow_html=True)
        c9.markdown(metric_tile("Td", f'{r["Td"]:.3f}'), unsafe_allow_html=True)
    else:
        st.info("Nenhuma simulação executada ainda.")

    st.markdown(f'<div class="scp-section-title">📈 Tendência</div>', unsafe_allow_html=True)
    if r and r.get("grafico"):
        img_bytes = base64.b64decode(r["grafico"])
        st.image(img_bytes, use_container_width=True)
    else:
        st.info("Nenhuma simulação executada ainda.")

    st.markdown(f'<div class="scp-section-title">🔀 Workflow</div>', unsafe_allow_html=True)
    if app_workflow is not None:
        rotulo_no = {
            "SIMULAR": "🔧 SIMULAR — nó `simular`",
            "OTIMIZAR": "🧠 OTIMIZAR — nó `otimizar`",
            "TEORIA": "📚 TEORIA — nó `teoria` (RAG)",
        }.get(st.session_state.ultima_classificacao, "— nenhuma execução via chat ainda —")

        st.markdown(
            f'<div style="margin-bottom:10px;">'
            f'<span style="color:{COR["texto_mudo"]};font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Último agente acionado</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
            f'color:{COR["texto"]};">{rotulo_no}</span></div>',
            unsafe_allow_html=True,
        )

        diagrama = render_diagrama_destacado(app_workflow, st.session_state.ultima_classificacao)
        if diagrama:
            st.image(diagrama, use_container_width=True)
            st.markdown(
                f'<div style="font-size:0.78rem;color:{COR["texto_mudo"]};margin-top:8px;">'
                f'<span style="color:{COR["serie_medida"]};">●</span> nó executado nesta interação &nbsp;·&nbsp; '
                f'<span style="color:{COR["bom"]};">●</span> início / fim &nbsp;·&nbsp; '
                f'<span style="color:{COR["texto_mudo"]};">●</span> não utilizado</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Não foi possível renderizar o diagrama (sem acesso à internet?).")
    else:
        st.info("Workflow indisponível — configure a GROQ_API_KEY.")
