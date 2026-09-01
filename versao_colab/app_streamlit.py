# -*- coding: utf-8 -*-
"""
SCP-01 — Painel Industrial de Controle PID
============================================

Interface Streamlit para o assistente de engenharia de controle do minicurso.
Mantém a planta e o agente originais (LLM de triagem + RAG sobre PDFs +
LangGraph), apenas trocando a interface Gradio por um painel de estilo
SCADA/HMI.

Toda a lógica de negócio (planta RNA, PID, RAG, agente LangGraph) vive em
`scp01_core.py` — este arquivo só cuida de layout, CSS, estado de sessão e
finas camadas de cache (`st.cache_resource`/`st.cache_data`) por cima das
funções de carregamento caras definidas lá.

Planta: RNA (Modelo_AI_v1.h5) treinada com dados reais de campo de um
sistema de distribuição de água (PID_Aut_Inteligente) — recebe a frequência
do inversor e devolve a pressão prevista (PT_4A). Depende de
Modelo_AI_v1.h5 e DadosTratados.xlsx estarem acessíveis (mesma pasta dos
PDFs). PID em forma paralela (Kp, Ki, Kd diretos, sem Ti/Td) igual à classe
PID de PID_Aut_Inteligente/notebook.ipynb — ganhos padrão Kp=1, Ki=0.1,
Kd=0.05 @ Ts=1s (1200 amostras, degraus de 400) reproduzem as Figuras
7-10 de PID_Aut_Inteligente/UFPB (1).pdf ("Kd=5" na legenda da Figura 7 é
um erro de digitação do relatório — confirmado rodando o código literal
do PDF: com Kd=5 o segundo degrau oscila (não bate com a figura), com
Kd=0.05 fica suave e o overshoot do primeiro degrau bate 16.4% ≈ os 16.6%
medidos na imagem).

Como rodar (Colab, com proxy nativo):
  1) Envie este arquivo, scp01_core.py, os PDFs de conhecimento e os
     arquivos da RNA (Modelo_AI_v1.h5, DadosTratados.xlsx) para a sessão do
     Colab (aba de arquivos, /content/).
  2) Rode o notebook de lançamento (colab_launch_streamlit.py) célula a
     célula.

Como rodar localmente:
  export GROQ_API_KEY=...
  streamlit run app_streamlit.py
"""

import base64
import os
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

import scp01_core as core

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

COR = core.COR

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
# Credenciais (Colab ou ambiente local) + camadas de cache do Streamlit
# sobre as funções caras definidas em scp01_core
# ============================================================================

def _get_groq_key() -> Optional[str]:
    if hasattr(st, "secrets"):
        try:
            chave = st.secrets.get("GROQ_API_KEY")
            if chave:
                return chave
        except Exception:
            pass
    return core.resolve_groq_key()


GROQ_API_KEY = _get_groq_key()


@st.cache_resource(show_spinner=False)
def get_llm():
    return core.get_llm(GROQ_API_KEY)


@st.cache_resource(show_spinner=False)
def build_retriever(_llm):
    return core.build_retriever(_llm)


@st.cache_resource(show_spinner=False)
def build_workflow(_llm, _retriever):
    return core.build_workflow(_llm, _retriever)


@st.cache_data(show_spinner=False)
def render_diagrama_destacado(_app, classificacao: Optional[str]) -> Optional[bytes]:
    return core.render_diagrama_destacado(_app, classificacao)


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

# Carregado depois do RAG (torch) de propósito — ver nota em
# core.carregar_planta_rna() sobre a ordem TF/PyTorch.
_modelo_rna, _, _ = core.carregar_planta_rna()
rna_disponivel = _modelo_rna is not None

# Cenário padrão: roda uma simulação com os ganhos de referência assim que
# o app abre, antes de qualquer pergunta do usuário, para o painel (Métricas
# / Tendência / Workflow) já vir populado em vez de "nenhuma simulação
# executada ainda". Só roda uma vez por sessão (resultado fica em cache no
# session_state depois disso).
if st.session_state.resultado is None and rna_disponivel:
    try:
        with st.spinner("CARREGANDO CENÁRIO PADRÃO..."):
            st.session_state.resultado = core.simular_planta(plot=True)
    except Exception:
        pass

# ============================================================================
# Cabeçalho (banner HMI)
# ============================================================================

status_geral = "on" if llm is not None else "off"
st.markdown(f"""
<div class="scp-banner">
    <div>
        <h1>SCP-01 · SISTEMA DE CONTROLE PID — MALHA FECHADA</h1>
        <p>Planta: RNA treinada com dados de campo (PT_4A) · sistema de distribuição de água</p>
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
    st.markdown(
        led("RNA · modelo carregado" if rna_disponivel else "RNA · arquivos ausentes",
            "on" if rna_disponivel else "off"),
        unsafe_allow_html=True,
    )
    if not rna_disponivel:
        st.error("Modelo_AI_v1.h5 / DadosTratados.xlsx não encontrados. Envie os "
                 "arquivos para a sessão (mesma pasta dos PDFs) antes de simular.")

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
        # Padrão = Figuras 7-10 de PID_Aut_Inteligente/UFPB (1).pdf
        # (classe PID: Kp=1, Ki=0.1, Kd=0.05).
        kp_manual = st.slider("Kp", 0.0, 50.0, 1.0, 0.1)
        ki_manual = st.slider("Ki", 0.0, 1.0, 0.1, 0.01)
        kd_manual = st.slider("Kd", 0.0, 1.0, 0.05, 0.01)
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

if "erro_manual" not in st.session_state:
    st.session_state.erro_manual = None

if modo == "🎛️ Manual (Operador)" and simular_manual:
    st.session_state.erro_manual = None
    with st.spinner("SIMULANDO..."):
        try:
            st.session_state.resultado = core.simular_planta(kp_manual, ki_manual, kd_manual, plot=True)
        except Exception as e:
            st.session_state.erro_manual = str(e)

elif modo == "🎛️ Manual (Operador)" and otimizar_manual:
    st.session_state.erro_manual = None
    with st.spinner("BUSCANDO PARÂMETROS ÓTIMOS..."):
        try:
            st.session_state.resultado = core.buscar_melhores_parametros()
        except Exception as e:
            st.session_state.erro_manual = str(e)

if st.session_state.erro_manual:
    st.markdown(led(f"ALARME · {st.session_state.erro_manual}", "off"), unsafe_allow_html=True)

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


def status_erro(v_pct: float):
    """v_pct: erro final como % da referência — proporcional em vez de
    absoluto, já que a escala do setpoint muda conforme a planta (era ~1 na
    TF discretizada, é ~6-8 kPa na RNA)."""
    if v_pct <= 2:
        return "bom", "OK"
    if v_pct <= 10:
        return "alerta", "ATENÇÃO"
    return "critico", "CRÍTICO"


col_chat, col_painel = st.columns([3, 2], gap="large")

with col_chat:
    st.markdown(f'<div class="scp-section-title">💬 Assistente</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="scp-console">Aguardando comando. Digite uma pergunta ou pedido no campo '
            'abaixo — ex.: "o que é overshoot?", "simule a planta com kp=1 ki=0.1 kd=0.05", '
            '"otimize os parâmetros".</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🏭"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_citacoes(msg.get("citacoes", []))

    pergunta_chat = st.chat_input(
        "Digite sua pergunta ou comando (ex.: simule a planta com kp=1 ki=0.1 kd=0.05)...",
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
        ref = r.get("referencia") or 1.0
        s_err, l_err = status_erro(abs(r["erro_final"]) / ref * 100)

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
        c8.markdown(metric_tile("Ki", f'{r["Ki"]:.3f}'), unsafe_allow_html=True)
        c9.markdown(metric_tile("Kd", f'{r["Kd"]:.3f}'), unsafe_allow_html=True)
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
