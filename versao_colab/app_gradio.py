# -*- coding: utf-8 -*-
"""
SCP-01 — Painel Industrial de Controle PID (interface Gradio)
================================================================

Interface Gradio para o assistente de engenharia de controle do minicurso
(LLM de triagem + RAG sobre PDFs + LangGraph + planta RNA/PID). Toda a
lógica de negócio vive em `scp01_core.py` — este arquivo só cuida de
layout, CSS e do fiação dos eventos da UI por cima das funções definidas lá.

Como rodar localmente:
  export GROQ_API_KEY=...            (ou crie um arquivo .env com essa linha)
  export MINICURSO_CONTENT_DIR=./dados   (pasta com o .h5, o .xlsx e os PDFs)
  python app_gradio.py
"""

import base64
import io
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

# precisa rodar antes de importar scp01_core: CONTENT_DIR é lido de
# MINICURSO_CONTENT_DIR no momento do import do módulo.
load_dotenv()

import scp01_core as core

COR = core.COR

# ============================================================================
# Inicialização (uma vez, no processo do servidor Gradio)
# ============================================================================

GROQ_API_KEY = core.resolve_groq_key()
LLM = core.get_llm(GROQ_API_KEY)

if LLM is not None:
    RETRIEVER, N_PAGINAS, N_CHUNKS, ARQUIVOS_PDF = core.build_retriever(LLM)
else:
    RETRIEVER, N_PAGINAS, N_CHUNKS, ARQUIVOS_PDF = (None, 0, 0, [])

APP_WORKFLOW = core.build_workflow(LLM, RETRIEVER) if LLM is not None else None

# Carregado depois do RAG (torch) de propósito — ver nota em
# core.carregar_planta_rna() sobre a ordem TF/PyTorch.
_MODELO_RNA, _, _ = core.carregar_planta_rna()
RNA_DISPONIVEL = _MODELO_RNA is not None

RESULTADO_INICIAL = None
if RNA_DISPONIVEL:
    try:
        RESULTADO_INICIAL = core.simular_planta(plot=True)
    except Exception:
        RESULTADO_INICIAL = None


# ============================================================================
# Helpers de renderização (HTML/imagem)
# ============================================================================

def _led(texto: str, estado: str) -> str:
    """estado: 'on' | 'off' | 'warn'"""
    cor = {"on": COR["bom"], "off": COR["critico"], "warn": COR["alerta"]}[estado]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
        f'text-transform:uppercase;letter-spacing:0.04em;color:{COR["texto_sec"]};'
        f'padding:4px 10px;border:1px solid {COR["borda"]};border-radius:999px;'
        f'background:{COR["painel_alt"]};margin:2px 6px 2px 0;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{cor};'
        f'box-shadow:0 0 6px {cor};"></span>{texto}</span>'
    )


def _metric_tile(label: str, value: str, unit: str = "", status: Optional[str] = None,
                  status_label: str = "") -> str:
    cor_badge = {"bom": COR["bom"], "alerta": COR["alerta"], "critico": COR["critico"]}.get(status)
    badge_html = ""
    if status and cor_badge:
        badge_html = (
            f'<div style="display:inline-block;margin-top:6px;font-size:0.7rem;font-weight:600;'
            f'letter-spacing:0.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;'
            f'background:{cor_badge}22;color:{cor_badge};border:1px solid {cor_badge}66;">'
            f'{status_label}</div>'
        )
    return f"""
    <div style="background:{COR['painel']};border:1px solid {COR['borda']};border-radius:6px;
                padding:12px 14px;text-align:left;">
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;
                    color:{COR['texto_mudo']};margin-bottom:4px;">{label}</div>
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:600;
                          color:{COR['texto']};">{value}</span>
             <span style="font-size:0.82rem;color:{COR['texto_mudo']};margin-left:4px;">{unit}</span></div>
        {badge_html}
    </div>
    """


def _status_overshoot(v: float) -> Tuple[str, str]:
    if v <= 10:
        return "bom", "OK"
    if v <= 30:
        return "alerta", "ATENÇÃO"
    return "critico", "CRÍTICO"


def _status_erro(v_pct: float) -> Tuple[str, str]:
    if v_pct <= 2:
        return "bom", "OK"
    if v_pct <= 10:
        return "alerta", "ATENÇÃO"
    return "critico", "CRÍTICO"


def render_metrics_panel(r: Optional[Dict]) -> str:
    if not r or r.get("overshoot") is None:
        return f'<div style="color:{COR["texto_mudo"]};padding:8px 2px;">Nenhuma simulação executada ainda.</div>'

    s_over, l_over = _status_overshoot(r["overshoot"])
    ref = r.get("referencia") or 1.0
    s_err, l_err = _status_erro(abs(r["erro_final"]) / ref * 100)

    tiles = [
        _metric_tile("Overshoot", f'{r["overshoot"]:.2f}', "%", s_over, l_over),
        _metric_tile("Erro final", f'{r["erro_final"]:.4f}', "", s_err, l_err),
        _metric_tile("Tempo de acomodação", f'{r["tempo_acomod"]:.2f}', "s"),
        _metric_tile("ISE", f'{r["ise"]:.2f}'),
        _metric_tile("IAE", f'{r["iae"]:.2f}'),
        _metric_tile("ITAE", f'{r["itae"]:.2f}'),
        _metric_tile("Kp", f'{r["Kp"]:.3f}'),
        _metric_tile("Ki", f'{r["Ki"]:.3f}'),
        _metric_tile("Kd", f'{r["Kd"]:.3f}'),
    ]
    grid = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">' + "".join(tiles) + "</div>"
    return grid


def _img_from_b64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _img_from_bytes(b: Optional[bytes]) -> Optional[Image.Image]:
    if not b:
        return None
    return Image.open(io.BytesIO(b))


def render_workflow_label(classificacao: Optional[str]) -> str:
    rotulo = {
        "SIMULAR": "🔧 SIMULAR — nó `simular`",
        "OTIMIZAR": "🧠 OTIMIZAR — nó `otimizar`",
        "TEORIA": "📚 TEORIA — nó `teoria` (RAG)",
    }.get(classificacao, "— nenhuma execução via chat ainda —")
    return (
        f'<div style="margin-bottom:6px;"><span style="color:{COR["texto_mudo"]};font-size:0.8rem;'
        f'text-transform:uppercase;letter-spacing:0.04em;">Último agente acionado</span><br>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.05rem;'
        f'color:{COR["texto"]};">{rotulo}</span></div>'
    )


def render_diagram(classificacao: Optional[str]) -> Optional[Image.Image]:
    if APP_WORKFLOW is None:
        return None
    diagrama_bytes = core.render_diagrama_destacado(APP_WORKFLOW, classificacao)
    return _img_from_bytes(diagrama_bytes)


def render_citacoes_md(citacoes: List[Dict]) -> str:
    if not citacoes:
        return ""
    linhas = ["\n\n**Citações técnicas:**"]
    for c in citacoes:
        linhas.append(f"- *{c['documento']}, pág. {c['pagina']}*: {c['trecho']}")
    return "\n".join(linhas)


# ============================================================================
# Eventos — modo Assistente (chat)
# ============================================================================

def enviar_pergunta(mensagem: str, historico: List[Dict], resultado_atual: Optional[Dict],
                     classificacao_atual: Optional[str]):
    mensagem = (mensagem or "").strip()
    if not mensagem:
        yield (historico, resultado_atual, classificacao_atual, "",
                render_metrics_panel(resultado_atual),
                _img_from_b64(resultado_atual["grafico"]) if resultado_atual and resultado_atual.get("grafico") else None,
                render_workflow_label(classificacao_atual),
                render_diagram(classificacao_atual))
        return

    historico = historico + [{"role": "user", "content": mensagem}]
    historico = historico + [{"role": "assistant", "content": "⏳ Processando..."}]
    yield (historico, resultado_atual, classificacao_atual, "",
            render_metrics_panel(resultado_atual),
            _img_from_b64(resultado_atual["grafico"]) if resultado_atual and resultado_atual.get("grafico") else None,
            render_workflow_label(classificacao_atual),
            render_diagram(classificacao_atual))

    if APP_WORKFLOW is None:
        historico[-1] = {"role": "assistant", "content": "⚠️ ALARME · GROQ_API_KEY não configurada."}
        yield (historico, resultado_atual, classificacao_atual, "",
                render_metrics_panel(resultado_atual),
                _img_from_b64(resultado_atual["grafico"]) if resultado_atual and resultado_atual.get("grafico") else None,
                render_workflow_label(classificacao_atual),
                render_diagram(classificacao_atual))
        return

    try:
        saida = APP_WORKFLOW.invoke({"pergunta": mensagem}, config={"recursion_limit": 10})
        resposta = saida.get("resposta") or "Sem resposta."
        resposta += render_citacoes_md(saida.get("citacoes", []))
        if saida.get("resultado"):
            resultado_atual = saida["resultado"]
        classificacao_atual = saida.get("classificacao", classificacao_atual)
        historico[-1] = {"role": "assistant", "content": resposta}
    except Exception as e:
        historico[-1] = {"role": "assistant", "content": f"⚠️ ALARME · Falha na execução: {e}"}

    trend_img = _img_from_b64(resultado_atual["grafico"]) if resultado_atual and resultado_atual.get("grafico") else None
    yield (historico, resultado_atual, classificacao_atual, "",
            render_metrics_panel(resultado_atual), trend_img,
            render_workflow_label(classificacao_atual), render_diagram(classificacao_atual))


def limpar_conversa():
    return [], None


# ============================================================================
# Eventos — modo Manual (sliders + botões)
# ============================================================================

def simular_manual(kp: float, ki: float, kd: float):
    try:
        r = core.simular_planta(kp, ki, kd, plot=True)
        alarme = ""
    except Exception as e:
        r = None
        alarme = f"⚠️ ALARME · {e}"
    trend_img = _img_from_b64(r["grafico"]) if r and r.get("grafico") else None
    return r, None, alarme, render_metrics_panel(r), trend_img, render_workflow_label(None), render_diagram(None)


def otimizar_manual():
    try:
        r = core.buscar_melhores_parametros()
        alarme = "" if r else "⚠️ ALARME · Nenhuma combinação válida encontrada."
    except Exception as e:
        r = None
        alarme = f"⚠️ ALARME · {e}"
    trend_img = _img_from_b64(r["grafico"]) if r and r.get("grafico") else None
    return r, None, alarme, render_metrics_panel(r), trend_img, render_workflow_label(None), render_diagram(None)


# ============================================================================
# Layout
# ============================================================================

CSS = f"""
:root {{ color-scheme: dark; }}
.gradio-container {{
    background-color: {COR['pagina']} !important;
    font-family: 'Inter', sans-serif;
}}
#scp-banner {{
    background: linear-gradient(180deg, {COR['painel_alt']} 0%, {COR['painel']} 100%);
    border: 1px solid {COR['borda']};
    border-left: 4px solid {COR['serie_medida']};
    border-radius: 6px;
    padding: 14px 20px;
    margin-bottom: 10px;
}}
#scp-banner h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    letter-spacing: 0.04em;
    margin: 0;
    color: {COR['texto']};
}}
#scp-banner p {{
    margin: 2px 0 0 0;
    color: {COR['texto_mudo']};
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.scp-section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COR['texto_sec']};
    border-bottom: 1px solid {COR['borda']};
    padding-bottom: 6px;
    margin: 8px 0 10px 0;
}}
"""

STATUS_LLM = "on" if LLM is not None else "off"
STATUS_RAG = "on" if RETRIEVER is not None else "warn"
STATUS_RNA = "on" if RNA_DISPONIVEL else "off"

banner_html = f"""
<div id="scp-banner">
    <h1>SCP-01 · SISTEMA DE CONTROLE PID — MALHA FECHADA</h1>
    <p>Planta: RNA treinada com dados de campo (PT_4A) · sistema de distribuição de água</p>
</div>
<div>
    {_led("SISTEMA " + ("ONLINE" if LLM is not None else "OFFLINE"), STATUS_LLM)}
    {_led("LLM (Groq)", STATUS_LLM)}
    {_led(f"RAG · {N_PAGINAS} pág / {N_CHUNKS} chunks" if RETRIEVER else "RAG · sem PDFs", STATUS_RAG)}
    {_led("RNA · modelo carregado" if RNA_DISPONIVEL else "RNA · arquivos ausentes", STATUS_RNA)}
</div>
"""

avisos = []
if not RNA_DISPONIVEL:
    avisos.append(
        "**Modelo_AI_v1.h5 / DadosTratados.xlsx não encontrados.** Configure "
        "`MINICURSO_CONTENT_DIR` apontando para a pasta com esses arquivos (e os PDFs)."
    )
if LLM is None:
    avisos.append("**GROQ_API_KEY não configurada.** Defina em `.env` ou como variável de ambiente.")


with gr.Blocks(title="SCP-01 · Controle PID") as demo:
    gr.HTML(banner_html)
    if avisos:
        gr.Markdown("\n\n".join(f"⚠️ {a}" for a in avisos))

    resultado_state = gr.State(RESULTADO_INICIAL)
    classificacao_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("🤖 Assistente (IA)"):
                    chatbot = gr.Chatbot(
                        height=460,
                        avatar_images=(None, None),
                        placeholder="Aguardando comando. Digite uma pergunta ou pedido no campo abaixo — ex.: "
                                     '"o que é overshoot?", "simule a planta com kp=1 ki=0.1 kd=0.05", '
                                     '"otimize os parâmetros".',
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Digite sua pergunta ou comando...",
                            show_label=False,
                            scale=5,
                            interactive=(LLM is not None),
                        )
                        enviar_btn = gr.Button("Enviar", scale=1, variant="primary", interactive=(LLM is not None))
                    limpar_btn = gr.Button("🗑️ Limpar conversa", size="sm")

                with gr.Tab("🎛️ Manual (Operador)"):
                    gr.Markdown("**Parâmetros do controlador**")
                    kp_slider = gr.Slider(0.0, 50.0, value=1.0, step=0.1, label="Kp")
                    ki_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Ki")
                    kd_slider = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Kd")
                    with gr.Row():
                        simular_btn = gr.Button("▶ Simular", variant="primary")
                        otimizar_btn = gr.Button("🔍 Otimizar (leva alguns minutos)")
                    alarme_manual = gr.Markdown("")

        with gr.Column(scale=2):
            gr.HTML('<div class="scp-section-title">📊 Métricas</div>')
            metrics_html = gr.HTML(render_metrics_panel(RESULTADO_INICIAL))

            gr.HTML('<div class="scp-section-title">📈 Tendência</div>')
            trend_image = gr.Image(
                value=_img_from_b64(RESULTADO_INICIAL["grafico"]) if RESULTADO_INICIAL else None,
                show_label=False, container=False,
            )

            gr.HTML('<div class="scp-section-title">🔀 Workflow</div>')
            workflow_label = gr.HTML(render_workflow_label(None))
            workflow_image = gr.Image(
                value=render_diagram(None), show_label=False, container=False,
            )

    outputs_comuns = [chatbot, resultado_state, classificacao_state, msg_box,
                       metrics_html, trend_image, workflow_label, workflow_image]

    enviar_btn.click(enviar_pergunta, [msg_box, chatbot, resultado_state, classificacao_state], outputs_comuns)
    msg_box.submit(enviar_pergunta, [msg_box, chatbot, resultado_state, classificacao_state], outputs_comuns)
    limpar_btn.click(limpar_conversa, None, [chatbot, classificacao_state])

    outputs_manual = [resultado_state, classificacao_state, alarme_manual,
                       metrics_html, trend_image, workflow_label, workflow_image]
    simular_btn.click(simular_manual, [kp_slider, ki_slider, kd_slider], outputs_manual)
    otimizar_btn.click(otimizar_manual, None, outputs_manual)

    gr.HTML(
        f'<div style="color:{COR["texto_mudo"]};font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.72rem;text-align:right;margin-top:16px;border-top:1px solid {COR["borda"]};'
        f'padding-top:8px;">LangGraph · Groq GPT-OSS-120B</div>'
    )


if __name__ == "__main__":
    import sys

    # No Colab não há acesso direto ao localhost da VM — precisa do túnel
    # público do Gradio (link https://xxxx.gradio.live) para abrir a UI.
    em_colab = "google.colab" in sys.modules
    demo.queue().launch(css=CSS, theme=gr.themes.Base(), share=em_colab)
