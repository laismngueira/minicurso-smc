# -*- coding: utf-8 -*-
"""
SCP-01 — Núcleo compartilhado (planta, PID, RAG, agente)
==========================================================

Toda a lógica de negócio do minicurso (planta RNA + PID, RAG sobre PDFs,
agente LangGraph) vive neste único módulo, sem depender do Streamlit. Tanto
o notebook didático (Blocos II-IV, célula a célula) quanto `app_streamlit.py`
(a interface) importam daqui — evita ter a mesma lógica escrita duas vezes
em lugares diferentes.

`app_streamlit.py` só adiciona por cima: layout, CSS, estado de sessão e
finas camadas de cache (`st.cache_resource`/`st.cache_data`) sobre as
funções de carregamento caras definidas aqui.
"""

import base64
import io
import os
import pathlib
import re
from functools import lru_cache
from typing import Dict, List, Literal, Optional, TypedDict

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

CONTENT_DIR = pathlib.Path(os.environ.get("MINICURSO_CONTENT_DIR", "/content"))

# Paleta validada (dataviz skill): superfícies e tinta em modo escuro, duas
# séries categóricas (slot 1 azul / slot 2 laranja) e paleta de status fixa
# (nunca reaproveitada para séries). Usada tanto nos gráficos matplotlib
# (simular_planta, render_diagrama_destacado) quanto na CSS da interface.
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


# ============================================================================
# Bloco I — Credenciais (LLM)
# ============================================================================

def resolve_groq_key() -> Optional[str]:
    """Resolve a GROQ_API_KEY a partir do Colab (userdata) ou de variável de
    ambiente. Checagens específicas de Streamlit (st.secrets) ficam em
    app_streamlit.py, não aqui, para este módulo não depender do Streamlit."""
    try:
        from google.colab import userdata  # type: ignore
        chave = userdata.get("GROQ_API_KEY")
        if chave:
            return chave
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def get_llm(api_key: Optional[str]) -> Optional[ChatGroq]:
    if not api_key:
        return None
    # "llama-3.3-70b-versatile" (usado no notebook original) foi descontinuado
    # pela Groq e não está mais disponível no catálogo desta chave — testado
    # em 31/08/2026, ver `curl https://api.groq.com/openai/v1/models`.
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, api_key=api_key)


# ============================================================================
# Bloco II — Planta (RNA treinada com dados reais de campo) + PID
# ============================================================================

COLUNAS_ENTRADA_RNA = [
    "Inversor", "PT_1B", "PT_1C", "PT_1D", "PT_2A", "PT_2B",
    "PT_2C", "PT_3A", "PT_5A", "FT_1A", "FT_3A", "FT_4A", "FT_6A",
]
IDX_VARIAVEL_MANIPULADA = 0  # 'Inversor': frequência do inversor, ação do PID
LIMITE_INF_U, LIMITE_SUP_U = 0.0, 100.0  # faixa operacional do inversor (Hz)
# 100 (não 50) para bater com PID_Aut_Inteligente/notebook.ipynb
# (`np.clip(entrada_atual[0], 0, 100)`) — com o limite em 50 os setpoints
# 6/8/7 eram fisicamente inalcançáveis (a RNA só chega a ~4.36 de pressão
# no ponto de operação usado como estado_base, saturando o atuador).
# Acima de 50 Hz a pressão continua subindo e alcança a faixa 6-8 por volta
# de 60-75 Hz.


def _localizar_arquivo_rna(nome: str) -> Optional[pathlib.Path]:
    encontrados = list(CONTENT_DIR.rglob(nome))
    return encontrados[0] if encontrados else None


@lru_cache(maxsize=1)
def carregar_planta_rna():
    """Carrega a RNA (Modelo_AI_v1.h5) e reproduz a normalização exata de
    PID_Aut_Inteligente/notebook.ipynb (scaler.fit no DadosTratados.xlsx
    inteiro, sem separar treino/teste).

    Retorna (modelo, scaler, estado_base) ou (None, None, None) se os
    arquivos não forem encontrados — falha graciosamente em vez de derrubar
    o app.

    `lru_cache` (sem argumentos, então trivialmente cacheável) evita
    recarregar o modelo e reajustar o scaler do disco a cada chamada — o que
    seria desastroso dentro de buscar_melhores_parametros(), que chama
    simular_planta() (e portanto esta função) até 48 vezes seguidas.
    Persiste tanto numa célula de notebook quanto num processo do Streamlit
    (o cache vive no módulo, que fica importado durante toda a sessão).

    Import do TensorFlow é local (lazy) de propósito: TF e o PyTorch do RAG
    (via sentence-transformers) não podem ser inicializados na mesma sessão
    do processo na ordem errada sem segfault (conflito de runtime OpenMP/
    MKL) — importar TF só aqui garante que o PyTorch do `build_retriever()`
    já tenha sido carregado primeiro, já que essa é a ordem em que o app
    executa (RAG é montado antes de qualquer simulação ser disparada)."""
    caminho_modelo = _localizar_arquivo_rna("Modelo_AI_v1.h5")
    caminho_dados = _localizar_arquivo_rna("DadosTratados.xlsx")
    if caminho_modelo is None or caminho_dados is None:
        return None, None, None

    import warnings

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import load_model

    # scaler.transform() recebe um array numpy puro (não um DataFrame com
    # nomes de coluna) em planta_caixa_preta() — cosmético, não afeta o
    # resultado, mas o sklearn avisa a cada chamada; silencia só esse aviso
    # específico (não todos os UserWarning) para não esconder outros por
    # engano.
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    modelo = load_model(caminho_modelo)
    dados = pd.read_excel(caminho_dados)

    X = dados[COLUNAS_ENTRADA_RNA]

    # Igual a PID_Aut_Inteligente/notebook.ipynb: scaler.fit(X) no dataset
    # inteiro, sem train_test_split.
    scaler = StandardScaler()
    scaler.fit(X)

    estado_base = X.iloc[0].values.copy()
    return modelo, scaler, estado_base


def planta_caixa_preta(u: float, modelo, scaler, estado_base) -> float:
    """Aplica o sinal de controle 'u' (frequência do inversor) e devolve a
    pressão medida na planta real (PT_4A), prevista pela RNA."""
    entrada = estado_base.copy()
    entrada[IDX_VARIAVEL_MANIPULADA] = np.clip(u, LIMITE_INF_U, LIMITE_SUP_U)
    entrada_escalada = scaler.transform([entrada]).astype("float32")
    y_pred = modelo(entrada_escalada, training=False).numpy()[0][0]
    return float(y_pred)


def simular_planta(
    # Ganhos de referência do PID_Aut_Inteligente/notebook.ipynb (classe PID,
    # forma paralela: saida = Kp*erro + Ki*integral + Kd*derivada).
    Kp: float = 1.0,
    Ki: float = 0.1,
    Kd: float = 0.05,
    T_amostragem: float = 1.0,
    N_amostras: int = 1200,
    plot: bool = True,
) -> Dict:
    """Kp=1, Ki=0.1, Kd=0.05, 1200 amostras (3 degraus de 400) — reproduz as
    Figuras 7-10 de PID_Aut_Inteligente/UFPB (1).pdf: setpoint 5.0 -> 6.0
    (passo 400) -> 4.0 (passo 800)."""
    modelo, scaler, estado_base = carregar_planta_rna()
    if modelo is None:
        raise FileNotFoundError(
            "Modelo_AI_v1.h5 / DadosTratados.xlsx não encontrados. Envie os "
            "arquivos para a sessão (mesma pasta dos PDFs) antes de simular."
        )

    segmento = N_amostras // 3
    yr = np.zeros(N_amostras)
    y_med = np.zeros(N_amostras)
    yr[0:segmento] = 5.0
    yr[segmento:2 * segmento] = 6.0
    yr[2 * segmento:3 * segmento] = 4.0

    y = np.zeros(N_amostras)
    u = np.zeros(N_amostras)
    erro = np.zeros(N_amostras)

    # Igual a PID_Aut_Inteligente/notebook.ipynb:
    # - entrada_atual = X.iloc[0].values.copy() -> o atuador começa no valor
    #   de "Inversor" da própria linha usada como estado_base (não em zero).
    # - a cada troca de setpoint (passo 400/800) o notebook recria o objeto
    #   PID (`pid = PID(...)`), o que zera self.integral e
    #   self.erro_anterior — reproduzido aqui reiniciando o integrador e o
    #   erro anterior nas mesmas fronteiras.
    u_atual = float(estado_base[IDX_VARIAVEL_MANIPULADA])
    integral_acc = 0.0
    erro_anterior = 0.0

    for k in range(N_amostras):
        if k == segmento or k == 2 * segmento:
            integral_acc = 0.0
            erro_anterior = 0.0

        u[k] = u_atual
        y[k] = planta_caixa_preta(u_atual, modelo, scaler, estado_base)
        y_med[k] = y[k]  # ruído desligado (ver versão original para reativar)
        erro[k] = yr[k] - y_med[k]
        # Mesma forma da classe PID em PID_Aut_Inteligente/notebook.ipynb:
        # self.integral += erro*dt; derivada = (erro-erro_anterior)/dt;
        # saida = Kp*erro + Ki*integral + Kd*derivada.
        integral_acc += erro[k] * T_amostragem
        derivada = (erro[k] - erro_anterior) / T_amostragem
        erro_anterior = erro[k]
        controle = Kp * erro[k] + Ki * integral_acc + Kd * derivada
        # PID incremental: a saída do PID é somada ao valor anterior do
        # atuador (não recalculada do zero), igual a
        # PID_Aut_Inteligente/notebook.ipynb (`entrada_atual[0] += controle`).
        u_atual = np.clip(u_atual + controle, LIMITE_INF_U, LIMITE_SUP_U)

    tempo = np.arange(0, N_amostras) * T_amostragem

    ise = np.sum(erro**2) * T_amostragem
    iae = np.sum(np.abs(erro)) * T_amostragem
    itae = np.sum(tempo * np.abs(erro)) * T_amostragem

    y_segment = y_med[:segmento]
    ref = yr[0]
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

        ax.plot(tempo, y, color=COR["serie_medida"], linewidth=1.1, label="Variável Controlada")
        ax.plot(tempo, yr, "--", color=COR["serie_setpoint"], linewidth=0.9, label="Setpoint")

        ax.set_title(f"PID  |  Kp={Kp:.2f}  Ki={Ki:.2f}  Kd={Kd:.2f}",
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
        "Kp": Kp, "Ki": Ki, "Kd": Kd,
        "overshoot": round(overshoot, 2),
        "tempo_acomod": round(float(tempo_acomod), 2),
        "erro_final": round(erro_final, 4),
        "referencia": float(ref),
        "ise": round(ise, 2),
        "iae": round(iae, 2),
        "itae": round(itae, 2),
        "grafico": img64,
    }


def buscar_melhores_parametros() -> Optional[Dict]:
    """Otimização por busca em grade (grid search) minimizando ISE + 2*overshoot.

    A penalidade em overshoot evita que a busca empurre Kp/Kd para os limites
    da grade em troca de um ISE menor à custa de uma resposta com sobressinal
    alto (o que acontecia minimizando só o ISE).

    Faixa recentrada em torno da referência do PID_Aut_Inteligente (Kp=1,
    Ki=0.1, Kd=0.05). Grade reduzida (4x4x3=48): cada simulação agora roda
    1200 amostras chamando a RNA de verdade a cada uma (bem mais lenta que a
    TF discretizada de 150 amostras usada antes)."""
    melhor, melhor_custo = None, 1e9
    for kp in np.linspace(0.2, 3.0, 4):
        for ki in np.linspace(0.02, 0.3, 4):
            for kd in np.linspace(0.0, 0.2, 3):
                try:
                    r = simular_planta(kp, ki, kd, plot=False)
                    custo = r["ise"] + 2 * r["overshoot"]
                    if np.isnan(custo) or np.isinf(custo):
                        continue
                    if custo < melhor_custo:
                        melhor_custo, melhor = custo, r
                except Exception:
                    continue
    if melhor is None:
        return None
    return simular_planta(melhor["Kp"], melhor["Ki"], melhor["Kd"], plot=True)


def simular_com_protecao_overshoot(kp: float, ki: float, kd: float) -> Dict:
    """Reduz Kp/Kd iterativamente se o overshoot passar de 30%."""
    kp = max(0.0, min(kp, 50.0))
    ki = max(0.0, min(ki, 1.0))
    kd = max(0.0, min(kd, 20.0))

    fator_reducao = 0.7
    melhor = None
    for _ in range(5):
        try:
            resultado = simular_planta(Kp=kp, Ki=ki, Kd=kd, plot=False)
            overshoot = resultado["overshoot"]
            if np.isnan(overshoot) or np.isinf(overshoot):
                raise ValueError("Overshoot inválido")
            melhor = resultado
            if overshoot <= 30:
                break
            kp *= fator_reducao
            kd *= fator_reducao
        except Exception:
            kp *= fator_reducao
            kd *= fator_reducao

    if melhor is None:
        return {
            "Kp": kp, "Ki": ki, "Kd": kd,
            "overshoot": None, "tempo_acomod": None, "erro_final": None,
            "ise": None, "iae": None, "itae": None, "grafico": None,
            "erro": "Falha na simulação",
        }
    return simular_planta(Kp=melhor["Kp"], Ki=melhor["Ki"], Kd=melhor["Kd"], plot=True)


# ============================================================================
# Bloco III — RAG (PDFs técnicos)
# ============================================================================

def build_retriever(llm):
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
    '  "ki": float | null,\n'
    '  "kd": float | null,\n'
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
    ki: Optional[float] = None
    kd: Optional[float] = None
    ganho_alvo: Optional[float] = None
    erro_alvo: Optional[float] = None


class AgentState(TypedDict, total=False):
    pergunta: str
    classificacao: str
    parametros: dict
    resultado: dict
    resposta_tecnica: str
    resposta: str
    citacoes: list


def build_workflow(llm, retriever):
    # method="json_mode" evita um bug do Groq com modelos "reasoning/tool"
    # (ex.: openai/gpt-oss-120b) em que o function-calling forçado do modo
    # padrão gera uma chamada de ferramenta sintética "json" inexistente
    # (BadRequestError: tool call validation failed). O prompt já descreve
    # o formato JSON esperado, então json_mode funciona sem mudanças nele.
    triagem_chain = llm.with_structured_output(TriagemOut, method="json_mode")

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
            "kp": saida.get("kp"), "ki": saida.get("ki"), "kd": saida.get("kd"),
            "ganho_alvo": saida.get("ganho_alvo"), "erro_alvo": saida.get("erro_alvo"),
        }
        return {**state, "classificacao": classificacao, "parametros": parametros}

    def node_teoria(state: AgentState) -> AgentState:
        pergunta = state["pergunta"]
        rag_resp = perguntar_controle_rag(pergunta, llm, retriever)

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
            resposta_llm = llm.invoke(prompt)
            resposta_final, citacoes = resposta_llm.content, []

        return {**state, "resposta": resposta_final, "citacoes": citacoes}

    def node_simular(state: AgentState) -> AgentState:
        p = state["parametros"]
        kp = p.get("kp") if p.get("kp") is not None else 1.0
        ki = p.get("ki") if p.get("ki") is not None else 0.1
        kd = p.get("kd") if p.get("kd") is not None else 0.05
        resultado_final = simular_com_protecao_overshoot(kp, ki, kd)
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
Ki = {r["Ki"]}
Kd = {r["Kd"]}

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

NÃO explique conceitos gerais de controle PID (o que é Kp, Ki, Kd, overshoot,
ISE, IAE, ITAE etc.) — o usuário já pediu uma {acao.lower()}, não uma aula de
teoria. Seja direto e objetivo.

Resultado:
{texto}
"""
        else:
            texto = state.get("resposta_tecnica", state.get("resposta", ""))
            prompt = f"""
Você é um especialista em sistemas de controle.

Explique o conteúdo abaixo de forma clara e didática, em texto corrido e
tabelas quando fizer sentido.

NÃO inclua seções como "resumo visual", "diagrama", "esquema" ou qualquer
tentativa de desenhar um diagrama/gráfico usando apenas texto ou caracteres
ASCII — isso não é um diagrama de verdade, só texto tentando parecer um, e
fica quebrado. Se quiser ilustrar algo visualmente, descreva em palavras ou
use uma tabela.

{texto}
"""
        resposta = llm.invoke(prompt)
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


def render_diagrama_destacado(app, classificacao: Optional[str]) -> Optional[bytes]:
    """Renderiza o fluxograma do agente destacando os nós que realmente
    executaram na última interação (azul), o início/fim (verde) e os nós não
    utilizados (cinza)."""
    try:
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png

        graph = app.get_graph()
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
