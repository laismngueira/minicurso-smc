# Plano de ação — trocar a planta discretizada pela planta RNA (PID_Aut_Inteligente)

## 1. Objetivo

Substituir a planta atual do `app_streamlit.py` — uma função de transferência
analítica `G(s) = 0.4 / (s² + 10s + 25)` discretizada por ZOH — pela planta
caixa-preta usada originalmente no `minicurso_smc (1).py`: uma **rede neural
(RNA)** treinada com dados reais de campo de um sistema de distribuição de
água, localizada em `PID_Aut_Inteligente/` (`RNA.py`, `Modelo_AI_v1.h5`,
`DadosTratados.xlsx`).

Isso muda a planta de "modelo analítico" para "modelo empírico identificado a
partir de dados reais" — mais fiel ao sistema real, ao custo de depender de
arquivos externos e de dependências mais pesadas (TensorFlow, scikit-learn).

## 2. Estado atual vs. estado alvo

| | Atual | Alvo |
|---|---|---|
| Planta | `ct.tf([0.4], [1, 10, 25])` (`app_streamlit.py:341`) | RNA `Modelo_AI_v1.h5` (Keras) |
| Como avança no tempo | Equação de diferenças da TF discretizada por ZOH | Chamada direta ao modelo por amostra (`y[k] = planta(u[k-1])`) |
| Dependências | `control`, `numpy`, `matplotlib` | + `tensorflow`, `scikit-learn`, `pandas`, `openpyxl` |
| Arquivos externos | nenhum | `Modelo_AI_v1.h5` + `DadosTratados.xlsx` |
| Setpoints (`yr`) | 1.0 → 2.0 → 0.75 (abstrato) | 6.0 → 8.0 → 7.0 (pressão `PT_4A`, kPa) |
| Variável manipulada | sinal de controle abstrato | frequência do inversor, 0–50 Hz |

## 3. Arquivos afetados

- `app_streamlit.py` — principal alvo das mudanças (planta, `simular_planta`,
  textos da interface, thresholds de status).
- `colab_launch_streamlit.py` — adicionar instalação das novas dependências e
  instruções para subir `Modelo_AI_v1.h5` / `DadosTratados.xlsx` para
  `/content/` (mesmo padrão já usado para os PDFs do RAG).
- Nenhuma mudança necessária em `codigoMinicurso.py` / `minicurso_smc (1).py`
  (ficam como referência histórica).

## 4. Dependências novas

```
tensorflow scikit-learn pandas openpyxl
```

Observações:
- `tensorflow` é uma dependência pesada (centenas de MB) — aumenta bastante o
  tempo de `pip install` e o tempo de import na primeira execução, tanto no
  Colab quanto localmente.
- Testado neste ambiente local: `tensorflow` **não está instalado** no venv
  atual (`.venv-run`) — será preciso instalar antes de validar.
- Verificar compatibilidade da versão do `.h5` salvo (`RNA.py` usa
  `tensorflow.keras`) com a versão do TensorFlow que for instalada — modelos
  `.h5` antigos podem exigir `tf.keras.models.load_model(..., compile=False)`
  se a versão de salvamento divergir muito da versão de carregamento.

## 5. Passo a passo técnico

### 5.1. Carregar os artefatos da RNA (uma vez, cacheado)

Criar uma função `@st.cache_resource` equivalente a:

```python
def _localizar_arquivo(nome, diretorio):
    encontrados = list(diretorio.glob(nome))
    if not encontrados:
        raise FileNotFoundError(...)
    return encontrados[0]

@st.cache_resource(show_spinner=False)
def carregar_planta_rna():
    modelo = load_model(_localizar_arquivo("Modelo_AI_v1.h5", CONTENT_DIR))
    dados = pd.read_excel(_localizar_arquivo("DadosTratados.xlsx", CONTENT_DIR))

    colunas_entrada = [
        "Inversor", "PT_1B", "PT_1C", "PT_1D", "PT_2A", "PT_2B",
        "PT_2C", "PT_3A", "PT_5A", "FT_1A", "FT_3A", "FT_4A", "FT_6A",
    ]
    X = dados[colunas_entrada]
    y = dados["PT_4A"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.7, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    estado_base = X.iloc[0].values.copy()
    return modelo, scaler, estado_base
```

**Importante:** manter `test_size=0.7, random_state=42` idênticos ao
`RNA.py` original — é o que garante que o `StandardScaler` reproduza a mesma
normalização usada no treinamento do modelo. Qualquer divergência aqui faz a
RNA prever valores sem sentido (o modelo foi treinado com dados normalizados
de uma forma específica).

Tratar ausência dos arquivos de forma graciosa (igual já é feito com os PDFs
do RAG): se não encontrar, mostrar um aviso claro na interface em vez de
derrubar o app, e desabilitar a simulação até os arquivos serem enviados.

### 5.2. Reescrever `planta_caixa_preta(u)`

```python
def planta_caixa_preta(u, modelo, scaler, estado_base,
                        limite_inf=0.0, limite_sup=50.0):
    entrada = estado_base.copy()
    entrada[0] = np.clip(u, limite_inf, limite_sup)  # 'Inversor'
    entrada_escalada = scaler.transform([entrada]).astype("float32")
    y_pred = modelo(entrada_escalada, training=False).numpy()[0][0]
    return float(y_pred)
```

### 5.3. Adaptar `simular_planta()` (`app_streamlit.py:344`)

- Remover a linha `sistema_discreto = PLANTA.sample(T_amostragem, method="zoh")`
  e o cálculo de `y[k]` via `NumD`/`DenD` (equação de diferenças da TF) —
  isso só existe para simular uma planta *analítica* discretizada.
- Substituir por `y[k] = planta_caixa_preta(u[k-1], modelo, scaler, estado_base)`
  — a RNA já é a "planta discreta" (recebe um valor de controle e devolve uma
  medição), não precisa de discretização adicional.
- `T_amostragem` deixa de discretizar um modelo contínuo e passa a ser
  **apenas** o intervalo de tempo lógico do laço (usado para o eixo de tempo
  e a ponderação das integrais ISE/IAE/ITAE) — papel que já cumpre hoje,
  então nenhuma mudança de assinatura é necessária.
- Atualizar os setpoints `yr` para a escala real de pressão:
  `yr[0:50] = 6.0; yr[50:100] = 8.0; yr[100:150] = 7.0` (em vez de
  `1.0 / 2.0 / 0.75`).
- Revisar os limites de saturação do sinal de controle: `_LIMITE_INF_U = 0.0`,
  `_LIMITE_SUP_U = 50.0` (frequência do inversor em Hz) — os sliders de
  Kp/Ti/Td do modo Manual continuam válidos, mas o **sinal de controle** `u`
  passa a ser saturado nessa faixa em vez de não ter saturação explícita.

### 5.4. Ajustar thresholds de status nas Métricas

Os limiares de "OK / ATENÇÃO / CRÍTICO" (`status_overshoot`, `status_erro`,
em `app_streamlit.py`) foram calibrados para a escala abstrata (setpoint ≈ 1).
Com setpoints de 6–8 kPa, o **erro final** especialmente precisa de novos
limiares em unidades absolutas (ex.: `erro_final <= 0.05` deixa de fazer
sentido — provavelmente precisa virar algo como `<= 0.1` kPa, a validar
empiricamente rodando algumas simulações).

### 5.5. Atualizar textos da interface

- Banner principal (`app_streamlit.py`, bloco `scp-banner`): trocar
  `"Planta: G(s) = 0.4 / (s² + 10s + 25) · discretizada por ZOH"` por algo como
  `"Planta: RNA treinada com dados de campo (PT_4A) · sistema de distribuição de água"`.
- Novo LED de status na sidebar: `"RNA · modelo carregado"` / `"RNA · arquivos ausentes"`,
  no mesmo padrão dos LEDs de LLM/RAG/Workflow já existentes.
- Placeholder do chat (`"Ex.: simule a planta com kp=10 ti=0.5 td=0.05"`) pode
  continuar igual — os parâmetros do PID não mudam, só a planta por trás.

### 5.6. `colab_launch_streamlit.py`

- Adicionar `tensorflow scikit-learn pandas openpyxl` à célula de instalação.
- Atualizar o comentário do topo do arquivo para instruir o envio de
  `Modelo_AI_v1.h5` e `DadosTratados.xlsx` para `/content/`, além dos PDFs.

## 6. Compatibilidade com os modos existentes

Tanto o **modo Assistente** (chat → LangGraph → `simular_planta`/
`buscar_melhores_parametros`) quanto o **modo Manual** (sliders → `simular_planta`
direto) chamam a mesma função `simular_planta()` — a troca de planta é
transparente para os dois, nenhuma mudança adicional é necessária nesses
fluxos além do que já está no passo 5.3.

O diagrama do Workflow (aba destacada) e a lógica de triagem/RAG não são
afetados — só a planta muda.

## 7. Plano de testes

1. **Teste de carregamento:** confirmar que `carregar_planta_rna()` encontra
   os arquivos e carrega sem erro (testar com e sem os arquivos presentes,
   para validar a mensagem de aviso gracioso).
2. **Teste de sanidade da RNA:** chamar `planta_caixa_preta(u)` para alguns
   valores de `u` (ex.: 0, 25, 50 Hz) e conferir que a pressão prevista varia
   de forma monotônica/plausível (compatível com o comportamento físico
   esperado: mais frequência → mais pressão).
3. **Teste de malha fechada:** rodar `simular_planta()` no modo Manual com
   Kp/Ti/Td padrão e observar se o sistema converge para os novos setpoints
   (6 → 8 → 7) dentro de um tempo razoável — ajustar os ganhos padrão do
   Manual se a RNA tiver dinâmica muito diferente da TF atual.
4. **Teste via chat:** repetir os mesmos comandos já usados nas validações
   anteriores ("simule a planta com kp=..., ti=..., td=...", "otimize os
   parâmetros") e conferir que os relatórios do `node_llm` continuam corretos
   (valores batendo com o `resultado` bruto).
5. **Teste de performance:** medir o tempo de resposta da otimização por
   grid search (600 simulações) com a RNA — chamadas ao modelo Keras são mais
   lentas que a equação de diferenças atual; pode ser necessário reduzir a
   grade de busca ou adicionar cache/paralelismo se ficar muito lento.
6. **Teste visual:** rodar via Playwright (mesmo processo usado nas mudanças
   anteriores) para confirmar que a interface renderiza corretamente e que os
   valores nas Métricas/Tendência fazem sentido na nova escala.

## 8. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| `Modelo_AI_v1.h5` / `DadosTratados.xlsx` não disponíveis no Colab | Mensagem de erro clara + instrução de upload, sem derrubar o app (mesmo padrão do RAG sem PDFs) |
| Versão do TensorFlow incompatível com o `.h5` salvo | Testar `load_model(..., compile=False)`; se necessário, re-salvar o modelo em formato `.keras` mais portável |
| Grid search da otimização fica lento com a RNA | Medir primeiro; se necessário, reduzir a grade (menos pontos) ou cachear previsões repetidas |
| Thresholds de status (OK/ATENÇÃO/CRÍTICO) errados na nova escala | Recalibrar empiricamente após rodar algumas simulações reais |
| `tensorflow` aumenta muito o tempo de setup local/Colab | Aceitável — é o preço de usar a planta real; documentar no `colab_launch_streamlit.py` |

## 9. Rollback

Manter a versão atual (planta discretizada) versionada antes da troca — como
não há git neste projeto ainda, recomenda-se copiar `app_streamlit.py` para
`app_streamlit_tf_discretizada.py` como backup antes de aplicar as mudanças
deste plano, para poder reverter rapidamente comparando os dois arquivos se
algo na RNA não se comportar bem.

## 10. Ordem de execução sugerida

1. Copiar `app_streamlit.py` → backup.
2. Instalar `tensorflow scikit-learn pandas openpyxl` no venv de teste.
3. Confirmar que `Modelo_AI_v1.h5` e `DadosTratados.xlsx` estão acessíveis
   (neste ambiente, já existem em `PID_Aut_Inteligente/`).
4. Implementar `carregar_planta_rna()` + `planta_caixa_preta()` (passos 5.1–5.2).
5. Adaptar `simular_planta()` (passo 5.3).
6. Rodar os testes de sanidade (passo 7, itens 1–2) antes de mexer na UI.
7. Recalibrar thresholds de status (passo 5.4) com base nos resultados reais.
8. Atualizar textos da interface (passo 5.5).
9. Atualizar `colab_launch_streamlit.py` (passo 5.6).
10. Rodar a bateria completa de testes (passo 7, itens 3–6) e revisar
    visualmente via Playwright antes de considerar concluído.
