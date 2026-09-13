"""
app_sangue.py
=============
Interface de apoio ao diagnóstico de doenças a partir de exames de
sangue (hemograma + glicemia + pressão arterial), classificando o
paciente em: Saudável, Hipertenso, Leucemia ou Diabetes.

Recursos:
- Cadastro clínico completo do paciente/exame.
- Formulário com os valores laboratoriais (sem upload de imagem).
- Seleção de um modelo específico OU votação (ensemble) entre modelos.
- Painel de métricas (acurácia, precisão, recall, F1-score) por modelo.
- Histórico de detecções persistido em banco de dados (SQLite).

Para rodar:
    streamlit run app_sangue.py

IMPORTANTE: esta ferramenta é um sistema de APOIO à decisão clínica.
O laudo final é sempre de responsabilidade do médico responsável.
"""

from datetime import date, datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import database_sangue as db
from blood_detector import (
    CLASSES,
    ENSEMBLE_LABEL,
    FEATURE_ORDER,
    PROCESSING_STEPS,
    REFERENCE_RANGES,
    DiagnosisResult,
    EnsembleResult,
    build_model_registry,
)

# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------
db.init_db()


@st.cache_resource(show_spinner=False)
def get_registry():
    return build_model_registry()


registry = get_registry()

st.set_page_config(
    page_title="VitaLabs Sangue",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "Saudável": "#2E9E5B",
    "Hipertenso": "#E8792D",
    "Leucemia": "#D64550",
    "Diabetes": "#5B4FCF",
}

CLASS_ICONS = {
    "Saudável": "🟢",
    "Hipertenso": "🟠",
    "Leucemia": "🔴",
    "Diabetes": "🟣",
}

# Quantidade máxima de exames exibidos por página no histórico de detecções.
ITENS_POR_PAGINA_HISTORICO = 5

ACHADOS_POR_CLASSE = {
    "Saudável": "Parâmetros hematológicos, glicêmicos e pressóricos <strong>dentro da normalidade</strong>, sem sinais sugestivos das condições avaliadas.",
    "Hipertenso": "Padrão <strong>sugestivo de hipertensão arterial</strong>, com base nos valores pressóricos informados.",
    "Leucemia": "Padrão hematológico <strong>sugestivo de doença hematológica maligna (leucemia)</strong>, com base nas alterações do hemograma (ex.: leucocitose acentuada e/ou plaquetopenia).",
    "Diabetes": "Padrão <strong>sugestivo de diabetes mellitus</strong>, com base na glicemia de jejum informada.",
}

st.markdown(
    """
    <style>
        .main { background-color: #F7F9FB; }
        .stApp { font-family: 'Segoe UI', sans-serif; }

        .clinical-header {
            background: linear-gradient(90deg, #7A0C2E 0%, #B3143E 100%);
            padding: 1.4rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .clinical-header h1 { margin: 0; font-size: 1.6rem; }
        .clinical-header p { margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            border: 1px solid #E3E8EE;
            margin-bottom: 0.5rem;
        }
        .metric-card .label { font-size: 0.75rem; color: #5A6B7B; font-weight: 600; text-transform: uppercase; }
        .metric-card .value { font-size: 1.35rem; color: #7A0C2E; font-weight: 700; }

        .diagnosis-box { border-radius: 12px; padding: 1.6rem; margin-top: 1rem; }

        .disclaimer {
            background-color: #FFF7E6;
            border-left: 4px solid #E8A33D;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #6B4E16;
            margin-top: 1.2rem;
        }
        .vote-row {
            display: flex; justify-content: flex-end; align-items: center;
            border: 1px solid #E3E8EE; border-radius: 8px;
            padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.9rem;
        }

        /* --- Caixa de exames/dados no histórico --- */
        .exam-info-row {
            display: flex; justify-content: space-between;
            padding: 0.3rem 0; font-size: 0.88rem;
            border-bottom: 1px dashed #E3E8EE;
        }
        .exam-info-row .rotulo { color: #5A6B7B; }
        .exam-info-row .valor { color: #2C3B4A; font-weight: 600; text-align: right; }

        /* --- Faz a caixa de tags do multiselect (sintomas, comorbidades,
             etc.) ficar em uma única linha, com rolagem horizontal, em vez
             de cortar os itens já adicionados. A rolagem com a roda do
             mouse é habilitada via JavaScript no final do arquivo. --- */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
            flex-shrink: 0;
            margin: 3px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="clinical-header">
        <h1>🩸 Sistema de Apoio ao Diagnóstico por Exame de Sangue | Vita Labs</h1>
        <p>Triagem assistida por Inteligência Artificial para Saudável, Hipertensão, Leucemia e Diabetes a partir de hemograma, glicemia e pressão arterial</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral — Seleção de modelo(s) e métricas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Modelo de Análise")

    available_models = registry.list_models()
    modo_opcoes = available_models + [ENSEMBLE_LABEL]

    modo_selecionado = st.selectbox(
        "Selecione o modelo ou o modo de votação",
        options=modo_opcoes,
        index=len(modo_opcoes) - 1,  # padrão: votação
        help="Escolha um modelo específico ou combine vários modelos por votação (ensemble).",
    )

    is_ensemble = modo_selecionado == ENSEMBLE_LABEL

    if is_ensemble:
        st.caption("Modelos participantes da votação:")
        modelos_ensemble = st.multiselect(
            "Modelos incluídos na votação",
            options=available_models,
            default=available_models,
            label_visibility="collapsed",
        )
        metodo_votacao = st.radio(
            "Método de consolidação do voto",
            options=["soft", "hard"],
            format_func=lambda x: "Média de probabilidades (soft-vote)" if x == "soft" else "Maioria simples (hard-vote)",
        )
    else:
        modelos_ensemble = [modo_selecionado]
        metodo_votacao = "soft"

    st.divider()
    st.markdown("### 📊 Métricas de Desempenho")
    st.caption("Métricas macro-médias entre as 4 classes (Saudável, Hipertenso, Leucemia, Diabetes).")

    metrics_to_show = (
        {name: registry.get_metrics(name) for name in modelos_ensemble}
        if is_ensemble
        else {modo_selecionado: registry.get_metrics(modo_selecionado)}
    )

    for name, m in metrics_to_show.items():
        with st.expander(f"**{name}**", expanded=not is_ensemble):
            st.caption(m.get("descricao", ""))
            st.markdown(
                f"""
                <div class="metric-card"><div class="label">Acurácia</div><div class="value">{m['acuracia']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Precisão (macro)</div><div class="value">{m['precisao']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Recall (macro)</div><div class="value">{m['recall']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">F1-Score (macro)</div><div class="value">{m['f1_score']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Especificidade (macro)</div><div class="value">{m['especificidade']*100:.1f}%</div></div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    total_stats = db.get_stats_summary()
    st.markdown("### 🗂️ Base de Dados")
    st.caption(
        f"{total_stats['total']} exame(s) registrados · "
        f"{total_stats['saudavel']} Saudável · {total_stats['hipertenso']} Hipertenso · "
        f"{total_stats['leucemia']} Leucemia · {total_stats['diabetes']} Diabetes"
    )

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <strong>Uso como apoio à decisão.</strong><br>
        Este sistema não substitui a avaliação clínica e o laudo do
        médico responsável.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Abas principais
# ---------------------------------------------------------------------------
tab_novo, tab_historico = st.tabs(["🩺 Novo Diagnóstico", "🗂️ Histórico de Detecções"])

# =============================================================================
# ABA 1 — NOVO DIAGNÓSTICO
# =============================================================================
with tab_novo:
    st.markdown("### 1. Dados do Paciente e do Exame")

    with st.form("patient_form", clear_on_submit=False):
        st.markdown("**Identificação**")
        c1, c2 = st.columns(2)
        nome = c1.text_input("Nome completo do paciente *")
        prontuario = c2.text_input("Prontuário / ID do exame")

        c3, c4, c5 = st.columns(3)
        data_nascimento = c3.date_input(
            "Data de nascimento", value=None, min_value=date(1900, 1, 1), max_value=date.today(), format="DD/MM/YYYY"
        )
        sexo = c4.selectbox("Sexo", ["Não informado", "Masculino", "Feminino", "Outro"])
        data_exame = c5.date_input("Data do exame", value=date.today(), format="DD/MM/YYYY")

        medico_solicitante = st.text_input("Médico solicitante")
        indicacao_clinica = st.text_area(
            "Indicação clínica / motivo do exame",
            placeholder="Ex.: Check-up de rotina, investigação de fadiga e emagrecimento...",
            height=70,
        )

        idade_manual = None
        if not data_nascimento:
            idade_manual = st.number_input("Idade (anos) — caso não informe a data de nascimento", min_value=0, max_value=120, value=0, step=1)

        st.markdown("**Quadro clínico**")
        c6, c7 = st.columns(2)
        sintomas = c6.multiselect(
            "Sintomas",
            ["Fadiga", "Febre", "Sede excessiva", "Poliúria", "Perda de peso",
             "Cefaleia", "Tontura", "Sangramentos/hematomas", "Palidez",
             "Dor óssea", "Assintomático", "Outros"],
        )
        comorbidades = c7.multiselect(
            "Comorbidades / fatores de risco",
            ["Diabetes prévio", "Hipertensão prévia", "Histórico familiar de leucemia",
             "Obesidade", "Tabagismo", "Sedentarismo", "Idade avançada (>65 anos)", "Outros"],
        )

        st.markdown("**Hemograma e exames complementares**")
        h1, h2, h3 = st.columns(3, vertical_alignment="bottom")
        hemoglobina = h1.number_input(
            "Hemoglobina (g/dL)", min_value=0.0, max_value=25.0, value=13.5, step=0.1,
            help=REFERENCE_RANGES["hemoglobina"],
        )
        hematocrito = h2.number_input(
            "Hematócrito (%)", min_value=0.0, max_value=70.0, value=41.0, step=0.1,
            help=REFERENCE_RANGES["hematocrito"],
        )
        hemacias = h3.number_input(
            "Hemácias (milhões/mm³)", min_value=0.0, max_value=10.0, value=4.8, step=0.1,
            help=REFERENCE_RANGES["hemacias"],
        )

        h4, h5 = st.columns(2)
        leucocitos = h4.number_input(
            "Leucócitos (células/mm³)", min_value=0, max_value=200000, value=7500, step=100,
            help=REFERENCE_RANGES["leucocitos"],
        )
        plaquetas = h5.number_input(
            "Plaquetas (em milhares/mm³)", min_value=0, max_value=1000, value=250, step=5,
            help=REFERENCE_RANGES["plaquetas"],
        )

        h6, h7, h8 = st.columns(3)
        glicemia = h6.number_input(
            "Glicemia de jejum (mg/dL)", min_value=0, max_value=600, value=90, step=1,
            help=REFERENCE_RANGES["glicemia"],
        )
        pressao_sistolica = h7.number_input(
            "Pressão sistólica (mmHg)", min_value=0, max_value=260, value=120, step=1,
            help=REFERENCE_RANGES["pressaoSistolica"],
        )
        pressao_diastolica = h8.number_input(
            "Pressão diastólica (mmHg)", min_value=0, max_value=180, value=80, step=1,
            help=REFERENCE_RANGES["pressaoDiastolica"],
        )

        observacoes = st.text_area("Observações clínicas adicionais", height=60)
        responsavel_tecnico = st.text_input("Responsável técnico pela análise (biomédico/médico)")

        submitted = st.form_submit_button(
            "🔎 Iniciar Análise Diagnóstica", type="primary", use_container_width=True
        )

    st.divider()
    st.markdown("### 2. Resultado da Análise")
    result_placeholder = st.empty()

    if not submitted:
        result_placeholder.info("O relatório aparecerá aqui...")

    if submitted:
            idade = db.calculate_age(data_nascimento) if data_nascimento else (int(idade_manual) if idade_manual else None)

            if not nome:
                result_placeholder.error("Informe ao menos o **nome do paciente** para prosseguir.")
            elif idade is None:
                result_placeholder.error("Informe a **data de nascimento** ou a **idade** do paciente.")
            elif is_ensemble and not modelos_ensemble:
                result_placeholder.error("Selecione ao menos um modelo para a votação.")
            else:
                with result_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    n_steps = len(PROCESSING_STEPS)
                    for i, step in enumerate(PROCESSING_STEPS, start=1):
                        status_text.markdown(f"🧪 **{step}**")
                        progress_bar.progress(i / n_steps)
                        time_module_sleep = __import__("time").sleep
                        time_module_sleep(0.3)

                    paciente_df = pd.DataFrame(
                        {
                            "idade": [idade],
                            "hemoglobina": [hemoglobina],
                            "hematocrito": [hematocrito],
                            "hemacias": [hemacias],
                            "leucocitos": [leucocitos],
                            "plaquetas": [plaquetas],
                            "glicemia": [glicemia],
                            "pressaoSistolica": [pressao_sistolica],
                            "pressaoDiastolica": [pressao_diastolica],
                        }
                    )[FEATURE_ORDER]

                    if is_ensemble:
                        result: EnsembleResult = registry.predict_ensemble(
                            paciente_df, model_names=modelos_ensemble, method=metodo_votacao
                        )
                    else:
                        single: DiagnosisResult = registry.predict_single(modo_selecionado, paciente_df)
                        result = EnsembleResult(
                            label=single.label,
                            confidence=single.confidence,
                            probabilities=single.probabilities,
                            individual_results=[single],
                            votes={c: (1 if c == single.label else 0) for c in CLASSES},
                            unanimous=True,
                        )

                    progress_bar.empty()
                    status_text.empty()

                # -- Persistência no banco de dados --------------------------------
                patient_id = db.insert_patient(
                    {
                        "nome": nome,
                        "prontuario": prontuario,
                        "data_nascimento": str(data_nascimento) if data_nascimento else None,
                        "idade": idade,
                        "sexo": sexo,
                        "data_exame": str(data_exame),
                        "medico_solicitante": medico_solicitante,
                        "indicacao_clinica": indicacao_clinica,
                        "sintomas": sintomas,
                        "comorbidades": comorbidades,
                        "observacoes": observacoes,
                        "hemoglobina": hemoglobina,
                        "hematocrito": hematocrito,
                        "hemacias": hemacias,
                        "leucocitos": leucocitos,
                        "plaquetas": plaquetas,
                        "glicemia": glicemia,
                        "pressao_sistolica": pressao_sistolica,
                        "pressao_diastolica": pressao_diastolica,
                    }
                )

                votos_individuais = [
                    {
                        "modelo": r.model_name,
                        "label": r.label,
                        "confidence": r.confidence,
                        "probabilities": r.probabilities,
                    }
                    for r in result.individual_results
                ]

                db.insert_detection(
                    {
                        "patient_id": patient_id,
                        "modo_analise": "votacao" if is_ensemble else "modelo_unico",
                        "modelo_utilizado": ENSEMBLE_LABEL if is_ensemble else modo_selecionado,
                        "label": result.label,
                        "confidence": result.confidence,
                        "prob_saudavel": result.probabilities.get("Saudável"),
                        "prob_hipertenso": result.probabilities.get("Hipertenso"),
                        "prob_leucemia": result.probabilities.get("Leucemia"),
                        "prob_diabetes": result.probabilities.get("Diabetes"),
                        "votos_individuais": votos_individuais if is_ensemble else None,
                        "concordancia_modelos": int(result.unanimous) if is_ensemble else None,
                        "responsavel_tecnico": responsavel_tecnico,
                        "observacoes_laudo": observacoes,
                    }
                )

                # -- Exibição do resultado ------------------------------------------
                with result_placeholder.container():
                    color = CLASS_COLORS.get(result.label, "#0B3D66")
                    icon = CLASS_ICONS.get(result.label, "⚪")
                    achado = ACHADOS_POR_CLASSE.get(result.label, "")

                    modelo_txt = (
                        f"Votação entre {len(modelos_ensemble)} modelos ({metodo_votacao})"
                        if is_ensemble
                        else modo_selecionado
                    )

                    st.markdown(
                        f"""
                        <div class="diagnosis-box" style="border: 2px solid {color};">
                            <h3>{icon} Impressão diagnóstica assistida por IA: {result.label}</h3>
                            <p>{achado}</p>
                            <p><strong>Confiança do resultado:</strong> {result.confidence*100:.1f}%</p>
                            <p style="font-size:0.85rem; color:#5A6B7B;">
                                Paciente: {nome} · Exame: {prontuario or "não identificado"} ·
                                Modelo(s): {modelo_txt} · Processado em {datetime.now().strftime('%d/%m/%Y %H:%M')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Distribuição de Probabilidades (consolidada)")
                    prob_cols = st.columns(4)
                    for col, classe in zip(prob_cols, CLASSES):
                        with col:
                            st.metric(classe, f"{result.probabilities[classe]*100:.1f}%")
                            st.progress(result.probabilities[classe])

                    if is_ensemble:
                        st.markdown("#### Detalhamento da Votação por Modelo")
                        if result.unanimous:
                            st.success("✅ Todos os modelos concordaram quanto ao resultado.")
                        else:
                            st.warning(
                                "⚠️ Os modelos **divergiram** quanto ao resultado — recomenda-se "
                                "atenção redobrada na correlação clínica e revisão manual do exame."
                            )

                        for r in result.individual_results:
                            vote_color = CLASS_COLORS.get(r.label, "#5A6B7B")
                            st.markdown(
                                f"""
                                <div class="vote-row">
                                    <span><strong>{r.model_name}:&nbsp</strong></span>
                                    <span style="color:{vote_color}; font-weight:700;">{r.label} ({r.confidence*100:.1f}%)</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        votos_str = " · ".join(f"{c} = {result.votes.get(c, 0)}" for c in CLASSES)
                        st.caption(
                            f"Votos: {votos_str} · "
                            f"Método de consolidação: {'média de probabilidades' if metodo_votacao == 'soft' else 'maioria simples'}"
                        )

                    st.markdown(
                        """
                        <div class="disclaimer">
                        📌 <strong>Conduta sugerida:</strong> correlacionar este resultado com
                        a história clínica, exame físico e, se necessário, exames
                        complementares (ex.: mielograma/biópsia de medula óssea para
                        suspeita de leucemia, MAPA para hipertensão, hemoglobina glicada
                        para diabetes). O laudo definitivo deve ser emitido pelo
                        profissional de saúde responsável.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success("Registro salvo no histórico de detecções.")

# =============================================================================
# ABA 2 — HISTÓRICO
# =============================================================================
with tab_historico:
    st.markdown("### Histórico de Detecções")

    fc1, fc2, fc3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    busca = fc1.text_input("Buscar por nome do paciente ou prontuário")
    filtro_resultado = fc2.selectbox("Resultado", ["Todos"] + CLASSES)
    if fc3.button("🔄 Atualizar", use_container_width=True):
        st.rerun()

    registros = db.fetch_history(nome_filtro=busca or None, resultado_filtro=filtro_resultado)

    if not registros:
        st.info("Nenhum registro encontrado. Realize uma análise na aba **Novo Diagnóstico**.")
    else:
        # -- Controle de qual exame está com os detalhes abertos --------------
        if "detalhe_id" not in st.session_state:
            st.session_state["detalhe_id"] = None

        ids_disponiveis = {r["detection_id"] for r in registros}
        if st.session_state["detalhe_id"] not in ids_disponiveis:
            # Se o exame selecionado não está mais na lista filtrada (ou é a
            # primeira visita à aba), abre os detalhes do mais recente.
            st.session_state["detalhe_id"] = registros[0]["detection_id"]

        # -- Controle de paginação ----------------------------------------------
        # Reinicia a página para a primeira sempre que os filtros de busca
        # mudam, para não deixar o usuário "perdido" numa página vazia.
        filtro_atual = (busca, filtro_resultado)
        if st.session_state.get("historico_filtro_anterior") != filtro_atual:
            st.session_state["historico_filtro_anterior"] = filtro_atual
            st.session_state["historico_pagina"] = 0

        if "historico_pagina" not in st.session_state:
            st.session_state["historico_pagina"] = 0

        total_paginas = max(1, -(-len(registros) // ITENS_POR_PAGINA_HISTORICO))  # ceil
        st.session_state["historico_pagina"] = min(
            st.session_state["historico_pagina"], total_paginas - 1
        )
        pagina_atual = st.session_state["historico_pagina"]

        inicio = pagina_atual * ITENS_POR_PAGINA_HISTORICO
        fim = inicio + ITENS_POR_PAGINA_HISTORICO
        registros_pagina = registros[inicio:fim]

        st.caption(
            f"{len(registros)} exame(s) encontrado(s) · "
            f"Exibindo {inicio + 1}–{min(fim, len(registros))} de {len(registros)}"
        )

        # -- Cabeçalho da tabela ------------------------------------------------
        col_widths = [1.3, 1.8, 1.1, 0.9, 1.1, 0.9]
        headers = ["Data/Hora", "Paciente", "Resultado", "Confiança", "Concordância", ""]
        header_cols = st.columns(col_widths)
        for col, texto in zip(header_cols, headers):
            col.markdown(f"<span style='font-size:0.8rem; color:#5A6B7B; font-weight:700; text-transform:uppercase;'>{texto}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.2rem 0 0.6rem 0;'>", unsafe_allow_html=True)

        # -- Linhas da tabela (apenas da página atual), cada uma com botão
        #    "Visualizar" ---------------------------------------------------
        for r in registros_pagina:
            is_selecionado = r["detection_id"] == st.session_state["detalhe_id"]
            with st.container(border=True):
                row_cols = st.columns(col_widths, vertical_alignment="center")
                row_cols[0].write(r["data_deteccao"])
                row_cols[1].write(f"**{r['nome']}**")

                cor = CLASS_COLORS.get(r["label"], "#5A6B7B")
                icone = CLASS_ICONS.get(r["label"], "⚪")
                row_cols[2].markdown(
                    f"<span style='color:{cor}; font-weight:700;'>{icone} {r['label']}</span>",
                    unsafe_allow_html=True,
                )
                row_cols[3].write(f"{r['confidence']*100:.1f}%")

                concordancia = r.get("concordancia_modelos")
                if concordancia == 1:
                    row_cols[4].markdown("✅ Unânime")
                elif concordancia == 0:
                    row_cols[4].markdown("⚠️ Divergência")
                else:
                    row_cols[4].write("—")

                botao_label = "🔎 Visualizando" if is_selecionado else "👁️ Visualizar"
                if row_cols[5].button(
                    botao_label, key=f"ver_{r['detection_id']}",
                    use_container_width=True,
                    type="primary" if is_selecionado else "secondary",
                ):
                    st.session_state["detalhe_id"] = r["detection_id"]
                    st.rerun()

        # -- Controles de paginação ----------------------------------------------
        if total_paginas > 1:
            nav_cols = st.columns([1, 2, 1])
            with nav_cols[0]:
                if st.button("⬅️ Anterior", disabled=(pagina_atual == 0), use_container_width=True):
                    st.session_state["historico_pagina"] -= 1
                    st.rerun()
            with nav_cols[1]:
                st.markdown(
                    f"<div style='text-align:center; padding-top:0.4rem; color:#5A6B7B;'>"
                    f"Página {pagina_atual + 1} de {total_paginas}</div>",
                    unsafe_allow_html=True,
                )
            with nav_cols[2]:
                if st.button("Próxima ➡️", disabled=(pagina_atual >= total_paginas - 1), use_container_width=True):
                    st.session_state["historico_pagina"] += 1
                    st.rerun()

        # -- Painel de detalhes do exame selecionado ----------------------------
        registro = next((r for r in registros if r["detection_id"] == st.session_state["detalhe_id"]), None)

        if registro:
            st.markdown("---")
            st.markdown(f"### 📋 Detalhes do Exame — {registro['nome']}")

            cor = CLASS_COLORS.get(registro["label"], "#0B3D66")
            icone = CLASS_ICONS.get(registro["label"], "⚪")
            achado = ACHADOS_POR_CLASSE.get(registro["label"], "")

            st.markdown(
                f"""
                <div class="diagnosis-box" style="border: 2px solid {cor};">
                    <h3>{icone} Resultado: {registro['label']}</h3>
                    <p>{achado}</p>
                    <p><strong>Confiança do resultado:</strong> {registro['confidence']*100:.1f}%</p>
                    <p style="font-size:0.85rem; color:#5A6B7B;">
                        Exame: {registro.get('prontuario') or "não identificado"} ·
                        Modelo(s): {registro['modelo_utilizado']} ·
                        Processado em {registro['data_deteccao']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dcol1, dcol2 = st.columns([1, 1.1], gap="large")

            with dcol1:
                st.markdown("#### 🧑‍⚕️ Dados do paciente e do exame")
                info_rows = [
                    ("Idade", f"{registro.get('idade', '—')} anos"),
                    ("Sexo", registro.get("sexo") or "—"),
                    ("Prontuário", registro.get("prontuario") or "—"),
                    ("Data do exame", registro.get("data_exame") or "—"),
                    ("Médico solicitante", registro.get("medico_solicitante") or "—"),
                    ("Indicação clínica", registro.get("indicacao_clinica") or "—"),
                    ("Responsável técnico", registro.get("responsavel_tecnico") or "—"),
                ]
                info_html = "".join(
                    f"""<div class="exam-info-row">
                            <span class="rotulo">{rotulo}</span>
                            <span class="valor">{valor}</span>
                        </div>"""
                    for rotulo, valor in info_rows
                )
                st.markdown(info_html, unsafe_allow_html=True)

                st.markdown("#### 🧪 Valores laboratoriais")
                lab_rows = [
                    ("Hemoglobina", f"{registro.get('hemoglobina', '—')} g/dL"),
                    ("Hematócrito", f"{registro.get('hematocrito', '—')} %"),
                    ("Hemácias", f"{registro.get('hemacias', '—')} milhões/mm³"),
                    ("Leucócitos", f"{registro.get('leucocitos', '—')} /mm³"),
                    ("Plaquetas", f"{registro.get('plaquetas', '—')} mil/mm³"),
                    ("Glicemia", f"{registro.get('glicemia', '—')} mg/dL"),
                    ("Pressão arterial", f"{registro.get('pressao_sistolica', '—')}/{registro.get('pressao_diastolica', '—')} mmHg"),
                ]
                lab_html = "".join(
                    f"""<div class="exam-info-row">
                            <span class="rotulo">{rotulo}</span>
                            <span class="valor">{valor}</span>
                        </div>"""
                    for rotulo, valor in lab_rows
                )
                st.markdown(lab_html, unsafe_allow_html=True)

            with dcol2:
                st.markdown("#### 📊 Probabilidades por classe")
                prob_cols = st.columns(2)
                probs_por_classe = [
                    ("Saudável", "prob_saudavel"),
                    ("Hipertenso", "prob_hipertenso"),
                    ("Leucemia", "prob_leucemia"),
                    ("Diabetes", "prob_diabetes"),
                ]
                for i, (classe, campo) in enumerate(probs_por_classe):
                    valor = registro.get(campo)
                    if valor is not None:
                        with prob_cols[i % 2]:
                            st.metric(f"{CLASS_ICONS.get(classe, '')} {classe}", f"{valor*100:.1f}%")
                            st.progress(valor)

                if registro.get("votos_individuais"):
                    import json
                    votos = json.loads(registro["votos_individuais"])
                    st.markdown("#### 🗳️ Detalhamento da votação por modelo")
                    for v in votos:
                        vote_color = CLASS_COLORS.get(v["label"], "#5A6B7B")
                        probs_modelo = v.get("probabilities", {})
                        detalhe_probs = " · ".join(
                            f"{c}: {probs_modelo.get(c, 0)*100:.1f}%" for c in CLASSES
                        ) if probs_modelo else ""
                        st.markdown(
                            f"""
                            <div class="vote-row" style="flex-direction: column; align-items: flex-start;">
                                <div style="display:flex; justify-content: space-between; width: 100%;">
                                    <strong>{v['modelo']}:&nbsp</strong>
                                    <span style="color:{vote_color}; font-weight:700;">{v['label']} ({v['confidence']*100:.1f}%)</span>
                                </div>
                                <div style="font-size:0.78rem; color:#5A6B7B; margin-top:0.2rem;">{detalhe_probs}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Selecione um exame na lista acima para ver os detalhes.")

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Sistema de apoio ao diagnóstico laboratorial · Uso restrito a profissionais de saúde · "
    "Não substitui avaliação médica presencial."
)

# ---------------------------------------------------------------------------
# Habilita rolagem horizontal com a roda do mouse na caixa de tags dos
# multiselects (sintomas, comorbidades, etc.).
#
# Colocado no FINAL do arquivo de propósito: assim, quando este script
# rodar dentro do iframe do componente, todos os widgets da página já
# foram renderizados no documento pai, evitando a corrida em que o script
# tenta encontrar elementos que ainda não existem.
#
# Em vez de depender de uma estrutura fixa de divs aninhadas (que muda
# entre versões do Streamlit), o script localiza o container correto
# dinamicamente: ele parte das próprias "tags" (os itens já selecionados)
# e usa o elemento-pai delas — que é, por definição, a caixa que precisa
# rolar. O estilo (uma linha só, com overflow horizontal) é aplicado
# diretamente via JavaScript, então não depende do CSS ter acertado o
# seletor. O listener de "wheel" é registrado em fase de captura, para
# rodar antes de qualquer outro handler que possa interceptar o evento.
# ---------------------------------------------------------------------------
components.html(
    """
    <script>
    function habilitarScrollHorizontalMultiselect() {
        let doc;
        try {
            doc = window.parent.document;
        } catch (erro) {
            return; // sem acesso ao documento pai (não deveria acontecer aqui)
        }

        const tags = doc.querySelectorAll('div[data-testid="stMultiSelect"] [data-baseweb="tag"]');
        const containers = new Set();
        tags.forEach((tag) => {
            if (tag.parentElement) {
                containers.add(tag.parentElement);
            }
        });

        containers.forEach((caixa) => {
            // Aplica o estilo diretamente no elemento, sem depender do CSS.
            caixa.style.display = "flex";
            caixa.style.flexWrap = "nowrap";
            caixa.style.overflowX = "auto";
            caixa.style.overflowY = "hidden";
            caixa.style.scrollbarWidth = "thin";

            if (!caixa.dataset.scrollHorizontalAtivo) {
                caixa.dataset.scrollHorizontalAtivo = "true";
                caixa.addEventListener(
                    "wheel",
                    function (evento) {
                        if (caixa.scrollWidth > caixa.clientWidth) {
                            evento.preventDefault();
                            evento.stopPropagation();
                            caixa.scrollLeft += evento.deltaY;
                        }
                    },
                    { passive: false, capture: true }
                );
            }
        });

        return containers.size;
    }

    // Primeira tentativa imediata.
    habilitarScrollHorizontalMultiselect();

    // Reaplica sempre que o Streamlit re-renderizar a página (troca de
    // aba, mudança de filtro, nova seleção, etc.), já que os elementos
    // podem ser recriados.
    try {
        const observador = new MutationObserver(habilitarScrollHorizontalMultiselect);
        observador.observe(window.parent.document.body, { childList: true, subtree: true });
    } catch (erro) {
        // ignora se não for possível observar (ambiente restrito)
    }

    // Rede de segurança: tenta novamente a cada meio segundo, caso o
    // MutationObserver perca alguma atualização. A função é segura de
    // rodar repetidamente (só registra o listener uma vez por elemento).
    setInterval(habilitarScrollHorizontalMultiselect, 500);
    </script>
    """,
    height=0,
)