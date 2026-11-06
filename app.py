"""Streamlit dashboard: classify, collect human feedback and monitor the model.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core.metrics import CONF_BINS, confidence_hist, load_reference, snapshot
from core.model import HFClassifier
from core.store import Store

LABEL_PT = {"no": "Não provido", "partial": "Parcialmente provido", "yes": "Provido"}
EXAMPLES = json.loads(Path("examples.json").read_text()) if Path("examples.json").exists() else []

st.set_page_config(page_title="Legal NLP Feedback Engine", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Carregando modelo do Hugging Face…")
def classifier() -> HFClassifier:
    return HFClassifier()


@st.cache_resource
def store() -> Store:
    return Store(os.getenv("DB_PATH", "data/feedback.db"))


@st.cache_resource
def reference() -> dict | None:
    return load_reference(os.getenv("REFERENCE_PATH", "reference.json"))


clf, db, ref = classifier(), store(), reference()

st.title("⚖️ Legal NLP Feedback Engine")
st.caption(
    f"Predição do desfecho de recursos (TJAL) com BERTimbau fine-tuned · human-in-the-loop · "
    f"observabilidade online · modelo `{clf.model_version}`"
)

tab_cls, tab_obs, tab_data = st.tabs(["🔎 Classificar & revisar", "📈 Observabilidade", "🗂️ Dados de feedback"])

with tab_cls:
    if EXAMPLES:
        pick = st.selectbox(
            "Exemplo do conjunto de teste (opcional)",
            ["—"] + [f"{i + 1}. {e['text'][:110]}…" for i, e in enumerate(EXAMPLES)],
        )
        default = EXAMPLES[int(pick.split(".")[0]) - 1]["text"] if pick != "—" else ""
    else:
        default = ""
    text = st.text_area("Descrição da decisão (ementa sem o dispositivo final)", value=default, height=180)

    if st.button("Classificar", type="primary", disabled=len(text.strip()) < 20):
        pred = clf.predict(text.strip())
        st.session_state["last"] = {"id": db.log_prediction(text.strip(), pred), "pred": pred}

    last = st.session_state.get("last")
    if last:
        pred = last["pred"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predição", LABEL_PT.get(pred.label, pred.label))
        c2.metric("Confiança", f"{pred.confidence:.1%}")
        c3.metric("Latência", f"{pred.latency_ms:.0f} ms")
        st.bar_chart(pd.Series({LABEL_PT.get(k, k): v for k, v in pred.scores.items()}, name="probabilidade"))

        st.subheader("A predição está correta?")
        ok, wrong = st.columns(2)
        if ok.button("👍 Correta", width="stretch"):
            db.log_feedback(last["id"], True)
            st.success("Feedback registrado — obrigado!")
        with wrong.popover("👎 Incorreta", width="stretch"):
            options = [l for l in clf.labels if l != pred.label]
            corrected = st.radio("Rótulo correto", options, format_func=lambda l: LABEL_PT.get(l, l))
            comment = st.text_input("Comentário (opcional)")
            if st.button("Enviar correção"):
                db.log_feedback(last["id"], False, corrected, comment or None)
                st.success("Correção registrada — entra no próximo ciclo de re-treino.")

with tab_obs:
    df = db.frame()
    snap = snapshot(df, ref)
    m = st.columns(5)
    m[0].metric("Predições", snap.n_predictions)
    m[1].metric("Revisadas por humanos", snap.n_reviewed)
    m[2].metric("Acurácia online", f"{snap.online_accuracy:.1%}" if snap.online_accuracy is not None else "—")
    m[3].metric("Latência p95", f"{snap.latency_p95_ms:.0f} ms" if snap.latency_p95_ms else "—")
    m[4].metric("Drift (PSI)", snap.drift)
    if ref:
        st.caption(
            f"Referência offline (teste): acurácia {ref['test_accuracy']:.1%} · F1-macro {ref['test_f1_macro']:.3f}. "
            "PSI < 0,1 estável · 0,1–0,25 moderado · > 0,25 significativo (mín. 20 predições)."
        )

    if df.empty:
        st.info("Ainda não há tráfego. Classifique alguns textos na primeira aba.")
    else:
        df["created_at"] = pd.to_datetime(df["created_at"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Distribuição de confiança: referência vs. produção**")
            mids = [(a + b) / 2 for a, b in zip(CONF_BINS[:-1], CONF_BINS[1:])]
            rows = [{"confiança": m_, "fração": v, "fonte": "produção"} for m_, v in zip(mids, confidence_hist(df["confidence"]))]
            if ref:
                rows += [{"confiança": m_, "fração": v, "fonte": "referência"} for m_, v in zip(mids, ref["confidence_hist"])]
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows)).mark_bar(opacity=0.6).encode(
                    x=alt.X("confiança:Q", bin=alt.Bin(extent=[0, 1], step=0.1)), y="fração:Q", color="fonte:N",
                    xOffset="fonte:N",
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Latência por requisição (ms)**")
            st.line_chart(df.set_index("id")["latency_ms"])

        reviewed = df.dropna(subset=["is_correct"])
        if len(reviewed):
            st.markdown("**Acurácia online acumulada (feedback humano)**")
            st.line_chart(reviewed["is_correct"].expanding().mean().reset_index(drop=True).rename("acurácia"))

with tab_data:
    export = db.training_export()
    st.markdown(f"**{len(export)} exemplos validados por humanos** prontos para o próximo fine-tuning.")
    st.dataframe(db.frame().drop(columns=["scores"]), width="stretch", hide_index=True)
    st.download_button(
        "Baixar dataset de re-treino (JSONL)",
        export.to_json(orient="records", lines=True, force_ascii=False),
        file_name="feedback_training.jsonl",
        disabled=export.empty,
    )
