from pathlib import Path
import io
import json
import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")

@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "best_mlp_model.keras"
    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.keras"
    model = keras.models.load_model(model_path)
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, preprocessor, label_encoder, metrics

try:
    model, preprocessor, label_encoder, metrics = load_artifacts()
except Exception as exc:
    st.error("No fue posible cargar los artefactos del modelo.")
    st.code(str(exc))
    st.stop()

EXPECTED_FEATURES = [
    "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course"
]
FEATURES = metrics.get("features", EXPECTED_FEATURES)
LEAKAGE_COLUMNS = {"math_score", "reading_score", "writing_score", "average_score"}
REQUIRED_METRICS = {"best_epoch", "test_loss", "test_accuracy", "test_f1_macro", "comparison"}

missing_metrics = REQUIRED_METRICS.difference(metrics)
if missing_metrics:
    st.error(f"metrics.json incompleto: {sorted(missing_metrics)}")
    st.stop()
if FEATURES != EXPECTED_FEATURES:
    st.error(f"FEATURES no coincide con el contrato esperado: {FEATURES}")
    st.stop()
if set(FEATURES) & LEAKAGE_COLUMNS:
    st.error("Se detectó data leakage en FEATURES.")
    st.stop()
for key in ["test_loss", "test_accuracy", "test_f1_macro"]:
    try:
        if not np.isfinite(float(metrics[key])):
            raise ValueError
    except Exception:
        st.error(f"La métrica {key} no es numérica o finita.")
        st.stop()
comparison_contract = metrics.get("comparison", [])
if not comparison_contract or any(
    "modelo" not in row or "f1_macro" not in row for row in comparison_contract
):
    st.error("La estructura comparison de metrics.json no es válida.")
    st.stop()

def predict_one(record):
    frame = pd.DataFrame([record])[FEATURES]
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    probabilities = model.predict(transformed, verbose=0)[0]
    position = int(probabilities.argmax())
    return label_encoder.inverse_transform([position])[0], probabilities

def safe_text(value, fallback="Por completar"):
    value = str(value).strip() if value is not None else ""
    return value if value else fallback

def add_box(ax, x, y, width, height, title, body, color="#EAF2FF", body_size=8.2):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.1, edgecolor="#1D4E89", facecolor=color,
        transform=ax.transAxes
    )
    ax.add_patch(patch)
    center_x = x + width / 2
    ax.text(center_x, y+height-.023, title, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="#12385B",
            va="top", ha="center")
    wrap_width = max(26, int(width*96))
    wrapped = "\n".join(textwrap.fill(p, width=wrap_width) for p in str(body).split("\n"))
    if str(body).strip():
        ax.text(center_x, y+height-.060, wrapped, transform=ax.transAxes,
                fontsize=body_size, color="#263238", va="top", ha="center",
                linespacing=1.18)

def create_poster(team, app_url, video_url, app_preview=None):
    fig = plt.figure(figsize=(11.69,16.54), dpi=150, facecolor="#F6F9FC")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    accuracy=float(metrics.get("test_accuracy",0)); f1=float(metrics.get("test_f1_macro",0))
    loss=float(metrics.get("test_loss",0)); epoch=metrics.get("best_epoch","N/D")
    comparison=metrics.get("comparison",[])
    best=max(comparison,key=lambda r:r.get("f1_macro",-1)) if comparison else {"modelo":"N/D","f1_macro":0}

    ax.add_patch(plt.Rectangle((0,.915),1,.085,color="#12385B",transform=ax.transAxes))
    ax.text(.5,.966,"EDUPREDICT AI",fontsize=29,fontweight="bold",color="white",va="center",ha="center",transform=ax.transAxes)
    ax.text(.5,.935,"Predicción temprana del rendimiento escolar con una red neuronal MLP",
            fontsize=12.5,color="#DDEBFF",va="center",ha="center",transform=ax.transAxes)
    ax.text(.5,.918,f"Integrantes: {safe_text(team)}",fontsize=8.0,color="white",ha="center",va="bottom",transform=ax.transAxes)

    add_box(ax,.035,.785,.445,.105,"Objetivo general",
        "Desarrollar un pipeline reproducible que clasifique el rendimiento en Bajo, Medio o Alto usando variables socioeducativas; comparar un MLP con modelos de referencia mediante F1 macro y desplegar la inferencia en Streamlit.","#EAF2FF")
    add_box(ax,.52,.785,.445,.105,"Situación problema y pregunta orientadora",
        "Las dificultades académicas pueden detectarse cuando las notas finales ya están consolidadas. ¿Cómo aprovechar una red neuronal para identificar patrones complejos y apoyar una alerta temprana mediante una interfaz interactiva?","#FFF2E2")
    add_box(ax,.035,.655,.445,.105,"Dataset seleccionado",
        "Students Performance in Exams contiene variables socioeducativas y puntajes en matemáticas, lectura y escritura. Las notas construyen la etiqueta Bajo/Medio/Alto, pero se excluyen de las entradas para evitar data leakage.","#EAF7F0")
    add_box(ax,.52,.655,.445,.105,"Variables y preparación",
        "Género, grupo anonimizado, educación parental, almuerzo y curso de preparación. One-Hot Encoding y división estratificada: 70% entrenamiento, 15% validación y 15% prueba.","#EDF4FF")

    add_box(ax,.035,.515,.93,.115,"Flujo metodológico","","#FFF9E9")
    steps=["CSV","Limpieza","SQLite","EDA","One-Hot","Modelos","Evaluación","Streamlit"]
    x0,gap,box_w=.055,.016,.095
    for i,step in enumerate(steps):
        x=x0+i*(box_w+gap)
        ax.add_patch(FancyBboxPatch((x,.545),box_w,.038,boxstyle="round,pad=0.004,rounding_size=0.008",
                    linewidth=1,edgecolor="#C47F00",facecolor="#FFF3C4",transform=ax.transAxes))
        ax.text(x+box_w/2,.564,step,ha="center",va="center",fontsize=7.5,fontweight="bold",color="#664200",transform=ax.transAxes)
        if i<len(steps)-1:
            ax.annotate("",xy=(x+box_w+gap*.85,.564),xytext=(x+box_w+.002,.564),
                        arrowprops=dict(arrowstyle="->",color="#C47F00",lw=1.2),xycoords=ax.transAxes,textcoords=ax.transAxes)
    ax.text(.5,.523,"Etiqueta: promedio < 60 = Bajo | 60 a < 80 = Medio | ≥ 80 = Alto",fontsize=8,color="#5E4B00",ha="center",transform=ax.transAxes)

    add_box(ax,.035,.365,.445,.125,"Arquitectura empleada: MLP",
        "MLP seleccionado porque el dataset es tabular. Dense 128 + ReLU + L2 → BatchNorm → Dropout 0.30 → Dense 64 + ReLU + L2 → BatchNorm → Dropout 0.20 → Dense 32 + ReLU → Softmax(3).","#F1ECFF",8.0)
    add_box(ax,.52,.365,.445,.125,"Principales métricas y resultados",
        f"MLP desplegado — Accuracy: {accuracy:.3f}\nMLP desplegado — F1 macro: {f1:.3f}\nLoss MLP: {loss:.3f}\nMejor época MLP: {epoch}\nMejor modelo comparado: {best.get('modelo','N/D')}\nF1 del ganador: {float(best.get('f1_macro',0)):.3f}","#E9F7EF",8.4)

    if comparison:
        chart=fig.add_axes([.065,.238,.405,.105],facecolor="#F6F9FC")
        names=[str(r.get("modelo","Modelo")) for r in comparison]; values=[float(r.get("f1_macro",0)) for r in comparison]
        chart.barh(names,values,color=["#2E86DE","#7D3C98","#7F8C8D","#17A589"][:len(names)])
        chart.set_xlim(0,1); chart.set_xlabel("F1 macro",fontsize=8); chart.set_title("Comparación de modelos",fontsize=10,fontweight="bold")
        chart.tick_params(axis="both",labelsize=7); chart.grid(axis="x",alpha=.2)

    add_box(ax,.52,.215,.445,.125,"Vista previa de la aplicación Streamlit","","#EAF2FF")
    preview=fig.add_axes([.545,.222,.395,.066]); preview.axis("off")
    if app_preview is not None:
        try:
            image = Image.open(app_preview).convert("RGB")
            image.thumbnail((1600, 900))
            preview.imshow(image); preview.set_aspect("auto")
        except Exception:
            preview.text(.5,.5,"No fue posible leer la captura.",ha="center",va="center")
    else:
        preview.add_patch(plt.Rectangle((0,0),1,1,facecolor="white",edgecolor="#8AA8C7"))
        preview.add_patch(plt.Rectangle((0,.78),1,.22,facecolor="#12385B"))
        preview.text(.5,.89,"EduPredict AI",color="white",fontsize=10,fontweight="bold",va="center",ha="center")
        preview.text(.5,.62,"Predicción individual",fontsize=8,fontweight="bold",ha="center")
        preview.text(.5,.32,"Formulario socioeducativo  →  Nivel + probabilidades",fontsize=7,ha="center")
        preview.text(.5,.12,"Predicción | Métricas | Póster",fontsize=7,color="#415A77",ha="center")

    add_box(ax,.035,.075,.58,.125,"Conclusiones",
        f"{best.get('modelo','N/D')} obtuvo el mayor F1 macro en la comparación. La predicción interactiva usa el MLP guardado como best_mlp_model.keras. Es un prototipo académico; para uso real requiere validación externa, calibración, análisis de sesgos y supervisión humana.","#FFF0F0",8.0)
    add_box(ax,.645,.075,.32,.125,"Enlaces",f"Aplicación:\n{safe_text(app_url)}\n\nVideo:\n{safe_text(video_url)}","#EAF7F0",7.8)
    ax.text(.5,.027,"Póster generado dinámicamente desde Streamlit con models/metrics.json",ha="center",fontsize=8,color="#607D8B",transform=ax.transAxes)

    buffer=io.BytesIO(); fig.savefig(buffer,format="png",dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); buffer.seek(0); return buffer

    # Valores predeterminados del póster.
DEFAULT_TEAM = "Diego Fernando López Ledezma"
DEFAULT_APP_URL = "https://edupred-dfl.streamlit.app/"
DEFAULT_VIDEO_URL = "https://drive.google.com/file/d/11qQHwck-XHX78KtASFTIWsoSlZp6euRB/view?usp=sharing"

st.title("🎓 EduPredict AI")
st.caption("Sistema académico demostrativo de predicción temprana del rendimiento escolar")
tab1,tab2,tab3=st.tabs(["Predicción individual","Métricas","Póster tipo Infografía"])

with tab1:
    st.subheader("Predicción temprana del rendimiento")
    c1,c2=st.columns(2)
    with c1:
        gender=st.selectbox("Género",["female","male"])
        race=st.selectbox("Grupo",["group A","group B","group C","group D","group E"])
        parent=st.selectbox("Nivel educativo de los padres",["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
    with c2:
        lunch=st.selectbox("Tipo de almuerzo",["standard","free/reduced"])
        preparation=st.selectbox("Curso de preparación",["none","completed"])
    if st.button("Realizar predicción",type="primary"):
        pred,prob=predict_one({"gender":gender,"race_ethnicity":race,"parental_level_of_education":parent,"lunch":lunch,"test_preparation_course":preparation})
        st.success(f"Nivel predicho: {pred}"); st.metric("Confianza del modelo",f"{float(prob.max()):.1%}")
        st.bar_chart(pd.DataFrame({"Nivel":label_encoder.classes_,"Probabilidad":prob}).set_index("Nivel"))

with tab2:
    st.subheader("Evaluación del modelo")
    a,b,c=st.columns(3)
    a.metric("Accuracy",f"{float(metrics.get('test_accuracy',0)):.3f}")
    b.metric("F1 macro",f"{float(metrics.get('test_f1_macro',0)):.3f}")
    c.metric("Mejor época",str(metrics.get("best_epoch","N/D")))
    comparison_df=pd.DataFrame(metrics.get("comparison",[]))
    if not comparison_df.empty:
        st.dataframe(comparison_df,use_container_width=True)
        cols=[x for x in ["accuracy","f1_macro"] if x in comparison_df.columns]
        if cols: st.bar_chart(comparison_df.set_index("modelo")[cols])

with tab3:
    st.subheader("Póster sintético tipo Infografía")
    st.markdown("El póster integra **integrantes, objetivo, problema/pregunta, dataset, metodología, arquitectura, métricas/resultados, vista previa, conclusiones y enlaces**.")
    team=st.text_input("Nombres de integrantes", value=DEFAULT_TEAM,
                       placeholder="Nombre 1, Nombre 2, ...", max_chars=140)
    app_url=st.text_input("Enlace de la aplicación", value=DEFAULT_APP_URL,
                          placeholder="https://...streamlit.app", max_chars=250)
    video_url=st.text_input("Enlace del video", value=DEFAULT_VIDEO_URL,
                            placeholder="https://...", max_chars=250)
    app_preview=st.file_uploader("Captura de la aplicación para el póster (opcional)",type=["png","jpg","jpeg"])
    if app_preview is not None and app_preview.size > 5 * 1024 * 1024:
        st.error("La captura no debe superar 5 MB.")
        st.stop()
    poster=create_poster(team,app_url,video_url,app_preview)
    st.image(poster.getvalue(),caption="Vista previa del póster EduPredict AI",use_container_width=True)
    st.download_button("⬇️ Descargar póster en PNG",data=poster.getvalue(),file_name="Poster_EduPredict_AI.png",mime="image/png",type="primary")
