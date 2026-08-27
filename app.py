"""
App de exploración de regímenes de mercado — TFM.
Explorador interactivo: el usuario elige un mercado y ve sus regímenes
detectados por el HMM sobre la serie histórica.
"""
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tfm  

# --- Configuración de la página ---
st.set_page_config(page_title="Estados del mercado", layout="wide")

st.title("Estados del mercado a la luz de la historia")
st.markdown("Detección de regímenes de mercado mediante aprendizaje no supervisado (HMM).")

# --- Universo de mercados disponibles ---
TICKERS = {
    "S&P 500 (EE.UU.)":       "^GSPC",
    "IBEX 35 (España)":       "^IBEX",
    "STOXX 600 (Europa)":     "^STOXX",
    "FTSE 100 (Reino Unido)": "^FTSE",
    "Nikkei 225 (Japón)":     "^N225",
    "Shanghái (China)":       "000001.SS",
    "Nifty 50 (India)":       "^NSEI",
    "KOSPI (Corea del Sur)":  "^KS11",
    "TAIEX (Taiwán)":         "^TWII",
    "Jakarta (Indonesia)":    "^JKSE",
}

COLORES = {"Calma alcista": "#2ca02c", "Normal": "#ffd92f",
           "Corrección": "#ff7f0e", "Crisis": "#d62728"}

# --- Menú lateral: el usuario elige el mercado ---
st.sidebar.header("Configuración")
nombre_mercado = st.sidebar.selectbox("Elige un mercado", list(TICKERS.keys()))
ticker = TICKERS[nombre_mercado]

# --- Descarga y análisis con caché ---
@st.cache_data(show_spinner="Descargando datos y detectando regímenes...")
def analizar(ticker):
    df = tfm.cargar_mercado(ticker)
    datos, _ = tfm.detectar_regimenes(df)
    datos["Close"] = df["Close"]
    return datos

datos = analizar(ticker)

# --- Gráfico de regímenes ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(datos.index, datos["Close"], color="black", linewidth=0.8, zorder=3)
for regimen, color in COLORES.items():
    mask = datos["regimen"] == regimen
    ax.fill_between(datos.index, 0, datos["Close"].max(),
                    where=mask, color=color, alpha=0.25, zorder=1)
ax.set_yscale("log")
ax.set_ylim(datos["Close"].min()*0.9, datos["Close"].max()*1.1)
ax.set_title(f"{nombre_mercado}: regímenes detectados por el HMM")
ax.set_ylabel("Precio de cierre (escala log)")
ax.legend(handles=[Patch(facecolor=c, alpha=0.5, label=r) for r, c in COLORES.items()],
          loc="upper left", ncol=4)
st.pyplot(fig)

# --- Resumen de los regímenes ---
st.subheader("Distribución de regímenes")
resumen = datos.groupby("regimen").agg(
    dias=("Close", "size"),
    vol_media=("vol_21d", "mean")
).round(4)
resumen["% del tiempo"] = (100 * resumen["dias"] / resumen["dias"].sum()).round(1)
st.dataframe(resumen)

st.caption("TFM · Detección de regímenes con HMM · Datos: Yahoo Finance")
