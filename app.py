"""
App de exploración de regímenes de mercado — TFM.
Explorador interactivo: el usuario elige un mercado y ve sus regímenes
detectados por el HMM, con los acontecimientos históricos anotados.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import tfm

st.set_page_config(page_title="Estados del mercado", layout="wide")

st.title("Estados del mercado a la luz de la historia")
st.markdown("Detección de regímenes de mercado mediante aprendizaje no supervisado (HMM), "
            "contrastados con los grandes acontecimientos históricos.")

# --- Universo de mercados: ticker y ámbito de eventos que le aplican ---
MERCADOS = {
    "S&P 500 (EE.UU.)":       {"ticker": "^GSPC",     "ambitos": ["global"]},
    "IBEX 35 (España)":       {"ticker": "^IBEX",     "ambitos": ["global", "europa"]},
    "STOXX 600 (Europa)":     {"ticker": "^STOXX",    "ambitos": ["global", "europa"]},
    "FTSE 100 (Reino Unido)": {"ticker": "^FTSE",     "ambitos": ["global", "europa"]},
    "Nikkei 225 (Japón)":     {"ticker": "^N225",     "ambitos": ["global"]},
    "Shanghái (China)":       {"ticker": "000001.SS", "ambitos": ["global"]},
    "Nifty 50 (India)":       {"ticker": "^NSEI",     "ambitos": ["global"]},
    "KOSPI (Corea del Sur)":  {"ticker": "^KS11",     "ambitos": ["global"]},
    "TAIEX (Taiwán)":         {"ticker": "^TWII",     "ambitos": ["global"]},
    "Jakarta (Indonesia)":    {"ticker": "^JKSE",     "ambitos": ["global"]},
}

COLORES = {"Calma alcista": "#2ca02c", "Normal": "#ffd92f",
           "Corrección": "#ff7f0e", "Crisis": "#d62728"}

# --- Cronología de acontecimientos históricos ---
EVENTOS = [
    {"nombre": "Estallido puntocom",        "tipo": "banda", "inicio": "2000-03-01", "fin": "2002-10-01", "ambito": "global"},
    {"nombre": "11-S",                       "tipo": "linea", "inicio": "2001-09-11", "fin": None,         "ambito": "global"},
    {"nombre": "Crisis financiera global",   "tipo": "banda", "inicio": "2007-08-01", "fin": "2009-06-01", "ambito": "global"},
    {"nombre": "COVID-19",                   "tipo": "linea", "inicio": "2020-03-01", "fin": None,         "ambito": "global"},
    {"nombre": "Inflación + guerra Ucrania", "tipo": "banda", "inicio": "2022-01-01", "fin": "2022-10-01", "ambito": "global"},
    {"nombre": "Crisis rusa / LTCM",         "tipo": "banda", "inicio": "1998-08-01", "fin": "1998-10-31", "ambito": "global"},
    {"nombre": "Crisis China / petróleo",    "tipo": "banda", "inicio": "2015-08-01", "fin": "2016-02-29", "ambito": "global"},
    {"nombre": "Crisis deuda europea",       "tipo": "banda", "inicio": "2010-05-01", "fin": "2012-09-01", "ambito": "europa"},
    {"nombre": "Rescate bancario español",   "tipo": "linea", "inicio": "2012-06-01", "fin": None,         "ambito": "europa"},
    {"nombre": "Brexit (referéndum)",        "tipo": "linea", "inicio": "2016-06-23", "fin": None,         "ambito": "europa"},
]
CRONO = pd.DataFrame(EVENTOS)
CRONO["inicio"] = pd.to_datetime(CRONO["inicio"])
CRONO["fin"] = pd.to_datetime(CRONO["fin"])

# --- Menú lateral ---
st.sidebar.header("Configuración")
nombre_mercado = st.sidebar.selectbox("Elige un mercado", list(MERCADOS.keys()))
mostrar_eventos = st.sidebar.checkbox("Mostrar acontecimientos históricos", value=True)

config = MERCADOS[nombre_mercado]
ticker = config["ticker"]

@st.cache_data(show_spinner="Descargando datos y detectando regímenes...")
def analizar(ticker):
    df = tfm.cargar_mercado(ticker)
    datos, _ = tfm.detectar_regimenes(df)
    datos["Close"] = df["Close"]
    return datos

datos = analizar(ticker)

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(datos.index, datos["Close"], color="black", linewidth=0.8, zorder=4)
for regimen, color in COLORES.items():
    mask = datos["regimen"] == regimen
    ax.fill_between(datos.index, 0, datos["Close"].max(),
                    where=mask, color=color, alpha=0.25, zorder=1)

# --- Eventos históricos (filtrados por ámbito del mercado y por rango de fechas) ---
if mostrar_eventos:
    inicio_datos, fin_datos = datos.index.min(), datos.index.max()
    # Fin efectivo del evento: su fin, o su inicio si es puntual (línea)
    fin_efectivo = CRONO["fin"].fillna(CRONO["inicio"])
    # Un evento se muestra si SE SOLAPA con el rango de datos (no solo si empieza dentro)
    eventos_mostrar = CRONO[
        (CRONO["ambito"].isin(config["ambitos"])) &
        (fin_efectivo >= inicio_datos) &        # el evento termina después de que empiecen los datos
        (CRONO["inicio"] <= fin_datos)          # y empieza antes de que terminen
    ]
    caja = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    ymax = datos["Close"].max()
    for _, ev in eventos_mostrar.iterrows():
        if ev["tipo"] == "linea":
            ax.axvline(ev["inicio"], color="black", linestyle="--", linewidth=1.1, alpha=0.9, zorder=5)
            ax.text(ev["inicio"], ymax*0.96, "  " + ev["nombre"], rotation=90,
                    fontsize=7, va="top", ha="left", color="black", zorder=6, bbox=caja)
        else:
            fin_ev = ev["fin"] if pd.notna(ev["fin"]) else ev["inicio"]
            # Recortamos la banda al rango de datos visible
            ini_banda = max(ev["inicio"], inicio_datos)
            fin_banda = min(fin_ev, fin_datos)
            ax.axvspan(ini_banda, fin_banda, color="dimgray", alpha=0.18, zorder=2)
            ax.axvline(ini_banda, color="dimgray", linestyle=":", linewidth=1, alpha=0.7, zorder=3)
            ax.axvline(fin_banda, color="dimgray", linestyle=":", linewidth=1, alpha=0.7, zorder=3)
            centro = ini_banda + (fin_banda - ini_banda) / 2
            ax.text(centro, ymax*0.96, ev["nombre"], rotation=0, ha="center", va="top",
                    fontsize=7, color="black", zorder=6, bbox=caja)

ax.set_yscale("log")
ax.set_ylim(datos["Close"].min()*0.9, datos["Close"].max()*1.1)
ax.set_xlim(datos.index.min(), datos.index.max())
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title(f"{nombre_mercado}: regímenes detectados por el HMM")
ax.set_ylabel("Precio de cierre (escala log)")
ax.legend(handles=[Patch(facecolor=c, alpha=0.5, label=r) for r, c in COLORES.items()],
          loc="upper left", ncol=4)
st.pyplot(fig)

# --- Tabla de distribución (ordenada por gravedad) ---
st.subheader("Distribución de regímenes")
orden_gravedad = ["Calma alcista", "Normal", "Corrección", "Crisis"]
resumen = datos.groupby("regimen").agg(
    dias=("Close", "size"), vol_media=("vol_21d", "mean")
).reindex(orden_gravedad).round(4)
resumen["% del tiempo"] = (100 * resumen["dias"] / resumen["dias"].sum()).round(1)
st.dataframe(resumen)

st.caption("TFM · Detección de regímenes con HMM · Datos: Yahoo Finance")
