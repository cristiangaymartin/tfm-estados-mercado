"""
Módulo de funciones del TFM 'Estados del mercado a la luz de la historia'.
Contiene las funciones reutilizables de carga de datos, detección de regímenes
y visualización de overlays históricos.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

def cargar_mercado(ticker, start="1990-01-01"):
    """
    Descarga un índice de Yahoo Finance y devuelve una tabla limpia
    con precio de cierre, log-rendimiento y volatilidad móvil de 21 días.
    """
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Close"]].copy()
    df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1))
    df["vol_21d"] = df["ret_log"].rolling(window=21).std()
    return df


def detectar_regimenes(df, n_estados=4, random_state=42):
    """
    Detecta regímenes de mercado con un HMM gaussiano sobre rendimiento y
    volatilidad. Devuelve (datos_con_regimenes, modelo). Los nombres de régimen
    se asignan por orden de volatilidad, de forma robusta a la etiqueta numérica.
    """
    datos = df[["ret_log", "vol_21d"]].dropna().copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(datos)
    modelo = hmm.GaussianHMM(n_components=n_estados, covariance_type="full",
                             n_iter=1000, random_state=random_state)
    modelo.fit(X)
    datos["estado"] = modelo.predict(X)
    nombres = ["Calma alcista", "Normal", "Corrección", "Crisis"][:n_estados]
    vol_por_estado = datos.groupby("estado")["vol_21d"].mean().sort_values()
    mapeo = {est: nom for est, nom in zip(vol_por_estado.index, nombres)}
    datos["regimen"] = datos["estado"].map(mapeo)
    return datos, modelo


def dibujar_overlay(datos, eventos, titulo, colores=None, guardar_en=None):
    """
    Dibuja un overlay de regímenes con acontecimientos históricos anotados.
    'datos' debe tener columnas 'Close' y 'regimen'. 'eventos' es un DataFrame
    con columnas nombre, tipo ('linea'/'banda'), inicio, fin.
    """
    
    if colores is None:
        colores = {"Calma alcista": "#2ca02c", "Normal": "#ffd92f",
                   "Corrección": "#ff7f0e", "Crisis": "#d62728"}

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(datos.index, datos["Close"], color="black", linewidth=0.8, zorder=4)
    for regimen, color in colores.items():
        mask = datos["regimen"] == regimen
        ax.fill_between(datos.index, 0, datos["Close"].max(),
                        where=mask, color=color, alpha=0.25, zorder=1)

    caja = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    ymax = datos["Close"].max()
    for _, ev in eventos.iterrows():
        if ev["tipo"] == "linea":
            ax.axvline(ev["inicio"], color="black", linestyle="--", linewidth=1.2, alpha=0.9, zorder=5)
            ax.text(ev["inicio"], ymax*0.97, "  " + ev["nombre"], rotation=90,
                    fontsize=8, va="top", ha="left", color="black", zorder=6, bbox=caja)
        else:
            ax.axvspan(ev["inicio"], ev["fin"], color="dimgray", alpha=0.18, zorder=2)
            ax.axvline(ev["inicio"], color="dimgray", linestyle=":", linewidth=1, alpha=0.7, zorder=3)
            ax.axvline(ev["fin"], color="dimgray", linestyle=":", linewidth=1, alpha=0.7, zorder=3)
            centro = ev["inicio"] + (ev["fin"] - ev["inicio"]) / 2
            ax.text(centro, ymax*0.97, ev["nombre"], rotation=0, ha="center", va="top",
                    fontsize=8, color="black", zorder=6, bbox=caja)

    ax.set_yscale("log")
    ax.set_ylim(datos["Close"].min()*0.9, datos["Close"].max()*1.1)
    ax.set_xlim(datos.index.min(), datos.index.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(titulo, fontsize=13, pad=12)
    ax.set_ylabel("Precio de cierre (escala log)")
    ax.legend(handles=[Patch(facecolor=c, alpha=0.5, label=r) for r, c in colores.items()],
              loc="upper left", ncol=4)
    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
