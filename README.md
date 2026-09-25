# Estados del mercado a la luz de la historia

> Detección de regímenes de mercado mediante aprendizaje no supervisado y su lectura histórica.

Trabajo Fin de Máster · Máster en Data Science, Big Data & Business Analytics (UCM).

**Aplicación web (demo en vivo):** https://tfm-estados-mercado.streamlit.app/

<!-- Recomendado para portfolio: añadir aquí una captura o GIF de la app en funcionamiento. Es lo primero que mira quien entra. -->

## Resumen

Los mercados financieros no se comportan de forma homogénea en el tiempo: alternan entre fases con dinámicas de riesgo y rentabilidad muy distintas (calma alcista, normalidad, corrección y crisis). Este proyecto detecta esas fases —o «regímenes»— de forma automática a partir de datos históricos, anticipa la entrada en periodos de estrés, analiza cómo se relacionan los distintos mercados entre sí, y contrasta los estados detectados con una cronología de acontecimientos históricos reales.

El hallazgo central: un modelo alimentado únicamente con el comportamiento de los precios, sin ninguna información histórica, reconoce de forma autónoma las grandes crisis económicas del periodo (la burbuja puntocom, 2008, la COVID-19).

## Objetivos

- Detectar de forma no supervisada los regímenes de los principales mercados de EE. UU., Europa, Asia y España.
- Anticipar la entrada en periodos de estrés del mercado mediante un modelo supervisado.
- Analizar las correlaciones dinámicas entre mercados y sus implicaciones para la diversificación.
- Interpretar los cambios de régimen a la luz de los acontecimientos históricos, con especial atención al caso español.
- Productivizar la solución en una aplicación web desplegada sobre datos actualizados.

## Datos

Series históricas diarias de doce activos financieros de acceso público, obtenidas a través de Yahoo Finance (yfinance): diez índices bursátiles —S&P 500 (EE. UU.), STOXX 600 (Europa), IBEX 35 (España), FTSE 100 (Reino Unido), Nikkei 225 (Japón), Shanghái (China), Nifty 50 (India), KOSPI (Corea del Sur), TAIEX (Taiwán) y Yakarta (Indonesia)— más dos activos refugio (oro y deuda pública estadounidense).

En total, más de 84.000 observaciones diarias, con hasta 35 años de profundidad histórica (desde 1990). El conjunto se congela con fecha de corte 31 de diciembre de 2024 para garantizar la reproducibilidad; la aplicación web, en cambio, opera sobre datos actualizados en tiempo real.

## Metodología

- **Detección de regímenes:** modelo oculto de Markov (HMM) sobre rendimiento y volatilidad, validado con clustering (K-Means) y detección de puntos de cambio (ruptures).
- **Predicción de estrés:** clasificador XGBoost con validación temporal, control del sesgo de anticipación y ponderación de clases.
- **Dinámica entre mercados:** correlaciones móviles para medir el contagio y el papel de los activos refugio.
- **Interpretación histórica:** contraste de los regímenes detectados con una cronología de acontecimientos reales.

## Tecnologías

Python · pandas · NumPy · scikit-learn · hmmlearn · XGBoost · Matplotlib · Streamlit · Snowflake

## Estructura del repositorio

- `notebooks/` — cuadernos numerados del análisis (ingesta y congelación de datos, detección de regímenes, predicción, overlays históricos, entrenamiento de modelos e integración con Snowflake).
- `src/` — módulo de funciones reutilizables (`tfm.py`).
- `app/` — aplicación web Streamlit (`app.py`).
- `modelos/` — modelo predictivo entrenado y serializado.
- `docs/` — figuras del proyecto.

## Cómo ejecutarlo

```bash
# 1. Clonar el repositorio
git clone https://github.com/[USUARIO]/tfm-estados-mercado.git
cd tfm-estados-mercado

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Lanzar la aplicación web
streamlit run app.py
```

Los notebooks del análisis pueden ejecutarse por separado desde la carpeta `notebooks/`. Los datos se descargan automáticamente de Yahoo Finance, por lo que el proyecto es plenamente reproducible a partir del código.

## Resultados

- **Detección de regímenes:** el modelo identifica cuatro regímenes coherentes (calma alcista, normalidad, corrección y crisis) que coinciden con notable precisión con las grandes crisis históricas, sin haber recibido ninguna información sobre ellas.
- **Predicción de estrés:** el clasificador alcanza un AUC de 0,85 sobre datos no vistos y detecta cerca de tres de cada cuatro periodos de estrés reales, funcionando como sistema de alerta temprana.
- **Contagio y diversificación:** la correlación media entre bolsas se dispara en las crisis (de ~0,15 en calma a ~0,68 durante la COVID-19), lo que revela que la diversificación geográfica falla precisamente cuando más se necesita.
- **Lectura histórica:** la comparación entre mercados reconstruye, solo a partir de los precios, la «década perdida» española tras 2008, frente a la rápida recuperación estadounidense.

## Líneas de trabajo futuro

- Extender el predictor de estrés a cada mercado con un modelo específico por índice.
- Profundizar en la interrelación de los mercados asiáticos.
- Incorporar el mercado inmobiliario español (con fuentes del INE y el Banco de España) para completar el retrato de la crisis más allá de las bolsas.

## Autor

Cristian Gay Martín

---

*Proyecto académico con fines formativos. No constituye asesoramiento de inversión.*
