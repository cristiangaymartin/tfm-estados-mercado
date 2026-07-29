# Estados del mercado a la luz de la historia

> Detección de fases y tendencias en los mercados globales mediante aprendizaje no supervisado.

Trabajo Fin de Máster · Máster en Data Science, Big Data & Business Analytics (UCM).

<!-- Cuando tengas la demo desplegada, añade aquí el enlace y una captura o GIF. Es lo primero que mira quien entra. -->

## Resumen

Los mercados financieros no se comportan de forma homogénea en el tiempo: alternan entre fases con dinámicas de riesgo y rentabilidad muy distintas (calma alcista, estrés bajista, alta volatilidad). Este proyecto detecta esas fases de forma automática a partir de datos históricos, clasifica la tendencia esperada, analiza cómo se relacionan los distintos mercados entre sí, y contrasta los estados detectados con una cronología de acontecimientos históricos reales.

## Objetivos

- Detectar de forma no supervisada los estados de los principales mercados de EE. UU., Europa, Asia y España.
- Clasificar la tendencia esperada mediante un modelo supervisado.
- Analizar las correlaciones dinámicas entre mercados y sus implicaciones para la diversificación.
- Interpretar los cambios de estado a la luz de los acontecimientos históricos.
- Productivizar el modelo en una aplicación web desplegada.

## Datos

Series históricas diarias de índices y ETFs de acceso público (yfinance / Stooq): S&P 500, STOXX 600, IBEX 35, Nikkei 225, CSI 300, Nifty 50 y otros mercados asiáticos, más activos refugio. Módulo inmobiliario español con datos del INE y del Banco de España.

<!-- A medida que confirmes las fuentes, detalla aquí los tickers y el rango temporal. -->

## Metodología

- **Detección de estados:** modelo oculto de Markov (HMM), con detección de puntos de cambio y clustering como validación cruzada.
- **Clasificación de tendencia:** XGBoost con validación temporal.
- **Dinámica entre mercados:** correlaciones móviles.
- **Interpretación histórica:** contraste de los estados detectados con eventos reales.

## Tecnologías

Python · Snowflake · scikit-learn · hmmlearn · XGBoost · Streamlit

## Estructura del repositorio

- `notebooks/` — cuadernos numerados del análisis (ingesta, EDA, modelado).
- `src/` — código fuente reutilizable.
- `app/` — aplicación Streamlit.
- `docs/` — memoria, capturas y diagramas.

## Cómo ejecutarlo

<!-- Pendiente: instrucciones de instalación y ejecución. Se completa en la Fase 1, cuando exista requirements.txt. -->

## Resultados

<!-- Pendiente: principales hallazgos, con figuras. Se completa según avanza el proyecto. -->

## Autor

Cristian Gay Martín

---

*Proyecto académico con fines formativos. No constituye asesoramiento de inversión.*
