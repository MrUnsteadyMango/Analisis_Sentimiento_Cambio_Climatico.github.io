import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cambio Climático", layout="wide")

st.title("🌍 Análisis de Sentimiento sobre el Cambio Climático")
st.subheader("Analisis_Sentimiento_Cambio_Climatico - UNMSM")

# Cargar datos
try:
    df = pd.read_csv("datos_con_sentimiento.csv")
    st.success(f"✅ {len(df):,} publicaciones analizadas")
except:
    st.error("No se encontró el archivo datos_con_sentimiento.csv")
    st.stop()

# Filtros
st.sidebar.header("Filtros")
sentiment_filter = st.sidebar.multiselect(
    "Sentimiento", 
    options=df['predicted_sentiment'].unique(),
    default=df['predicted_sentiment'].unique()
)

df_filtered = df[df['predicted_sentiment'].isin(sentiment_filter)]

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Negativo", len(df_filtered[df_filtered['predicted_sentiment']=='Negative']))
col2.metric("Neutral", len(df_filtered[df_filtered['predicted_sentiment']=='Neutral']))
col3.metric("Positivo", len(df_filtered[df_filtered['predicted_sentiment']=='Positive']))

# Gráficos
tab1, tab2, tab3 = st.tabs(["📊 Distribución", "☁️ Nube de Palabras", "📋 Datos"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df_filtered, names='predicted_sentiment', title="Distribución de Sentimientos")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_filtered, x='predicted_sentiment', color='predicted_sentiment', title="Cantidad por Sentimiento")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Nube de Palabras")
    text = " ".join(df_filtered['clean_message'].astype(str))
    if len(text) > 100:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No hay suficiente texto para generar la nube")

with tab3:
    st.dataframe(df_filtered[['message', 'predicted_sentiment', 'confidence']].head(50), use_container_width=True)

st.caption("Proyecto de Prácticas Pre Profesionales - Universidad Nacional Mayor de San Marcos")
