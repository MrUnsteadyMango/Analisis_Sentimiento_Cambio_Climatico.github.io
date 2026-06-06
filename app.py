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
    st.success(f"✅ {len(df):,} publicaciones cargadas")
except:
    st.error("No se encontró el archivo datos_con_sentimiento.csv")
    st.stop()

# Filtros
st.sidebar.header("🔎 Filtros")
if 'predicted_sentiment' in df.columns:
    sentiment_filter = st.sidebar.multiselect(
        "Sentimiento", 
        options=df['predicted_sentiment'].unique(),
        default=df['predicted_sentiment'].unique()
    )
    df_filtered = df[df['predicted_sentiment'].isin(sentiment_filter)]
else:
    df_filtered = df

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Total", len(df_filtered))
col2.metric("Negativo", len(df_filtered[df_filtered.get('predicted_sentiment') == 'Negative']))
col3.metric("Positivo", len(df_filtered[df_filtered.get('predicted_sentiment') == 'Positive']))

# Gráficos
tab1, tab2 = st.tabs(["📊 Gráficos", "☁️ Nube de Palabras"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        if 'predicted_sentiment' in df_filtered.columns:
            fig = px.pie(df_filtered, names='predicted_sentiment', title="Distribución de Sentimientos")
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if 'predicted_sentiment' in df_filtered.columns:
            fig2 = px.histogram(df_filtered, x='predicted_sentiment', color='predicted_sentiment')
            st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Nube de Palabras")
    # Corrección aquí
    text_column = 'clean_message' if 'clean_message' in df_filtered.columns else 'message'
    text = " ".join(df_filtered[text_column].fillna("").astype(str))
    
    if len(text.strip()) > 100:
        wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No hay suficiente texto para generar la nube")

st.subheader("📋 Ejemplos de Tweets")
st.dataframe(df_filtered[['message', 'predicted_sentiment', 'confidence']].head(30), use_container_width=True)

st.caption("Proyecto Prácticas Pre Profesionales - UNMSM")
