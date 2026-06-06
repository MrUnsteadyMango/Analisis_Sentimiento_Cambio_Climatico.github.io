import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cambio Climático", layout="wide")

st.title("🌍 Análisis de Sentimiento sobre el Cambio Climático")
st.subheader("Analisis_Sentimiento_Cambio_Climatico - UNMSM 2026")

# Cargar datos (sube tu archivo csv al repo)
try:
    df = pd.read_csv("datos_con_sentimiento.csv")
except:
    st.warning("No se encontró el archivo de datos")
    st.stop()

st.success(f"Datos cargados: {len(df)} publicaciones")

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Negativo", len(df[df['predicted_sentiment']=='Negative']))
col2.metric("Neutral", len(df[df['predicted_sentiment']=='Neutral']))
col3.metric("Positivo", len(df[df['predicted_sentiment']=='Positive']))

# Gráficos
tab1, tab2, tab3 = st.tabs(["📊 Distribución", "☁️ Nube de Palabras", "📅 Evolución"])

with tab1:
    fig = px.pie(df, names='predicted_sentiment', title="Distribución de Sentimientos")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    text = " ".join(df['clean_message'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab3:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        monthly = df.groupby(df['date'].dt.to_period('M'))['predicted_sentiment'].value_counts().unstack()
        st.line_chart(monthly)

st.dataframe(df[['message', 'predicted_sentiment']].head(50))
