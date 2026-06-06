import pandas as pd
import re
from tqdm import tqdm
from transformers import pipeline

ruta = "/content/drive/MyDrive/Analisis_Sentimiento"

# Cargar solo 10,000 tweets
df = pd.read_csv(f"{ruta}/twitter_sentiment_data.csv")
df = df.sample(n=10000, random_state=42).reset_index(drop=True)
print(f"✅ Procesando 10,000 tweets (de {len(pd.read_csv(f'{ruta}/twitter_sentiment_data.csv'))} totales)")

# Limpieza
def limpiar_texto(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("🧹 Limpiando texto...")
tqdm.pandas()
df['clean_message'] = df['message'].progress_apply(limpiar_texto)

# Análisis
print("🤖 Analizando sentimiento...")
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analizar(texto):
    try:
        result = sentiment_pipeline(texto[:512])[0]
        return result['label'].capitalize(), round(result['score'], 4)
    except:
        return "Neutral", 0.5

tqdm.pandas()
df[['predicted_sentiment', 'confidence']] = df['clean_message'].progress_apply(
    lambda x: pd.Series(analizar(x))
)

print("\n✅ ¡Listo!")
print(df['predicted_sentiment'].value_counts())

# Guardar
df.to_csv(f"{ruta}/datos_con_sentimiento.csv", index=False)
print("💾 Guardado en: Analisis_Sentimiento/datos_con_sentimiento.csv")
