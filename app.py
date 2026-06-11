import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Climate Change Classification",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Climate Change Publications Classification")
st.subheader("Machine Learning Analysis of Climate Change Discussions")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("modelo_climatico.pkl")

    vectorizer = joblib.load("vectorizador.pkl")

    return model, vectorizer

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'#\w+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:

    df = pd.read_csv("twitter_sentiment_data.csv")

    st.success(
        f"{len(df):,} records loaded successfully"
    )

except Exception:

    st.error(
        "twitter_sentiment_data.csv was not found."
    )

    st.stop()

# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

try:

    model, vectorizer = load_model()

except Exception:

    st.error(
        "Model files were not found."
    )

    st.stop()

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

with st.spinner("Generating predictions..."):

    df["clean_message"] = (
        df["message"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    X = vectorizer.transform(
        df["clean_message"]
    )

    predictions = model.predict(X)

    sentiment_map = {
        -1: "Anti",
         0: "Neutral",
         1: "Pro",
         2: "News"
    }

    df["predicted_sentiment"] = [
        sentiment_map[x]
        for x in predictions
    ]

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Category",
    options=sorted(
        df["predicted_sentiment"].unique()
    ),
    default=sorted(
        df["predicted_sentiment"].unique()
    )
)

df_filtered = df[
    df["predicted_sentiment"].isin(
        category_filter
    )
]

# --------------------------------------------------
# DATASET METRICS
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Records",
    len(df_filtered)
)

col2.metric(
    "Anti",
    len(
        df_filtered[
            df_filtered["predicted_sentiment"] == "Anti"
        ]
    )
)

col3.metric(
    "Neutral",
    len(
        df_filtered[
            df_filtered["predicted_sentiment"] == "Neutral"
        ]
    )
)

col4.metric(
    "Pro",
    len(
        df_filtered[
            df_filtered["predicted_sentiment"] == "Pro"
        ]
    )
)

col5.metric(
    "News",
    len(
        df_filtered[
            df_filtered["predicted_sentiment"] == "News"
        ]
    )
)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("Model Performance")

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        0.6976,
        0.6923,
        0.6976,
        0.6810
    ]
})

st.dataframe(
    metrics_df,
    use_container_width=True
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Charts",
    "Word Cloud",
    "Model Evaluation"
])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------

with tab1:

    col_a, col_b = st.columns(2)

    with col_a:

        fig_pie = px.pie(
            df_filtered,
            names="predicted_sentiment",
            title="Distribution of Categories"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col_b:

        fig_bar = px.histogram(
            df_filtered,
            x="predicted_sentiment",
            color="predicted_sentiment",
            title="Frequency by Category"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

# --------------------------------------------------
# TAB 2
# --------------------------------------------------

with tab2:

    st.subheader("Word Cloud")

    text = " ".join(
        df_filtered["clean_message"]
        .fillna("")
        .astype(str)
    )

    if len(text.strip()) > 100:

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            max_words=200
        ).generate(text)

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig)

# --------------------------------------------------
# TAB 3
# --------------------------------------------------

with tab3:

    st.subheader(
        "Classification Report"
    )

    report = pd.DataFrame({

        "Class": [
            "Anti",
            "Neutral",
            "Pro",
            "News"
        ],

        "Precision": [
            0.73,
            0.60,
            0.71,
            0.72
        ],

        "Recall": [
            0.37,
            0.39,
            0.87,
            0.66
        ],

        "F1 Score": [
            0.49,
            0.47,
            0.78,
            0.69
        ]
    })

    st.dataframe(
        report,
        use_container_width=True
    )

    st.markdown(
        """
        **Model:** Logistic Regression

        **Vectorization:** TF-IDF

        **Training Dataset:** Twitter Climate Change Sentiment Dataset

        **Categories**

        - Anti
        - Neutral
        - Pro
        - News

        The model achieved approximately 70% accuracy on unseen data.
        """
    )

# --------------------------------------------------
# SAMPLE PUBLICATIONS
# --------------------------------------------------

st.subheader("Sample Publications")

st.dataframe(
    df_filtered[
        [
            "message",
            "predicted_sentiment"
        ]
    ].head(30),
    use_container_width=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Climate Change Publication Classification using Machine Learning - UNMSM"
)
