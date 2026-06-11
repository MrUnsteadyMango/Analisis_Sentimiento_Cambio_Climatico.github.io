import streamlit as st
import pandas as pd
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

# LOAD DATA

# --------------------------------------------------

try:
df = pd.read_csv("datos_con_sentimiento.csv")

```
st.success(
    f"{len(df):,} records loaded successfully"
)
```

except Exception:

```
st.error(
    "File 'datos_con_sentimiento.csv' was not found."
)

st.stop()
```

# --------------------------------------------------

# SIDEBAR FILTERS

# --------------------------------------------------

st.sidebar.header("Filters")

if "predicted_sentiment" in df.columns:

```
category_filter = st.sidebar.multiselect(
    "Category",
    options=sorted(df["predicted_sentiment"].unique()),
    default=sorted(df["predicted_sentiment"].unique())
)

df_filtered = df[
    df["predicted_sentiment"].isin(category_filter)
]
```

else:

```
df_filtered = df.copy()
```

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

# TAB 1 - CHARTS

# --------------------------------------------------

with tab1:

```
col_a, col_b = st.columns(2)

with col_a:

    if "predicted_sentiment" in df_filtered.columns:

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

    if "predicted_sentiment" in df_filtered.columns:

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
```

# --------------------------------------------------

# TAB 2 - WORD CLOUD

# --------------------------------------------------

with tab2:

```
st.subheader("Word Cloud")

text_column = (
    "clean_message"
    if "clean_message" in df_filtered.columns
    else "message"
)

text = " ".join(
    df_filtered[text_column]
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

else:

    st.info(
        "Not enough text available."
    )
```

# --------------------------------------------------

# TAB 3 - MODEL EVALUATION

# --------------------------------------------------

with tab3:

```
st.subheader("Classification Report")

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
    **Model Used:** Logistic Regression

    **Training Dataset:** Twitter Climate Change Sentiment Dataset

    **Categories:**
    - Anti
    - Neutral
    - Pro
    - News

    The model achieved approximately 70% accuracy on unseen data.
    """
)
```

# --------------------------------------------------

# SAMPLE TWEETS

# --------------------------------------------------

st.subheader("Sample Publications")

columns_to_show = [
col for col in [
"message",
"predicted_sentiment",
"confidence"
]
if col in df_filtered.columns
]

st.dataframe(
df_filtered[columns_to_show].head(30),
use_container_width=True
)

# --------------------------------------------------

# FOOTER

# --------------------------------------------------

st.caption(
"Climate Change Publication Classification using Machine Learning - UNMSM"
)
