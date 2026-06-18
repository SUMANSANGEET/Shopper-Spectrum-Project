"""
╔══════════════════════════════════════════════════════════╗
║     🛒 SHOPPER SPECTRUM — Streamlit Deployment App       ║ 
║     Customer Segmentation & Product Recommendation       ║
║     Author : P Suman Sangeet | INNOVEXIS Data Science    ║
║                                     & Gen AI             ║
╚══════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ───────────────────────────────────────
import warnings
import re
import string
import datetime as dt
import io

# ─── Third-Party ────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL STYLING
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] * { color: #e8e0ff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f64f59 100%);
    border-radius: 16px;
    padding: 2.4rem 2rem;
    margin-bottom: 1.6rem;
    color: #fff;
    text-align: center;
}
.hero-banner h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-banner p  { font-size: 1.05rem; opacity: 0.88; margin: 0.5rem 0 0; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #1e1b4b;
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    color: #fff;
}
.metric-card .label  { font-size: 0.78rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value  { font-size: 1.9rem; font-weight: 700; color: #a5b4fc; font-family: 'JetBrains Mono', monospace; }
.metric-card .delta  { font-size: 0.82rem; color: #6ee7b7; margin-top: 2px; }

/* ── Section headers ── */
.section-header {
    border-left: 4px solid #7c3aed;
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem;
    font-size: 1.25rem;
    font-weight: 600;
    color: #312e81;
}

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.93rem;
    color: #3b0764;
}

/* ── Segment badge ── */
.segment-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.35rem 0.95rem;
    font-weight: 600;
    font-size: 1rem;
    margin: 0.4rem 0;
}

/* ── Recommendation card ── */
.rec-card {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #14532d;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #7c3aed !important;
    border-radius: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1e1b4b; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }

/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid #ddd6fe; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HELPERS — Text cleaning
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b\w*\d\w*\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()
    df["TotalPrice"]   = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"]  = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["InvoiceHour"]  = df["InvoiceDate"].dt.hour
    df["Weekday"]      = df["InvoiceDate"].dt.day_name()
    df["CustomerID"]   = df["CustomerID"].astype(str)
    df["Clean_Description"] = df["Description"].apply(clean_text)
    return df


# ════════════════════════════════════════════════════════════
#  RFM + CLUSTERING
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_rfm(df: pd.DataFrame):
    latest   = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency  =("InvoiceDate",  lambda x: (latest - x.max()).days),
        Frequency=("InvoiceNo",    "nunique"),
        Monetary =("TotalPrice",   "sum"),
    ).reset_index()
    rfm["Recency_log"]   = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"]  = np.log1p(rfm["Monetary"])
    return rfm


@st.cache_data(show_spinner=False)
def fit_kmeans(rfm: pd.DataFrame, k: int):
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency_log","Frequency_log","Monetary_log"]])
    model      = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm        = rfm.copy()
    rfm["Cluster"] = model.fit_predict(rfm_scaled)

    # Auto-label clusters
    summary = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    labels  = {}
    for c in summary.index:
        r, f, m = summary.loc[c, ["Recency","Frequency","Monetary"]]
        if f > 10 and m > 5000:
            labels[c] = ("💎 High-Value",  "#7c3aed")
        elif f > 5 and r < 100:
            labels[c] = ("🔁 Regular",     "#0891b2")
        elif r > 150:
            labels[c] = ("⚠️ At-Risk",      "#dc2626")
        else:
            labels[c] = ("🌱 Occasional",  "#16a34a")
    rfm["Segment"] = rfm["Cluster"].map(lambda c: labels[c][0])
    rfm["SegColor"]= rfm["Cluster"].map(lambda c: labels[c][1])
    sil = silhouette_score(rfm_scaled, rfm["Cluster"])
    return rfm, summary, sil, scaler, model, labels


@st.cache_data(show_spinner=False)
def elbow_data(rfm: pd.DataFrame):
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency_log","Frequency_log","Monetary_log"]])
    inertias, sils = [], []
    for k in range(2, 11):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(rfm_scaled, lbl))
    return list(range(2, 11)), inertias, sils


# ════════════════════════════════════════════════════════════
#  RECOMMENDATION MODELS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_content_recommender(df: pd.DataFrame):
    prod = (
        df[["Description","Clean_Description"]]
        .drop_duplicates("Description")
        .dropna()
        .reset_index(drop=True)
    )
    prod = prod[prod["Clean_Description"].str.strip() != ""]
    vec  = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(prod["Clean_Description"])
    sim   = cosine_similarity(tfidf)
    sim_df= pd.DataFrame(sim, index=prod["Description"], columns=prod["Description"])
    return sim_df, prod


@st.cache_data(show_spinner=False)
def build_collab_recommender(df: pd.DataFrame):
    user_item = df.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity",
        aggfunc="sum", fill_value=0
    )
    user_sim  = cosine_similarity(user_item)
    user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)
    return user_item, user_sim_df


def content_recommend(sim_df: pd.DataFrame, product_name: str, n: int = 5):
    if product_name not in sim_df.index:
        return []
    scores = sim_df[product_name].sort_values(ascending=False).iloc[1:n+1]
    return list(scores.index)


def collab_recommend(user_item, user_sim_df, cid: str, n: int = 5):
    if cid not in user_sim_df.index:
        return pd.DataFrame()
    similar = user_sim_df[cid].sort_values(ascending=False).iloc[1:11]
    recs = user_item.loc[similar.index].sum().sort_values(ascending=False)
    owned = user_item.loc[cid]
    recs = recs.drop(index=owned[owned > 0].index, errors="ignore").head(n)
    return recs.reset_index().rename(columns={"StockCode": "StockCode", 0: "Score"})


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
      <span style='font-size:2.6rem;'>🛒</span><br>
      <span style='font-size:1.25rem; font-weight:700;'>Shopper Spectrum</span><br>
      <span style='font-size:0.78rem; opacity:0.65;'>INNOVEXIS · Data Science and Gen AI</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload online_retail.csv",
        type=["csv"],
        help="UK Online Retail dataset (UCI / Kaggle).",
    )

    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 EDA & Insights",
         "🎯 Customer Segments",
         "🤖 Recommendations",
         "📈 Hypothesis Tests"],
    )

    if uploaded_file:
        st.divider()
        n_clusters = st.slider("KMeans Clusters (k)", 2, 8, 4)
        n_recs     = st.slider("Recommendations to show", 3, 10, 5)

    st.divider()
    st.caption("P Suman Sangeet · Cohort 2025\nFintech Domain · INNOVEXIS")


# ════════════════════════════════════════════════════════════
#  GATE — require upload
# ════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero-banner">
      <h1>🛒 Shopper Spectrum</h1>
      <p>Customer Segmentation &amp; Product Recommendation Engine</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🎯 RFM Clustering**\nSegment customers into High-Value, Regular, At-Risk & Occasional groups using KMeans.")
    with col2:
        st.info("**🤝 Collaborative Filtering**\nPersonalised product picks based on what similar shoppers bought.")
    with col3:
        st.info("**📝 Content-Based Recs**\nTF-IDF + cosine similarity on product descriptions for item-to-item suggestions.")

    st.markdown("""
    <div class="insight-box">
    👈 <strong>Upload your <code>online_retail.csv</code> in the sidebar to begin.</strong><br>
    Dataset source: <em>UCI ML Repository – Online Retail</em> (or Kaggle mirror).
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════
#  LOAD DATA  (cached)
# ════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading & cleaning data…"):
    df = load_and_clean(uploaded_file)

with st.spinner("⚙️ Building RFM & clustering…"):
    rfm_raw  = build_rfm(df)
    n_clusters_val = n_clusters if "n_clusters" in dir() else 4
    rfm, cluster_summary, sil_score, scaler, kmeans_model, seg_labels = fit_kmeans(rfm_raw, n_clusters_val)

with st.spinner("📦 Building recommendation engines…"):
    sim_df, prod_df  = build_content_recommender(df)
    user_item, user_sim_df = build_collab_recommender(df)


# ════════════════════════════════════════════════════════════
#  HERO (always visible)
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <h1>🛒 Shopper Spectrum</h1>
  <p>Customer Segmentation &amp; Product Recommendation · P Suman Sangeet</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # KPI cards
    n_customers = df["CustomerID"].nunique()
    n_products  = df["Description"].nunique()
    total_rev   = df["TotalPrice"].sum()
    n_invoices  = df["InvoiceNo"].nunique()
    n_countries = df["Country"].nunique()

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    kpi_html = ""
    kpis = [
        ("Customers",    f"{n_customers:,}",     "+unique"),
        ("Products",     f"{n_products:,}",       "+SKUs"),
        ("Invoices",     f"{n_invoices:,}",       "+orders"),
        ("Revenue",      f"£{total_rev/1e6:.2f}M","total"),
        ("Countries",    f"{n_countries}",         "+markets"),
        ("Silhouette",   f"{sil_score:.3f}",       "cluster quality"),
    ]
    for label, value, delta in kpis:
        kpi_html += f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="delta">{delta}</div>
        </div>"""
    st.markdown(f'<div class="metric-row">{kpi_html}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(300), use_container_width=True, height=320)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Missing Values (after clean)</p>', unsafe_allow_html=True)
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column","Missing"]
        fig = px.bar(mv, x="Missing", y="Column", orientation="h",
                     color="Missing", color_continuous_scale="Purpor",
                     template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Data Type Distribution</p>', unsafe_allow_html=True)
        dtypes = df.dtypes.astype(str).value_counts().reset_index()
        dtypes.columns = ["dtype","count"]
        fig = px.pie(dtypes, values="count", names="dtype",
                     color_discrete_sequence=px.colors.sequential.Purp,
                     template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Segment Distribution</p>', unsafe_allow_html=True)
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment","Count"]
    fig = px.bar(seg_counts, x="Segment", y="Count",
                 color="Segment",
                 color_discrete_sequence=["#7c3aed","#0891b2","#dc2626","#16a34a"],
                 template="plotly_dark")
    fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    tabs = st.tabs(["🌍 Geography","📦 Products","📈 Time Trends","💰 Spend Patterns","📊 RFM Distributions"])

    # ── Geography
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Top 10 Countries — Transaction Volume</p>', unsafe_allow_html=True)
            top_c = df["Country"].value_counts().head(10).reset_index()
            top_c.columns = ["Country","Transactions"]
            fig = px.bar(top_c, x="Transactions", y="Country", orientation="h",
                         color="Transactions", color_continuous_scale="Purp",
                         template="plotly_dark")
            fig.update_layout(height=360, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="insight-box">💡 <b>UK dominates</b> with ~85% of transactions. Netherlands, Germany, and France are secondary markets with growth potential.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-header">Avg Spend per Customer by Country</p>', unsafe_allow_html=True)
            ctry_m = df.groupby(["Country","CustomerID"])["TotalPrice"].sum().reset_index()
            avg_c  = ctry_m.groupby("Country")["TotalPrice"].mean().sort_values(ascending=False).head(10).reset_index()
            avg_c.columns = ["Country","AvgSpend"]
            fig = px.bar(avg_c, x="AvgSpend", y="Country", orientation="h",
                         color="AvgSpend", color_continuous_scale="Teal",
                         template="plotly_dark")
            fig.update_layout(height=360, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="insight-box">💡 <b>Australia & Netherlands</b> customers spend significantly more per visit — ideal for premium-tier campaigns.</div>', unsafe_allow_html=True)

    # ── Products
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Top 10 Products by Quantity Sold</p>', unsafe_allow_html=True)
            tp = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(tp, x="Quantity", y="Description", orientation="h",
                         color="Quantity", color_continuous_scale="Purpor",
                         template="plotly_dark")
            fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Top 10 Products by Revenue</p>', unsafe_allow_html=True)
            tp_r = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
            fig  = px.bar(tp_r, x="TotalPrice", y="Description", orientation="h",
                          color="TotalPrice", color_continuous_scale="Teal",
                          template="plotly_dark")
            fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Gift/lifestyle items dominate sales. Bundling top sellers for seasonal campaigns (Christmas, Valentine\'s Day) can amplify revenue significantly.</div>', unsafe_allow_html=True)

    # ── Time Trends
    with tabs[2]:
        st.markdown('<p class="section-header">Monthly Revenue Trend</p>', unsafe_allow_html=True)
        mr = df.groupby("InvoiceMonth")["TotalPrice"].sum().reset_index()
        mr["InvoiceMonth"] = mr["InvoiceMonth"].astype(str)
        fig = px.line(mr, x="InvoiceMonth", y="TotalPrice", markers=True,
                      template="plotly_dark",
                      labels={"TotalPrice":"Revenue (£)","InvoiceMonth":"Month"})
        fig.update_traces(line_color="#a78bfa", line_width=2.5, marker=dict(size=7))
        fig.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Nov–Dec spike</b> confirms strong holiday-season demand. Plan inventory + logistics 6 weeks ahead. Jan dip = ideal time for loyalty campaigns.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Transactions by Hour</p>', unsafe_allow_html=True)
            ht = df.groupby("InvoiceHour")["InvoiceNo"].nunique().reset_index()
            ht.columns = ["Hour","Transactions"]
            fig = px.area(ht, x="Hour", y="Transactions",
                          template="plotly_dark", color_discrete_sequence=["#7c3aed"])
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Sales by Weekday</p>', unsafe_allow_html=True)
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            wd = (df.groupby("Weekday")["TotalPrice"].sum()
                    .reindex(order).reset_index())
            fig = px.bar(wd, x="Weekday", y="TotalPrice",
                         color="TotalPrice", color_continuous_scale="Purp",
                         template="plotly_dark",
                         labels={"TotalPrice":"Revenue (£)"})
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # ── Spend Patterns
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Transaction Value Distribution</p>', unsafe_allow_html=True)
            tp_clip = df[df["TotalPrice"] < 1000]["TotalPrice"]
            fig = px.histogram(tp_clip, nbins=80, template="plotly_dark",
                               color_discrete_sequence=["#7c3aed"],
                               labels={"value":"TotalPrice (£)"})
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Customer Lifetime Spend Distribution</p>', unsafe_allow_html=True)
            ltv = df.groupby("CustomerID")["TotalPrice"].sum()
            ltv_clip = ltv[ltv < ltv.quantile(0.95)]
            fig = px.histogram(ltv_clip, nbins=60, template="plotly_dark",
                               color_discrete_sequence=["#0891b2"],
                               labels={"value":"Lifetime Spend (£)"})
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Both distributions are <b>right-skewed</b>. A small cluster of high-LTV customers drives outsized revenue — protect them with VIP retention programs.</div>', unsafe_allow_html=True)

    # ── RFM Distributions
    with tabs[4]:
        st.markdown('<p class="section-header">RFM Feature Distributions</p>', unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Recency (days)","Frequency","Monetary (£)"])
        for i, col_name in enumerate(["Recency","Frequency","Monetary"], 1):
            fig.add_trace(go.Histogram(x=rfm[col_name], nbinsx=50,
                                       marker_color=["#7c3aed","#0891b2","#16a34a"][i-1],
                                       name=col_name), row=1, col=i)
        fig.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 All three RFM dimensions are right-skewed — log-transformation before clustering is essential to avoid bias toward high-value outliers.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segments":
    tabs = st.tabs(["📐 Elbow & Silhouette","🗂 Cluster Profiles","🔭 Visualizations","🔮 Predict Segment"])

    # ── Elbow
    with tabs[0]:
        k_range, inertias, sils = elbow_data(rfm_raw)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Elbow Method</p>', unsafe_allow_html=True)
            fig = px.line(x=k_range, y=inertias, markers=True,
                          labels={"x":"Clusters (k)","y":"Inertia"},
                          template="plotly_dark", color_discrete_sequence=["#a78bfa"])
            fig.update_traces(line_width=2.5, marker_size=8)
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Silhouette Scores</p>', unsafe_allow_html=True)
            fig = px.line(x=k_range, y=sils, markers=True,
                          labels={"x":"Clusters (k)","y":"Silhouette Score"},
                          template="plotly_dark", color_discrete_sequence=["#6ee7b7"])
            fig.update_traces(line_width=2.5, marker_size=8)
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        best_k = k_range[int(np.argmax(sils))]
        st.markdown(f'<div class="insight-box">✅ Best k by silhouette = <b>{best_k}</b> | Current k = <b>{n_clusters_val}</b> | Silhouette = <b>{sil_score:.4f}</b></div>', unsafe_allow_html=True)

    # ── Cluster Profiles
    with tabs[1]:
        st.markdown('<p class="section-header">Cluster Summary (mean RFM)</p>', unsafe_allow_html=True)
        cs = cluster_summary.copy().reset_index()
        cs["Label"] = cs["Cluster"].map(lambda c: seg_labels[c][0])
        cs.columns = ["Cluster","Avg Recency","Avg Frequency","Avg Monetary","Segment"]
        st.dataframe(cs.style.background_gradient(subset=["Avg Recency","Avg Frequency","Avg Monetary"],
                                                   cmap="Purples"),
                     use_container_width=True)

        st.markdown('<p class="section-header">Segment Counts</p>', unsafe_allow_html=True)
        sc = rfm["Segment"].value_counts().reset_index()
        sc.columns = ["Segment","Customers"]
        fig = px.pie(sc, values="Customers", names="Segment",
                     color_discrete_sequence=["#7c3aed","#0891b2","#dc2626","#16a34a"],
                     template="plotly_dark", hole=0.45)
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Visualizations
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Frequency vs Monetary</p>', unsafe_allow_html=True)
            fig = px.scatter(rfm, x="Frequency", y="Monetary",
                             color="Segment",
                             color_discrete_sequence=["#7c3aed","#0891b2","#dc2626","#16a34a"],
                             template="plotly_dark", opacity=0.65,
                             hover_data=["CustomerID","Recency"])
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-header">Recency vs Frequency</p>', unsafe_allow_html=True)
            fig = px.scatter(rfm, x="Recency", y="Frequency",
                             color="Segment",
                             size="Monetary",
                             size_max=18,
                             color_discrete_sequence=["#7c3aed","#0891b2","#dc2626","#16a34a"],
                             template="plotly_dark", opacity=0.6,
                             hover_data=["CustomerID","Monetary"])
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">3D RFM Cluster View</p>', unsafe_allow_html=True)
        fig = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
                            color="Segment",
                            color_discrete_sequence=["#7c3aed","#0891b2","#dc2626","#16a34a"],
                            template="plotly_dark", opacity=0.7, size_max=5,
                            hover_data=["CustomerID"])
        fig.update_layout(height=480, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Predict Segment
    with tabs[3]:
        st.markdown('<p class="section-header">🔮 Predict Your Customer Segment</p>', unsafe_allow_html=True)
        st.markdown("Enter RFM values to instantly classify a customer into a segment.")
        col1, col2, col3 = st.columns(3)
        with col1:
            r_in = st.number_input("Recency (days since last purchase)", 0, 730, 60)
        with col2:
            f_in = st.number_input("Frequency (# unique invoices)", 1, 500, 8)
        with col3:
            m_in = st.number_input("Monetary (£ total spent)", 1.0, 100000.0, 1200.0)

        if st.button("🎯 Classify Customer", type="primary"):
            inp = np.array([[np.log1p(r_in), np.log1p(f_in), np.log1p(m_in)]])
            inp_scaled = scaler.transform(inp)
            pred_cluster = kmeans_model.predict(inp_scaled)[0]
            pred_label, pred_color = seg_labels[pred_cluster]
            avg = cluster_summary.loc[pred_cluster]

            st.markdown(f"""
            <div style="background:{pred_color}22; border:2px solid {pred_color};
                        border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem;">
              <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7;">
                Predicted Segment
              </div>
              <div style="font-size:2rem; font-weight:700; color:{pred_color}; margin:0.3rem 0;">
                {pred_label}
              </div>
              <div style="font-size:0.9rem; opacity:0.8;">
                Cluster {pred_cluster} — Avg Recency: {avg['Recency']:.0f}d |
                Avg Frequency: {avg['Frequency']:.1f} |
                Avg Monetary: £{avg['Monetary']:,.0f}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
elif page == "🤖 Recommendations":
    tabs = st.tabs(["🏷 Content-Based (Item→Item)","👥 Collaborative (User→Items)","🔥 Similarity Heatmap"])

    # ── Content-Based
    with tabs[0]:
        st.markdown('<p class="section-header">Content-Based Product Recommendations</p>', unsafe_allow_html=True)
        st.markdown("Select a product — we find the most description-similar items using TF-IDF + Cosine Similarity.")

        all_products = sorted(sim_df.index.tolist())
        selected_product = st.selectbox("🔍 Choose a product", all_products)

        if st.button("🚀 Get Similar Products", type="primary"):
            recs = content_recommend(sim_df, selected_product, n_recs)
            if recs:
                st.markdown(f"**Top {len(recs)} products similar to _{selected_product}_:**")
                for i, r in enumerate(recs, 1):
                    st.markdown(f'<div class="rec-card">#{i} &nbsp; {r}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-box">💡 These products share similar description keywords — bundle them together or show as "Customers also viewed" on product pages.</div>', unsafe_allow_html=True)
            else:
                st.warning("No recommendations found.")

    # ── Collaborative
    with tabs[1]:
        st.markdown('<p class="section-header">User-Based Collaborative Filtering</p>', unsafe_allow_html=True)
        st.markdown("Enter a Customer ID — we identify similar shoppers and recommend products they bought.")

        all_customers = sorted(user_sim_df.index.tolist())
        cid_input = st.selectbox("👤 Choose a Customer ID", all_customers)

        if st.button("🚀 Get Personalised Recs", type="primary"):
            result = collab_recommend(user_item, user_sim_df, cid_input, n_recs)
            if isinstance(result, pd.DataFrame) and not result.empty:
                st.markdown(f"**Recommended StockCodes for Customer `{cid_input}`:**")
                result.columns = ["StockCode","Collaborative Score"]
                result["Collaborative Score"] = result["Collaborative Score"].round(2)
                st.dataframe(result.style.background_gradient(subset=["Collaborative Score"],
                                                              cmap="Purples"),
                             use_container_width=True)
                st.markdown('<div class="insight-box">💡 These recommendations are generated from customers with similar purchase patterns — great for "Recommended For You" email campaigns.</div>', unsafe_allow_html=True)
            else:
                st.warning("Customer not found or no recommendation available.")

    # ── Heatmap
    with tabs[2]:
        st.markdown('<p class="section-header">Product Cosine Similarity Heatmap</p>', unsafe_allow_html=True)
        n_heat = st.slider("Number of top products to compare", 5, 20, 10)
        top_p  = df["Description"].value_counts().head(n_heat).index.tolist()
        heat_df = prod_df[prod_df["Description"].isin(top_p)].reset_index(drop=True)

        if len(heat_df) > 1:
            vec_h   = TfidfVectorizer(stop_words="english")
            tfidf_h = vec_h.fit_transform(heat_df["Clean_Description"])
            sim_h   = cosine_similarity(tfidf_h)

            fig = px.imshow(
                sim_h,
                x=heat_df["Description"].tolist(),
                y=heat_df["Description"].tolist(),
                color_continuous_scale="Purp",
                template="plotly_dark",
                text_auto=".2f",
                aspect="auto",
            )
            fig.update_layout(height=600, margin=dict(l=0,r=0,t=20,b=0),
                              xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="insight-box">💡 Scores close to 1.0 indicate near-identical descriptions. Use these pairs for cross-selling and "similar items" widgets on product detail pages.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 5 — HYPOTHESIS TESTS
# ════════════════════════════════════════════════════════════
elif page == "📈 Hypothesis Tests":
    from scipy import stats

    st.markdown('<p class="section-header">Statistical Hypothesis Testing</p>', unsafe_allow_html=True)
    st.markdown("Three business-relevant hypotheses tested on the cleaned dataset.")

    # ── H1: UK vs Non-UK spend
    with st.expander("📌 H1 — UK vs Non-UK Customer Spending (Welch's t-test)", expanded=True):
        uk     = df[df["Country"] == "United Kingdom"]["TotalPrice"]
        non_uk = df[df["Country"] != "United Kingdom"]["TotalPrice"]
        t_stat, p_val = stats.ttest_ind(uk, non_uk, equal_var=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("T-Statistic", f"{t_stat:.4f}")
        col2.metric("P-Value",     f"{p_val:.4e}")
        col3.metric("Conclusion",  "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀")

        fig = px.box(df[df["TotalPrice"] < 500], x="Country" if False else None,
                     y="TotalPrice", template="plotly_dark",
                     color_discrete_sequence=["#7c3aed"])
        # Use violin instead
        h1_df = pd.DataFrame({"TotalPrice": list(uk[:5000]) + list(non_uk[:5000]),
                               "Region": ["UK"]*5000 + ["Non-UK"]*5000})
        fig = px.violin(h1_df[h1_df["TotalPrice"] < 500], x="Region", y="TotalPrice",
                        color="Region", box=True,
                        color_discrete_sequence=["#7c3aed","#0891b2"],
                        template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if p_val < 0.05:
            st.markdown('<div class="insight-box">✅ <b>Reject H₀</b>: UK and Non-UK customers spend significantly differently. Tailor pricing & promotions per region.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">❌ <b>Fail to Reject H₀</b>: No significant spending difference detected.</div>', unsafe_allow_html=True)

    # ── H2: Quantity vs UnitPrice correlation
    with st.expander("📌 H2 — Quantity ↔ UnitPrice Correlation (Pearson)"):
        corr, p2 = stats.pearsonr(df["Quantity"][:50000], df["UnitPrice"][:50000])
        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r",  f"{corr:.4f}")
        col2.metric("P-Value",    f"{p2:.4e}")
        col3.metric("Conclusion", "Reject H₀" if p2 < 0.05 else "Fail to Reject H₀")

        sample = df.sample(min(3000, len(df)), random_state=42)
        sample_clip = sample[sample["TotalPrice"] < 500]
        fig = px.scatter(sample_clip, x="UnitPrice", y="Quantity",
                         trendline="ols",
                         template="plotly_dark",
                         color_discrete_sequence=["#6ee7b7"],
                         opacity=0.4)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">{"✅ Reject H₀" if p2 < 0.05 else "❌ Fail to Reject H₀"}: Pearson r = {corr:.4f}. {"Weak but statistically significant relationship between price and quantity." if p2 < 0.05 else "No significant correlation found."}</div>', unsafe_allow_html=True)

    # ── H3: ANOVA — weekday spending
    with st.expander("📌 H3 — Weekday Spending Variation (One-Way ANOVA)"):
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wd_df = df[df["TotalPrice"] > 0].copy()
        groups = [wd_df[wd_df["Weekday"] == d]["TotalPrice"].values
                  for d in order if len(wd_df[wd_df["Weekday"] == d]) > 30]
        f_stat, p3 = stats.f_oneway(*groups)

        col1, col2, col3 = st.columns(3)
        col1.metric("F-Statistic", f"{f_stat:.4f}")
        col2.metric("P-Value",     f"{p3:.4e}")
        col3.metric("Conclusion",  "Reject H₀" if p3 < 0.05 else "Fail to Reject H₀")

        wd_avg = wd_df.groupby("Weekday")["TotalPrice"].mean().reindex(order).reset_index()
        wd_avg.columns = ["Weekday","Avg TotalPrice"]
        fig = px.bar(wd_avg, x="Weekday", y="Avg TotalPrice",
                     color="Avg TotalPrice", color_continuous_scale="Purp",
                     template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if p3 < 0.05:
            st.markdown('<div class="insight-box">✅ <b>Reject H₀</b>: Customer spending differs significantly across weekdays. Schedule flash sales and email campaigns on peak-spend days (Thu/Mon).</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">❌ <b>Fail to Reject H₀</b>: No statistically significant weekday spending variation.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#6b7280; font-size:0.82rem; padding:0.5rem 0 1rem;">
  🛒 <b>Shopper Spectrum</b> · P Suman Sangeet · LABMENTIX Bold Analytics · Cohort 2025 &nbsp;|&nbsp;
  Built with Streamlit · Plotly · scikit-learn
</div>
""", unsafe_allow_html=True)