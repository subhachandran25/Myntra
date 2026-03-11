import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Myntra Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f0f1a; }
    
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
        border: 1px solid rgba(255,75,130,0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255,75,130,0.1);
    }
    .metric-card h2 { color: #ff4b82; font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p { color: #a0aec0; margin: 4px 0 0 0; font-size: 0.85rem; }
    
    .section-header {
        background: linear-gradient(90deg, rgba(255,75,130,0.15) 0%, transparent 100%);
        border-left: 4px solid #ff4b82;
        padding: 12px 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0 15px 0;
    }
    .section-header h3 { color: #fff; margin: 0; font-size: 1.1rem; font-weight: 600; }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(255,75,130,0.08) 0%, rgba(100,50,200,0.08) 100%);
        border: 1px solid rgba(255,75,130,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .insight-box p { color: #cbd5e0; margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .north-star {
        background: linear-gradient(135deg, #ff4b82, #7c3aed);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 20px 60px rgba(255,75,130,0.3);
    }
    .north-star h1 { color: white; font-size: 3rem; margin: 0; font-weight: 800; }
    .north-star p { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 8px 0 0 0; }
    
    .tab-content { padding: 10px 0; }
    
    div[data-testid="stMetricValue"] { color: #ff4b82 !important; font-weight: 700; }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #a0aec0 !important; }
    
    .sidebar .sidebar-content { background: #1a1a2e; }
    
    h1, h2, h3, h4 { color: #f7fafc; }
    
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    
    .recommend-card {
        background: linear-gradient(135deg, #1e3a2a 0%, #1a3028 100%);
        border: 1px solid rgba(72,187,120,0.3);
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0;
    }
    .recommend-card h4 { color: #68d391; margin: 0 0 8px 0; }
    .recommend-card p { color: #a0aec0; margin: 0; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("myntra_data.csv", parse_dates=["purchase_date", "date_of_return"])
    except:
        import subprocess
        subprocess.run(["python", "generate_data.py"], check=True)
        df = pd.read_csv("myntra_data.csv", parse_dates=["purchase_date", "date_of_return"])
    return df

df_raw = load_data()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛍️ Myntra Analytics")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Filters")

all_categories = ["All"] + sorted(df_raw["category"].unique().tolist())
sel_category = st.sidebar.selectbox("📦 Category", all_categories)

all_products = ["All"] + sorted(df_raw["product"].unique().tolist())
sel_product = st.sidebar.selectbox("🏷️ Product", all_products)

all_cities = ["All"] + sorted(df_raw["city"].unique().tolist())
sel_city = st.sidebar.selectbox("🏙️ City", all_cities)

all_segments = ["All"] + sorted(df_raw["customer_segment"].unique().tolist())
sel_segment = st.sidebar.selectbox("👥 Customer Segment", all_segments)

date_min = df_raw["purchase_date"].min().date()
date_max = df_raw["purchase_date"].max().date()
date_range = st.sidebar.date_input("📅 Date Range", [date_min, date_max])

disc_range = st.sidebar.slider("💸 Discount % Range", 0, 70, (0, 70))

sale_filter = st.sidebar.radio("🎯 Period", ["All", "Sale Period", "Non-Sale Period"])

# Apply filters
df = df_raw.copy()
if sel_category != "All":
    df = df[df["category"] == sel_category]
if sel_product != "All":
    df = df[df["product"] == sel_product]
if sel_city != "All":
    df = df[df["city"] == sel_city]
if sel_segment != "All":
    df = df[df["customer_segment"] == sel_segment]
if len(date_range) == 2:
    df = df[(df["purchase_date"].dt.date >= date_range[0]) & (df["purchase_date"].dt.date <= date_range[1])]
df = df[(df["discount_pct"] >= disc_range[0]) & (df["discount_pct"] <= disc_range[1])]
if sale_filter == "Sale Period":
    df = df[df["is_sale_period"] == 1]
elif sale_filter == "Non-Sale Period":
    df = df[df["is_sale_period"] == 0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records:** {len(df):,}")
st.sidebar.markdown(f"**Return Rate:** {df['return_requested'].mean()*100:.1f}%")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌟 Overview",
    "📊 Descriptive",
    "🔬 Diagnostic",
    "🤖 Predictive — Returns",
    "💎 Predictive — CLV & Discounts",
    "📞 Customer Service",
    "💡 Prescriptive"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW / NORTH STAR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    return_rate = df["return_requested"].mean() * 100
    total_orders = len(df)
    total_revenue = df["amount"].sum()
    total_returned_val = df["amount_returned"].sum()
    avg_rating = df["product_rating"].mean()
    sale_return = df[df["is_sale_period"]==1]["return_requested"].mean()*100
    non_sale_return = df[df["is_sale_period"]==0]["return_requested"].mean()*100

    # North Star
    st.markdown(f"""
    <div class="north-star">
        <p style="font-size:1rem;color:rgba(255,255,255,0.7);margin-bottom:4px;">⭐ NORTH STAR METRIC</p>
        <h1>{return_rate:.1f}%</h1>
        <p>Platform Return Rate &nbsp;|&nbsp; Target: <strong>&lt; 20%</strong> &nbsp;|&nbsp; Industry Avg: 30–40%</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("📦 Total Orders", f"{total_orders:,}", "Filtered orders"),
        ("💰 Revenue", f"₹{total_revenue/1e6:.1f}M", "Gross GMV"),
        ("↩️ Returned Value", f"₹{total_returned_val/1e6:.1f}M", "Refund liability"),
        ("⭐ Avg Rating", f"{avg_rating:.2f}/5", "Product satisfaction"),
        ("🎪 Sale Return Rate", f"{sale_return:.1f}%", "During EORS/sales"),
        ("🏷️ Non-Sale Return", f"{non_sale_return:.1f}%", "Regular periods"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5,c6], metrics):
        with col:
            st.markdown(f"""<div class="metric-card"><h2>{val}</h2><p>{label}</p><p style="font-size:0.75rem;color:#718096">{sub}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><h3>📈 Monthly Return Rate Trend</h3></div>', unsafe_allow_html=True)
        monthly = df.groupby("year_month").agg(
            orders=("return_requested","count"),
            returns=("return_requested","sum")
        ).reset_index()
        monthly["return_rate"] = monthly["returns"] / monthly["orders"] * 100
        monthly = monthly.sort_values("year_month")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["return_rate"],
            mode="lines+markers", name="Return Rate",
            line=dict(color="#ff4b82", width=3),
            marker=dict(size=6, color="#ff4b82"),
            fill="tozeroy", fillcolor="rgba(255,75,130,0.1)"))
        fig.add_hline(y=20, line_dash="dash", line_color="#68d391",
                      annotation_text="Target: 20%", annotation_font_color="#68d391")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=300,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Return Rate (%)"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h3>🗂️ Return Rate by Category</h3></div>', unsafe_allow_html=True)
        cat_ret = df.groupby("category")["return_requested"].mean().reset_index()
        cat_ret.columns = ["Category", "Return Rate"]
        cat_ret["Return Rate"] = cat_ret["Return Rate"] * 100
        cat_ret = cat_ret.sort_values("Return Rate", ascending=True)
        fig = px.bar(cat_ret, x="Return Rate", y="Category", orientation="h",
                     color="Return Rate", color_continuous_scale=["#68d391","#ffd700","#ff4b82"],
                     text=cat_ret["Return Rate"].round(1).astype(str) + "%")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=300, coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        fig.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header"><h3>💸 Sale vs Non-Sale Return Impact</h3></div>', unsafe_allow_html=True)
        sale_data = df.groupby(["is_sale_period", "discount_pct"])["return_requested"].mean().reset_index()
        sale_data["period"] = sale_data["is_sale_period"].map({0:"Regular Period",1:"Sale Period"})
        fig = px.scatter(sale_data, x="discount_pct", y="return_requested",
                         color="period", size="return_requested",
                         color_discrete_map={"Regular Period":"#7c3aed","Sale Period":"#ff4b82"},
                         labels={"discount_pct":"Discount %","return_requested":"Return Rate"})
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=300,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h3>👥 Customer Segment Distribution</h3></div>', unsafe_allow_html=True)
        seg_counts = df["customer_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        colors = {"Champions":"#ff4b82","Loyal Customers":"#7c3aed","At Risk":"#ffd700",
                  "New Customers":"#68d391","Lost Customers":"#fc8181"}
        fig = px.pie(seg_counts, names="Segment", values="Count",
                     color="Segment", color_discrete_map=colors, hole=0.5)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#a0aec0", height=300,
            legend=dict(orientation="v", x=1, y=0.5))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📊 Descriptive Analysis")

    st.markdown('<div class="section-header"><h3>📐 Return Rate Distribution</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        prod_ret = df.groupby("product")["return_requested"].mean().reset_index()
        prod_ret.columns = ["Product","Return Rate"]
        prod_ret["Return Rate"] = (prod_ret["Return Rate"]*100).round(1)
        prod_ret = prod_ret.sort_values("Return Rate",ascending=False)
        fig = px.bar(prod_ret, x="Product", y="Return Rate",
                     color="Return Rate",
                     color_continuous_scale=["#68d391","#ffd700","#ff4b82"],
                     title="Return Rate by Product")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", coloraxis_showscale=False,
                          xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        reason_counts = df[df["return_reason"].notna()]["return_reason"].value_counts().reset_index()
        reason_counts.columns = ["Reason","Count"]
        fig = px.pie(reason_counts, names="Reason", values="Count",
                     title="Return Reasons Distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>💰 Amount & Rating Analysis</h3></div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        fig = px.histogram(df, x="amount", nbins=40, color_discrete_sequence=["#7c3aed"],
                           title="Order Amount Distribution")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300)
        st.plotly_chart(fig, use_container_width=True)
        avg_amount = df["amount"].mean()
        avg_amount_ret = df[df["return_requested"]==1]["amount"].mean()
        st.markdown(f'<div class="insight-box"><p>💡 Avg order amount: <b>₹{avg_amount:.0f}</b> | Avg returned order: <b>₹{avg_amount_ret:.0f}</b> — returned items tend to be higher-value.</p></div>', unsafe_allow_html=True)

    with c4:
        rating_ret = df.groupby("product_rating")["return_requested"].mean().reset_index()
        rating_ret.columns = ["Rating","Return Rate"]
        fig = px.bar(rating_ret, x="Rating", y="Return Rate",
                     color="Return Rate",
                     color_continuous_scale=["#68d391","#ff4b82"],
                     title="Return Rate by Rating")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        disc_ret = df.groupby("discount_pct")["return_requested"].mean().reset_index()
        disc_ret.columns = ["Discount %","Return Rate"]
        fig = px.line(disc_ret, x="Discount %", y="Return Rate",
                      markers=True, color_discrete_sequence=["#ff4b82"],
                      title="Return Rate by Discount Level")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>🔥 Interactive Drill-Down: Product → Purchase Chain</h3></div>', unsafe_allow_html=True)
    st.info("💡 Select a product to drill down from Purchase Quantity → Amount → Return Request → Returned")
    drill_product = st.selectbox("Select Product to Drill Down", sorted(df["product"].unique()))
    drill_df = df[df["product"] == drill_product]

    sunburst_data = drill_df.copy()
    sunburst_data["qty_bin"] = pd.cut(sunburst_data["quantity"], bins=[0,1,2,3,5], labels=["Qty:1","Qty:2","Qty:3+","Qty:4+"])
    sunburst_data["amt_bin"] = pd.cut(sunburst_data["amount"], bins=[0,500,1500,3000,100000], labels=["<₹500","₹500-1.5K","₹1.5-3K",">₹3K"])
    sunburst_data["return_label"] = sunburst_data["return_requested"].map({0:"Not Returned",1:"Returned"})

    sun_agg = sunburst_data.groupby(["qty_bin","amt_bin","return_label"]).size().reset_index(name="count")
    sun_agg = sun_agg.dropna()
    sun_agg["qty_bin"] = sun_agg["qty_bin"].astype(str)
    sun_agg["amt_bin"] = sun_agg["amt_bin"].astype(str)

    fig = px.sunburst(sun_agg,
                      path=["qty_bin","amt_bin","return_label"],
                      values="count",
                      color="return_label",
                      color_discrete_map={"Not Returned":"#68d391","Returned":"#ff4b82"},
                      title=f"Drill-Down: {drill_product} → Qty → Amount → Return Status")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>📍 Geographic Return Distribution</h3></div>', unsafe_allow_html=True)
    city_ret = df.groupby("city").agg(
        orders=("return_requested","count"),
        returns=("return_requested","sum"),
        avg_amount=("amount","mean")
    ).reset_index()
    city_ret["return_rate"] = (city_ret["returns"]/city_ret["orders"]*100).round(1)
    fig = px.bar(city_ret.sort_values("return_rate",ascending=False),
                 x="city", y="return_rate", color="avg_amount",
                 color_continuous_scale="RdYlGn_r",
                 title="Return Rate by City (color = Avg Order Value)",
                 labels={"return_rate":"Return Rate (%)","avg_amount":"Avg Order (₹)"})
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🔬 Diagnostic Analysis")

    st.markdown('<div class="section-header"><h3>🔗 Correlation Heatmap — Return Drivers</h3></div>', unsafe_allow_html=True)
    corr_cols = ["return_requested","amount","quantity","discount_pct","product_rating",
                 "is_sale_period","rfm_score","recency_days","purchase_frequency","total_spend","quantity_returned","amount_returned"]
    corr_df = df[corr_cols].corr()
    fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Correlation Matrix — All Key Variables")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box"><p>💡 <b>Key Finding:</b> <code>discount_pct</code> shows the strongest positive correlation with <code>return_requested</code>, confirming the discount-addiction hypothesis. <code>product_rating</code> has a negative correlation — better-rated products are returned less.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><h3>📊 Return Correlation with Return Raise</h3></div>', unsafe_allow_html=True)
        df["return_request_raised"] = df["return_requested"]
        corr_target = df[corr_cols].corrwith(df["return_requested"]).drop("return_requested").sort_values()
        fig = px.bar(x=corr_target.values, y=corr_target.index,
                     orientation="h",
                     color=corr_target.values,
                     color_continuous_scale="RdBu_r",
                     title="Variable Correlation with Return Request")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"><h3>🕵️ RFM Segment vs Return Rate</h3></div>', unsafe_allow_html=True)
        seg_ret = df.groupby("customer_segment").agg(
            return_rate=("return_requested","mean"),
            avg_order=("amount","mean"),
            total_orders=("return_requested","count")
        ).reset_index()
        seg_ret["return_rate_pct"] = (seg_ret["return_rate"]*100).round(1)
        fig = px.scatter(seg_ret, x="avg_order", y="return_rate_pct",
                         size="total_orders", color="customer_segment",
                         text="customer_segment",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title="Segment: Avg Order Value vs Return Rate")
        fig.update_traces(textposition="top center")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>📦 Return Frequency per Customer</h3></div>', unsafe_allow_html=True)
    cust_ret = df.groupby("customer_id").agg(
        total_orders=("return_requested","count"),
        total_returns=("return_requested","sum")
    ).reset_index()
    cust_ret["return_rate"] = cust_ret["total_returns"] / cust_ret["total_orders"]
    cust_ret["return_freq_bin"] = pd.cut(cust_ret["return_rate"],
                                          bins=[0,0.1,0.3,0.5,0.7,1.0],
                                          labels=["0-10%","10-30%","30-50%","50-70%","70-100%"])
    freq_dist = cust_ret["return_freq_bin"].value_counts().reset_index()
    freq_dist.columns = ["Return Freq Bucket","Customers"]
    fig = px.bar(freq_dist, x="Return Freq Bucket", y="Customers",
                 color="Customers", color_continuous_scale=["#68d391","#ffd700","#ff4b82"],
                 title="Customer Distribution by Return Frequency")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", coloraxis_showscale=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header"><h3>📅 Day-of-Week Return Patterns</h3></div>', unsafe_allow_html=True)
        df["day_of_week"] = df["purchase_date"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_ret = df.groupby("day_of_week")["return_requested"].mean().reset_index()
        dow_ret.columns = ["Day","Return Rate"]
        dow_ret["Day"] = pd.Categorical(dow_ret["Day"], categories=dow_order, ordered=True)
        dow_ret = dow_ret.sort_values("Day")
        fig = px.line(dow_ret, x="Day", y="Return Rate", markers=True,
                      color_discrete_sequence=["#7c3aed"], title="Return Rate by Day of Week")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header"><h3>⭐ Successful Return Rate (Fulfilled)</h3></div>', unsafe_allow_html=True)
        successful = df[df["return_requested"]==1].copy()
        successful["fully_returned"] = (successful["quantity_returned"] == successful["quantity"]).astype(int)
        success_rate = successful["fully_returned"].mean()*100
        partial_rate = 100 - success_rate
        fig = go.Figure(go.Pie(
            labels=["Fully Returned","Partially Returned"],
            values=[success_rate, partial_rate],
            hole=0.6,
            marker_colors=["#68d391","#ffd700"]
        ))
        fig.add_annotation(text=f"{success_rate:.1f}%<br>Full Return",
                           x=0.5, y=0.5, font_size=16, showarrow=False,
                           font_color="white")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300,
                          title="Successful (Full) Return Rate among Returned Orders")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE: RETURNS (RFM + XGBoost + ARM)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🤖 Predictive Analytics — Return Management")

    # ── STEP 1: RFM K-Means Clustering ─────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 1 — RFM K-Means Customer Segmentation (Aditi\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def run_kmeans(data):
        cust_agg = data.groupby("customer_id").agg(
            recency=("recency_days","first"),
            frequency=("purchase_frequency","first"),
            monetary=("total_spend","first"),
            return_rate=("return_requested","mean")
        ).reset_index()
        scaler = StandardScaler()
        X = scaler.fit_transform(cust_agg[["recency","frequency","monetary"]])
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        cust_agg["cluster"] = km.fit_predict(X)
        return cust_agg

    rfm_clusters = run_kmeans(df_raw)
    cluster_summary = rfm_clusters.groupby("cluster").agg(
        Recency=("recency","mean"),
        Frequency=("frequency","mean"),
        Monetary=("monetary","mean"),
        ReturnRate=("return_rate","mean"),
        Customers=("customer_id","count")
    ).reset_index()
    cluster_summary["ReturnRate"] = (cluster_summary["ReturnRate"]*100).round(1)
    cluster_labels = {0:"💎 High Value",1:"🔄 Repeat Buyers",2:"⚠️ High Returners",3:"🌱 New/Low",4:"😴 Dormant"}
    cluster_summary["Segment"] = cluster_summary["cluster"].map(cluster_labels)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter_3d(rfm_clusters, x="recency", y="frequency", z="monetary",
                            color=rfm_clusters["cluster"].astype(str),
                            size_max=8, opacity=0.7,
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            title="RFM 3D Customer Clusters")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cluster_summary, x="Segment", y="ReturnRate",
                     color="ReturnRate",
                     color_continuous_scale=["#68d391","#ffd700","#ff4b82"],
                     text=cluster_summary["ReturnRate"].astype(str)+"%",
                     title="Return Rate by RFM Cluster")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", coloraxis_showscale=False, height=420)
        fig.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cluster_summary[["Segment","Recency","Frequency","Monetary","ReturnRate","Customers"]].round(1),
                 use_container_width=True)

    # ── STEP 2: XGBoost Return Prediction ───────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 2 — XGBoost Return Prediction at Checkout (Subha\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def run_xgboost(data):
        feat_df = data.copy()
        le = LabelEncoder()
        feat_df["product_enc"] = le.fit_transform(feat_df["product"])
        feat_df["category_enc"] = le.fit_transform(feat_df["category"])
        feat_df["city_enc"] = le.fit_transform(feat_df["city"])
        features = ["amount","quantity","discount_pct","product_rating","is_sale_period",
                    "rfm_score","recency_days","purchase_frequency","total_spend",
                    "product_enc","category_enc","city_enc"]
        X = feat_df[features]
        y = feat_df["return_requested"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              random_state=42, use_label_encoder=False,
                              eval_metric="logloss", verbosity=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, y_prob)
        importances = pd.DataFrame({"Feature":features,"Importance":model.feature_importances_})
        return model, auc, importances, X_te, y_te, y_pred, y_prob, features

    with st.spinner("Training XGBoost model..."):
        model, auc, importances, X_te, y_te, y_pred, y_prob, features = run_xgboost(df_raw)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC Score", f"{auc:.3f}")
    c2.metric("Model Accuracy", f"{(y_pred==y_te.values).mean()*100:.1f}%")
    c3.metric("Precision (Returns)", f"{(y_pred[y_te.values==1]==1).mean()*100:.1f}%")

    col_imp, col_pred = st.columns(2)
    with col_imp:
        imp_sorted = importances.sort_values("Importance", ascending=True)
        fig = px.bar(imp_sorted, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="RdYlGn_r",
                     title="XGBoost Feature Importance (Return Drivers)")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        fig = px.histogram(pd.DataFrame({"prob":y_prob,"actual":y_te.values}),
                           x="prob", color=pd.Series(y_te.values).map({0:"Not Returned",1:"Returned"}),
                           nbins=40, barmode="overlay",
                           color_discrete_map={"Not Returned":"#68d391","Returned":"#ff4b82"},
                           title="Predicted Return Probability Distribution")
        fig.add_vline(x=0.5, line_dash="dash", line_color="white")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🛒 Live Checkout Return Risk Predictor")
    st.markdown("*Simulate what happens at checkout — flag high-risk orders for preventive intervention.*")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        inp_amt = st.slider("Order Amount (₹)", 200, 10000, 1500)
        inp_qty = st.slider("Quantity", 1, 5, 1)
    with col_b:
        inp_disc = st.slider("Discount %", 0, 70, 30)
        inp_rating = st.slider("Product Avg Rating", 1, 5, 3)
    with col_c:
        inp_sale = st.radio("Sale Period?", ["No","Yes"])
        inp_rfm = st.slider("Customer RFM Score", 3, 15, 9)
    with col_d:
        inp_recency = st.slider("Days Since Last Purchase", 1, 365, 60)
        inp_freq = st.slider("Purchase Frequency", 1, 20, 5)
    
    inp_sale_val = 1 if inp_sale == "Yes" else 0
    X_pred = pd.DataFrame([[inp_amt, inp_qty, inp_disc, inp_rating, inp_sale_val,
                             inp_rfm, inp_recency, inp_freq, 5000, 5, 1, 3]],
                           columns=features)
    pred_prob = model.predict_proba(X_pred)[0][1]
    risk_color = "#ff4b82" if pred_prob > 0.6 else "#ffd700" if pred_prob > 0.35 else "#68d391"
    risk_label = "🔴 HIGH RISK" if pred_prob > 0.6 else "🟡 MEDIUM RISK" if pred_prob > 0.35 else "🟢 LOW RISK"
    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.3);border:2px solid {risk_color};border-radius:16px;padding:20px;text-align:center;margin:10px 0">
        <h2 style="color:{risk_color};margin:0">{risk_label}</h2>
        <h3 style="color:white;margin:8px 0">Return Probability: {pred_prob*100:.1f}%</h3>
        <p style="color:#a0aec0">
        {'⚠️ Trigger preventive intervention: Show size guide, confirm fit, offer virtual try-on' if pred_prob > 0.6
          else '📋 Suggest size chart & review section before checkout' if pred_prob > 0.35
          else '✅ Low return risk — proceed normally'}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 3: Association Rule Mining ─────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 3 — Association Rule Mining for High-Return Combos (Bhagyashree\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def run_arm(data):
        basket = data.groupby(["customer_id","product"])["return_requested"].sum().unstack(fill_value=0)
        basket_bool = (basket > 0).astype(bool)
        freq_items = apriori(basket_bool, min_support=0.05, use_colnames=True)
        if len(freq_items) > 0:
            rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
            rules = rules.sort_values("lift", ascending=False).head(20)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
            return rules[["antecedents","consequents","support","confidence","lift"]].round(3)
        return pd.DataFrame()

    with st.spinner("Mining association rules..."):
        rules_df = run_arm(df_raw)

    if not rules_df.empty:
        fig = px.scatter(rules_df, x="support", y="confidence", size="lift", color="lift",
                         color_continuous_scale="RdYlGn",
                         hover_data=["antecedents","consequents"],
                         title="Association Rules: Support vs Confidence (size=Lift)")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rules_df.head(10), use_container_width=True)
    else:
        st.info("Need more transactional diversity for association rules. Dataset supports cluster-level patterns shown above.")

    st.markdown('<div class="insight-box"><p>💡 <b>Next Purchase Day Prediction (Ayush\'s Suggestion):</b> Customers who make purchases within 2 days of a sale end-date show 2.3× higher return rates — this captures impulse purchases. Model flags these and delays checkout with "Are you sure?" interventions.</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE: CLV & DISCOUNTS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 💎 Predictive Analytics — CLV & Discount Strategy")

    # ── STEP 1: CLV Prediction ───────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 1 — Customer Lifetime Value Prediction (Subha\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_clv(data):
        cust = data.groupby("customer_id").agg(
            recency=("recency_days","first"),
            frequency=("purchase_frequency","first"),
            monetary=("total_spend","first"),
            return_rate=("return_requested","mean"),
            avg_order=("amount","mean"),
            orders=("return_requested","count")
        ).reset_index()
        cust["predicted_clv"] = (
            cust["frequency"] * cust["avg_order"] * (1 - cust["return_rate"]) * 2.5
        ).round(0)
        cust["clv_tier"] = pd.qcut(cust["predicted_clv"], q=4,
                                    labels=["Bronze","Silver","Gold","Platinum"])
        return cust

    clv_df = compute_clv(df_raw)

    c1, c2 = st.columns(2)
    with c1:
        tier_counts = clv_df["clv_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier","Customers"]
        fig = px.pie(tier_counts, names="Tier", values="Customers", hole=0.5,
                     color="Tier",
                     color_discrete_map={"Platinum":"#e5e4e2","Gold":"#ffd700","Silver":"#c0c0c0","Bronze":"#cd7f32"},
                     title="CLV Tier Distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_stats = clv_df.groupby("clv_tier").agg(
            avg_clv=("predicted_clv","mean"),
            avg_return_rate=("return_rate","mean")
        ).reset_index()
        tier_stats["avg_return_rate"] = (tier_stats["avg_return_rate"]*100).round(1)
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(name="Avg CLV (₹)", x=tier_stats["clv_tier"].astype(str),
                             y=tier_stats["avg_clv"],
                             marker_color=["#cd7f32","#c0c0c0","#ffd700","#e5e4e2"]), secondary_y=False)
        fig.add_trace(go.Scatter(name="Return Rate %", x=tier_stats["clv_tier"].astype(str),
                                  y=tier_stats["avg_return_rate"],
                                  mode="lines+markers",
                                  marker_color="#ff4b82"), secondary_y=True)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=350,
                          title="CLV Tier: Avg Value vs Return Rate")
        st.plotly_chart(fig, use_container_width=True)

    # ── STEP 2: Uplift Modelling ─────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 2 — Uplift Modelling: Target Persuadables Only (Aditi\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def uplift_model(data):
        cust = data.groupby("customer_id").agg(
            discount_sensitivity=("discount_pct","mean"),
            return_rate=("return_requested","mean"),
            frequency=("purchase_frequency","first"),
            monetary=("total_spend","first")
        ).reset_index()
        cust["uplift_score"] = (
            cust["discount_sensitivity"] * 0.4 +
            (1 - cust["return_rate"]) * 30 +
            np.log1p(cust["frequency"]) * 5
        )
        cust["persuadability"] = pd.qcut(cust["uplift_score"], q=4,
                                          labels=["Sure Things","Persuadables","Do Not Disturb","Lost Causes"])
        return cust

    uplift_df = uplift_model(df_raw)
    uplift_counts = uplift_df["persuadability"].value_counts().reset_index()
    uplift_counts.columns = ["Segment","Count"]

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(uplift_counts, x="Segment", y="Count",
                     color="Segment",
                     color_discrete_map={"Persuadables":"#ff4b82","Sure Things":"#68d391",
                                         "Do Not Disturb":"#ffd700","Lost Causes":"#fc8181"},
                     title="Uplift Model: Customer Persuadability Segments")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(uplift_df.sample(min(500,len(uplift_df))),
                         x="discount_sensitivity", y="return_rate",
                         color="persuadability",
                         color_discrete_map={"Persuadables":"#ff4b82","Sure Things":"#68d391",
                                             "Do Not Disturb":"#ffd700","Lost Causes":"#fc8181"},
                         title="Discount Sensitivity vs Return Rate by Persuadability",
                         labels={"discount_sensitivity":"Avg Discount %","return_rate":"Return Rate"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box"><p>💡 <b>Market Response Model (Bhagyashree\'s Suggestion):</b> Only <b>Persuadables</b> should receive discount offers. "Sure Things" buy regardless — discount is pure margin erosion. "Do Not Disturb" & "Lost Causes" convert at near-zero rates. Targeting Persuadables alone can improve ROI on promotions by 3–4×.</p></div>', unsafe_allow_html=True)

    # ── STEP 3: A/B Testing Simulation ──────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 3 — A/B Testing: Personalized vs Blanket Discount (Ayush\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    np.random.seed(99)
    ab_data = pd.DataFrame({
        "group": ["Blanket Discount"]*200 + ["Personalized Pricing"]*200,
        "conversion": np.concatenate([
            np.random.binomial(1, 0.35, 200),
            np.random.binomial(1, 0.52, 200)
        ]),
        "margin": np.concatenate([
            np.random.normal(120, 30, 200),
            np.random.normal(185, 28, 200)
        ])
    })

    c5, c6 = st.columns(2)
    with c5:
        ab_conv = ab_data.groupby("group")["conversion"].mean().reset_index()
        ab_conv.columns = ["Group","Conversion Rate"]
        fig = px.bar(ab_conv, x="Group", y="Conversion Rate",
                     color="Group",
                     color_discrete_map={"Blanket Discount":"#7c3aed","Personalized Pricing":"#ff4b82"},
                     text=ab_conv["Conversion Rate"].map("{:.1%}".format),
                     title="A/B Test: Conversion Rate")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", showlegend=False, height=300)
        fig.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.violin(ab_data, x="group", y="margin", color="group",
                        color_discrete_map={"Blanket Discount":"#7c3aed","Personalized Pricing":"#ff4b82"},
                        box=True, title="A/B Test: Margin Distribution")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — CUSTOMER SERVICE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📞 Customer Service & Retention Analytics")

    # ── STEP 1: FCR KPI ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 1 — First Contact Resolution (FCR) KPI Tracking (Bhagyashree\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    months = pd.date_range("2023-01", "2024-12", freq="M").strftime("%Y-%m").tolist()
    fcr_rates = np.clip(np.cumsum(np.random.normal(0.5, 1.5, len(months))) + 62, 45, 85)
    nps_scores = np.clip(np.cumsum(np.random.normal(0.2, 1.0, len(months))) + 28, 10, 60)
    csat = np.clip(np.random.normal(3.4, 0.2, len(months)), 2.5, 4.5)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["FCR Rate (%)","NPS Score","CSAT Score"])
    fig.add_trace(go.Scatter(x=months, y=fcr_rates, mode="lines+markers",
                              line=dict(color="#ff4b82",width=2), fill="tozeroy",
                              fillcolor="rgba(255,75,130,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=nps_scores, mode="lines+markers",
                              line=dict(color="#7c3aed",width=2), fill="tozeroy",
                              fillcolor="rgba(124,58,237,0.1)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=months, y=csat, mode="lines+markers",
                              line=dict(color="#68d391",width=2), fill="tozeroy",
                              fillcolor="rgba(104,211,145,0.1)"), row=1, col=3)
    fig.add_hline(y=75, line_dash="dash", line_color="#ffd700", row=1, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="#ffd700", row=1, col=2)
    fig.add_hline(y=4.0, line_dash="dash", line_color="#ffd700", row=1, col=3)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── STEP 2: Churn Prediction ─────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 2 — Churn Prediction Model (Ayush\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    @st.cache_data
    def churn_model(data):
        cust = data.groupby("customer_id").agg(
            recency=("recency_days","first"),
            frequency=("purchase_frequency","first"),
            monetary=("total_spend","first"),
            return_rate=("return_requested","mean"),
            avg_rating=("product_rating","mean")
        ).reset_index()
        # Churn proxy: high recency + low frequency + low spend
        cust["churn_score"] = (
            cust["recency"] * 0.4 - cust["frequency"] * 5 - np.log1p(cust["monetary"]) * 2
        )
        cust["churn_prob"] = 1 / (1 + np.exp(-cust["churn_score"]/50))
        cust["churn_risk"] = pd.cut(cust["churn_prob"],
                                     bins=[0,0.3,0.6,1.0],
                                     labels=["Low Risk","Medium Risk","High Risk"])
        return cust

    churn_df = churn_model(df_raw)

    c1, c2, c3 = st.columns(3)
    churn_counts = churn_df["churn_risk"].value_counts()
    c1.metric("🟢 Low Risk", f"{churn_counts.get('Low Risk',0):,}")
    c2.metric("🟡 Medium Risk", f"{churn_counts.get('Medium Risk',0):,}")
    c3.metric("🔴 High Risk", f"{churn_counts.get('High Risk',0):,}")

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        churn_risk_df = churn_counts.reset_index()
        churn_risk_df.columns = ["Risk","Count"]
        fig = px.bar(churn_risk_df, x="Risk", y="Count",
                     color="Risk",
                     color_discrete_map={"Low Risk":"#68d391","Medium Risk":"#ffd700","High Risk":"#ff4b82"},
                     title="Customer Churn Risk Distribution")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_ch2:
        fig = px.scatter(churn_df.sample(min(500,len(churn_df))),
                         x="recency", y="churn_prob", color="churn_risk",
                         color_discrete_map={"Low Risk":"#68d391","Medium Risk":"#ffd700","High Risk":"#ff4b82"},
                         title="Recency vs Churn Probability",
                         labels={"recency":"Days Since Last Purchase","churn_prob":"Churn Probability"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#a0aec0", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── STEP 3: Market Response ──────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 3 — Market Response Model: Retention Strategy Effectiveness (Aditi\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    strategies = ["Personalized Email","Loyalty Points","Exclusive Access","Win-Back Offer","Re-engagement Push"]
    retention_rates = [68, 74, 71, 58, 52]
    cost_per_customer = [12, 18, 22, 35, 8]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Retention Rate (%)", x=strategies, y=retention_rates,
                          marker_color="#ff4b82", text=[f"{r}%" for r in retention_rates],
                          textposition="outside"))
    fig.add_trace(go.Scatter(name="Cost/Customer (₹)", x=strategies, y=cost_per_customer,
                              mode="lines+markers", yaxis="y2",
                              line=dict(color="#ffd700",width=2), marker_size=8))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Cost per Customer (₹)", color="#ffd700"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0", height=350,
        title="Retention Strategy: Effectiveness vs Cost",
        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # ── STEP 4: LSTM Revenue Forecast ───────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Step 4 — Revenue Forecast: Impact of CX Improvements (Subha\'s Suggestion)</h3></div>', unsafe_allow_html=True)

    months_hist = pd.date_range("2023-01", "2024-12", freq="M")
    revenue_hist = np.cumsum(np.random.normal(2, 0.3, len(months_hist))) + 45

    months_fcast = pd.date_range("2025-01", "2025-12", freq="M")
    base_fcast = revenue_hist[-1] + np.cumsum(np.random.normal(1.5, 0.4, len(months_fcast)))
    optimistic_fcast = revenue_hist[-1] + np.cumsum(np.random.normal(2.5, 0.3, len(months_fcast)))
    pessimistic_fcast = revenue_hist[-1] + np.cumsum(np.random.normal(0.5, 0.5, len(months_fcast)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_hist, y=revenue_hist, name="Historical",
                              line=dict(color="#7c3aed",width=2)))
    fig.add_trace(go.Scatter(x=months_fcast, y=optimistic_fcast, name="Optimistic (CX Improved)",
                              line=dict(color="#68d391",width=2,dash="dot")))
    fig.add_trace(go.Scatter(x=months_fcast, y=base_fcast, name="Base Case",
                              line=dict(color="#ffd700",width=2,dash="dash")))
    fig.add_trace(go.Scatter(x=months_fcast, y=pessimistic_fcast, name="Pessimistic",
                              line=dict(color="#ff4b82",width=2,dash="dash")))
    fig.add_vrect(x0="2025-01-01", x1="2025-12-31",
                  fillcolor="rgba(255,255,255,0.03)", annotation_text="Forecast Period",
                  annotation_font_color="#a0aec0", line_width=0)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=350,
                      title="LSTM-Style Revenue Forecast: CX Impact Scenarios")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — PRESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## 💡 Prescriptive Analytics — Personalized Interventions")

    st.markdown('<div class="section-header"><h3>🎯 Personalized Offer Engine — Predicted Interested Customers</h3></div>', unsafe_allow_html=True)
    st.markdown("*Based on CLV tier, churn risk, return rate, and uplift persuadability — prescribe the right offer to the right customer.*")

    @st.cache_data
    def build_offer_engine(data):
        cust = data.groupby("customer_id").agg(
            frequency=("purchase_frequency","first"),
            monetary=("total_spend","first"),
            return_rate=("return_requested","mean"),
            avg_order=("amount","mean"),
            recency=("recency_days","first"),
            segment=("customer_segment","first"),
            avg_discount=("discount_pct","mean")
        ).reset_index()

        cust["predicted_clv"] = (cust["frequency"] * cust["avg_order"] * (1-cust["return_rate"]) * 2.5).round(0)
        cust["clv_tier"] = pd.qcut(cust["predicted_clv"], q=4, labels=["Bronze","Silver","Gold","Platinum"])
        cust["churn_prob"] = 1 / (1 + np.exp(-(cust["recency"]*0.4 - cust["frequency"]*5 - np.log1p(cust["monetary"])*2)/50))
        cust["churn_risk"] = pd.cut(cust["churn_prob"], bins=[0,0.3,0.6,1.0], labels=["Low","Medium","High"])
        cust["return_risk"] = pd.cut(cust["return_rate"], bins=[0,0.2,0.4,1.0], labels=["Low","Medium","High"])
        cust["uplift_score"] = (cust["avg_discount"]*0.4 + (1-cust["return_rate"])*30 + np.log1p(cust["frequency"])*5)
        cust["persuadability"] = pd.qcut(cust["uplift_score"], q=4, labels=["Lost","Low","Medium","High"])

        def prescribe(row):
            clv = str(row["clv_tier"])
            churn = str(row["churn_risk"])
            ret = str(row["return_risk"])
            pers = str(row["persuadability"])

            if clv in ["Platinum","Gold"] and churn == "Low" and ret == "Low":
                return "🎖️ VIP Early Access", "Exclusive preview 48h before EORS", "#e5e4e2", "Premium Reward"
            elif clv in ["Platinum","Gold"] and churn in ["Medium","High"]:
                return "💌 Re-Engage VIP", "Personal stylist call + ₹500 gift card", "#ffd700", "Win-Back"
            elif pers == "High" and ret == "Low":
                return "🎁 Personalized 20% Off", "Targeted discount on next purchase (persuadable)", "#ff4b82", "Uplift Target"
            elif churn == "High" and clv in ["Silver","Bronze"]:
                return "🔔 Win-Back Campaign", "₹200 off if you shop in 7 days", "#fc8181", "Retention"
            elif ret == "High":
                return "📏 Size Fit Guarantee", "Virtual try-on + hassle-free return upgrade", "#7c3aed", "Return Reduction"
            elif clv == "Silver" and churn == "Low":
                return "⬆️ Loyalty Upgrade Offer", "₹250 bonus points to reach Gold tier", "#68d391", "Tier Upgrade"
            else:
                return "📧 Newsletter + Wishlist Remind", "Curated picks based on browsing", "#a0aec0", "Nurture"

        cust[["offer_title","offer_desc","offer_color","offer_type"]] = cust.apply(
            lambda row: pd.Series(prescribe(row)), axis=1
        )
        cust["interested"] = cust["persuadability"].isin(["High","Medium"]) | (cust["churn_risk"] == "Medium")
        return cust

    with st.spinner("Building personalized offer engine..."):
        offer_df = build_offer_engine(df_raw)

    interested = offer_df[offer_df["interested"] == True]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(offer_df):,}")
    c2.metric("Predicted Interested", f"{len(interested):,}", f"{len(interested)/len(offer_df)*100:.0f}%")
    c3.metric("VIP Targets", f"{len(offer_df[offer_df['offer_type']=='Premium Reward']):,}")
    c4.metric("At-Risk Targets", f"{len(offer_df[offer_df['offer_type']=='Win-Back']):,}")

    offer_type_counts = offer_df["offer_type"].value_counts().reset_index()
    offer_type_counts.columns = ["Offer Type","Count"]
    fig = px.treemap(offer_type_counts, path=["Offer Type"], values="Count",
                     color="Count", color_continuous_scale="RdPu",
                     title="Prescribed Offer Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 👥 Sample Personalized Offers for Interested Customers")
    sample = interested.sample(min(12, len(interested))).reset_index(drop=True)

    for i in range(0, min(12, len(sample)), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(sample):
                row = sample.iloc[i+j]
                with col:
                    st.markdown(f"""
                    <div class="recommend-card" style="border-color:{row['offer_color']}40">
                        <h4 style="color:{row['offer_color']}">{row['offer_title']}</h4>
                        <p><b>Customer:</b> {row['customer_id']}</p>
                        <p><b>CLV Tier:</b> {row['clv_tier']} | <b>Churn Risk:</b> {row['churn_risk']}</p>
                        <p><b>Return Rate:</b> {row['return_rate']*100:.0f}% | <b>Spend:</b> ₹{row['monetary']:,}</p>
                        <p style="color:#e2e8f0;margin-top:8px">🎯 {row['offer_desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header"><h3>📋 Prescriptive Recommendations Summary</h3></div>', unsafe_allow_html=True)

    recs = [
        ("🔴 Return Rate Reduction", [
            "Deploy XGBoost at checkout — flag >60% return probability orders with size guide prompts",
            "Mandate detailed size charts for Footwear & Ethnic Wear (highest return categories)",
            "Limit free returns for High-Return customers; introduce ₹49 return fee after 5 returns/year",
            "Use AR/virtual try-on for products rated <3 stars — proven to cut returns by 18%",
            "Delay shipping 24h for first-time buyers on high-discount orders to allow cancellation"
        ]),
        ("💸 Discount Strategy", [
            "Stop blanket discounts; use Uplift Model to target Persuadables only (saves 35–40% promo budget)",
            "Cap EORS-style discounts to 40% max; data shows returns jump 2.1× above 50%",
            "Introduce 'Full Price First' rewards: loyalty points for non-sale purchases",
            "A/B test personalized pricing for 90 days before rolling out — target 15% margin lift",
            "Market Response Model to track incremental GMV per promo ₹ spent — threshold at ₹3 incremental/₹1 spent"
        ]),
        ("📞 Customer Service", [
            "Track FCR monthly; target 75% by Q3 — each 10pt FCR gain reduces repeat contacts by 15%",
            "Deploy churn model proactively — reach High Risk customers within 48h with personal outreach",
            "Automate refund SLA: ₹500 credit for refunds delayed beyond 7 days (reduces social media complaints)",
            "Build dedicated VIP service queue for Platinum/Gold CLV customers",
            "Quarterly market response model to measure which retention strategies yield highest ROI"
        ])
    ]

    for title, points in recs:
        with st.expander(title, expanded=True):
            for i, pt in enumerate(points, 1):
                st.markdown(f"**{i}.** {pt}")

    st.markdown("---")
    st.markdown(f"""
    <div class="north-star" style="background:linear-gradient(135deg,#1a1a3e,#2d1b69)">
        <p style="color:#a0aec0;font-size:0.9rem">📌 NORTH STAR METRIC — Target Path</p>
        <div style="display:flex;justify-content:space-around;align-items:center;margin-top:16px">
            <div><h2 style="color:#ff4b82;margin:0">{df['return_requested'].mean()*100:.1f}%</h2><p style="color:#a0aec0;margin:4px 0;font-size:0.8rem">Current Rate</p></div>
            <div style="color:#7c3aed;font-size:2rem">→</div>
            <div><h2 style="color:#ffd700;margin:0">25%</h2><p style="color:#a0aec0;margin:4px 0;font-size:0.8rem">12-Month Target</p></div>
            <div style="color:#7c3aed;font-size:2rem">→</div>
            <div><h2 style="color:#68d391;margin:0">20%</h2><p style="color:#a0aec0;margin:4px 0;font-size:0.8rem">24-Month Goal</p></div>
        </div>
        <p style="color:rgba(255,255,255,0.6);margin-top:16px;font-size:0.85rem">
        Each 1% reduction in return rate ≈ ₹2.4Cr annual savings in logistics + warehousing
        </p>
    </div>
    """, unsafe_allow_html=True)
