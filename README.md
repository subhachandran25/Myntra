# 🛍️ Myntra Returns & Analytics Dashboard

A comprehensive Streamlit analytics dashboard addressing Myntra's three core business challenges:
1. **High Return Rates (30–40%)**
2. **Discount Addiction / EORS Dependency**
3. **Poor Customer Service & Churn**

## 📊 Dashboard Tabs

| Tab | Analysis Type | Key Features |
|-----|--------------|--------------|
| 🌟 Overview | North Star KPI | Return rate trend, category breakdown, segment distribution |
| 📊 Descriptive | Distribution Analysis | Product return rates, reasons, drill-down sunburst, geographic |
| 🔬 Diagnostic | Correlation & RFM | Correlation heatmap, return frequency, day-of-week patterns |
| 🤖 Predictive — Returns | ML Models | K-Means RFM clustering, XGBoost checkout predictor, ARM |
| 💎 Predictive — CLV | Uplift + A/B | CLV tiers, persuadability segmentation, A/B test simulation |
| 📞 Customer Service | FCR + Churn | KPI tracking, churn model, market response, LSTM forecast |
| 💡 Prescriptive | Offer Engine | Personalized interventions, recommendations, north star roadmap |

## 🚀 Deploying to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Myntra Analytics Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/myntra-dashboard.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

### Step 3: Using Your Own Data
Replace `myntra_data.csv` with your actual dataset. Ensure it has these columns:
- `customer_id`, `purchase_date`, `product`, `category`
- `quantity`, `amount`, `discount_pct`, `is_sale_period`
- `return_requested` (0/1), `return_reason`, `date_of_return`
- `quantity_returned`, `amount_returned`, `product_rating`
- `city`, `customer_segment`, `rfm_score`
- `recency_days`, `purchase_frequency`, `total_spend`

## 🏃 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🎯 North Star Metric
**Return Rate** — Target path: Current (~36%) → 25% (12 months) → 20% (24 months)

Each 1% reduction ≈ ₹2.4 Cr annual savings in logistics + warehousing.

## 👥 Team Credits
- **Aditi** — RFM Clustering, Uplift Modelling
- **Subha Chandran** — XGBoost Checkout Model, CLV Prediction, LSTM Forecasting
- **Bhagyashree** — Association Rule Mining, FCR KPI, Market Response Model
- **Ayush** — Next Purchase Day Prediction, A/B Testing, Churn Prediction
