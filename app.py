import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Fake Review Detection", layout="wide")

st.title("🔍 Fake Review Detection Dashboard")

@st.cache_data
def load_sample_data():
    """Generate sample prediction data for demo."""
    import random
    risk_levels = ["high", "medium", "low"]
    rules = [
        "account_age < 3",
        "time_between_reviews < 1",
        "report_count > 10",
        "rating == 5 and fake_prob > 0.6",
    ]
    
    data = []
    base_time = datetime.utcnow()
    for i in range(100):
        risk = random.choice(risk_levels)
        timestamp = base_time - timedelta(hours=random.randint(0, 168))
        
        if risk == "high":
            confidence = random.uniform(0.75, 1.0)
            fake_prob = random.uniform(0.6, 1.0)
            applied = random.sample(rules, k=random.randint(1, 3))
        elif risk == "medium":
            confidence = random.uniform(0.5, 0.75)
            fake_prob = random.uniform(0.3, 0.7)
            applied = random.sample(rules, k=random.randint(1, 2))
        else:
            confidence = random.uniform(0.0, 0.5)
            fake_prob = random.uniform(0.0, 0.3)
            applied = []
        
        data.append({
            "risk_level": risk,
            "confidence": round(confidence, 2),
            "fake_probability": round(fake_prob, 4),
            "decision_source": "hybrid" if applied else "model",
            "applied_rules": ", ".join(applied) if applied else "none",
            "explanation": f"Review analysis complete.",
            "timestamp": timestamp.isoformat() + "Z",
        })
    return pd.DataFrame(data)


def load_data_from_file():
    """Load prediction data from uploaded CSV or JSON file."""
    uploaded_file = st.sidebar.file_uploader("Upload predictions CSV or JSON", type=["csv", "json"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            data = json.load(uploaded_file)
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                st.warning("JSON must be an array of prediction objects")
                return None
    return None


def style_risk_level(val):
    """Return color code for risk level."""
    colors = {
        "high": "#ff4444",
        "medium": "#ffbb33",
        "low": "#00cc44",
    }
    color = colors.get(str(val).lower(), "#333333")
    return f"color: white; background-color: {color}; padding: 6px; border-radius: 4px; text-align: center; font-weight: bold;"


st.sidebar.header("📊 Dashboard Options")

# Data source selector
data_source = st.sidebar.radio("Data Source", ["Sample Data", "Upload File"])

if data_source == "Upload File":
    df = load_data_from_file()
    if df is None:
        st.info("No file uploaded. Using sample data for demo.")
        df = load_sample_data()
else:
    df = load_sample_data()

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Filter by risk level
st.sidebar.header("🔄 Filters")
selected_risks = st.sidebar.multiselect(
    "Filter by Risk Level",
    options=["high", "medium", "low"],
    default=["high", "medium", "low"],
)
df_filtered = df[df["risk_level"].isin(selected_risks)]

# Summary Metrics
st.header("📈 Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(df_filtered)
high_risk_count = len(df_filtered[df_filtered["risk_level"] == "high"])
medium_risk_count = len(df_filtered[df_filtered["risk_level"] == "medium"])
low_risk_count = len(df_filtered[df_filtered["risk_level"] == "low"])

with col1:
    st.metric("📋 Total Reviews", total_reviews)
with col2:
    st.metric("🔴 High Risk", high_risk_count, delta=f"{(high_risk_count/total_reviews*100):.1f}%" if total_reviews > 0 else "0%")
with col3:
    st.metric("🟡 Medium Risk", medium_risk_count, delta=f"{(medium_risk_count/total_reviews*100):.1f}%" if total_reviews > 0 else "0%")
with col4:
    st.metric("🟢 Low Risk", low_risk_count, delta=f"{(low_risk_count/total_reviews*100):.1f}%" if total_reviews > 0 else "0%")

# Charts
st.header("📊 Risk Distribution")
chart_col1, chart_col2 = st.columns(2)

# Bar Chart
with chart_col1:
    risk_counts = df_filtered["risk_level"].value_counts().sort_index()
    st.bar_chart(risk_counts)
    st.caption("Risk Level Distribution (Count)")

# Pie Chart
with chart_col2:
    pie_data = df_filtered["risk_level"].value_counts()
    if len(pie_data) > 0:
        st.write("")  # Spacing for alignment
        st.bar_chart(pie_data)
        st.caption("Risk Level Proportion (%)")
    else:
        st.info("No data to display")

# Data Table
st.header("📋 Detailed Review Data")

if len(df_filtered) > 0:
    display_df = df_filtered[
        ["risk_level", "confidence", "decision_source", "applied_rules", "explanation", "timestamp"]
    ].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    
    # Color-coded risk legend
    st.subheader("Risk Level Legend")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    with legend_col1:
        st.markdown("🔴 **High Risk** - Likely fake review", unsafe_allow_html=True)
    with legend_col2:
        st.markdown("🟡 **Medium Risk** - Suspicious behavior detected", unsafe_allow_html=True)
    with legend_col3:
        st.markdown("🟢 **Low Risk** - Review appears genuine", unsafe_allow_html=True)
else:
    st.warning("No data matches the selected filters.")

# Summary Stats
st.header("📊 Statistics")
stats_col1, stats_col2, stats_col3 = st.columns(3)

with stats_col1:
    avg_confidence = df_filtered["confidence"].mean()
    st.metric("Avg Confidence", f"{avg_confidence:.2f}" if not pd.isna(avg_confidence) else "N/A")

with stats_col2:
    avg_fake_prob = df_filtered["fake_probability"].mean()
    st.metric("Avg Fake Probability", f"{avg_fake_prob:.4f}" if not pd.isna(avg_fake_prob) else "N/A")

with stats_col3:
    hybrid_count = len(df_filtered[df_filtered["decision_source"] == "hybrid"])
    st.metric("Hybrid Decisions", hybrid_count)

# Footer
st.divider()
st.caption("🔐 Fake Review Detection Dashboard | Data updates in real-time")
