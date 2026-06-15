"""
Airline Loyalty Program — Retention Intelligence Dashboard
Consulting & Analytics Club, IIT Guwahati | Summer Projects 2026

Run with: streamlit run app.py
Requires: final_customer_segments.csv in the same folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings("ignore")

# Resolve paths relative to this script file so Streamlit Cloud finds the CSV
# regardless of which directory the process is launched from
HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty — Retention Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F8F9FC; }

    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 1.5rem; color: white;
    }
    .header-title { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .header-sub   { font-size: 0.95rem; color: #94a3b8; margin: 0.3rem 0 0; font-weight: 400; }
    .header-badge {
        display: inline-block; background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2); padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; color: #cbd5e1;
        margin-top: 0.8rem; margin-right: 6px;
    }

    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
        border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: #64748b; margin: 0 0 4px;
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; margin: 0; line-height: 1.1; }
    .metric-sub   { font-size: 0.8rem; color: #94a3b8; margin: 2px 0 0; }
    .purple { color: #6366f1; } .coral  { color: #ef4444; }
    .amber  { color: #f59e0b; } .green  { color: #10b981; }
    .blue   { color: #3b82f6; }

    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e293b;
        margin: 0 0 1rem; padding-bottom: 0.6rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .action-card {
        background: white; border-radius: 12px; padding: 1.4rem;
        border-left: 4px solid; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .action-title  { font-size: 1rem; font-weight: 700; margin: 0 0 4px; }
    .action-count  { font-size: 0.8rem; color: #64748b; margin: 0 0 1rem; }
    .action-row    { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px; font-size: 0.85rem; color: #374151; }
    .action-icon   { font-size: 0.9rem; flex-shrink: 0; margin-top: 1px; }
    .action-label  { font-weight: 600; color: #1e293b; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #f1f5f9; padding: 6px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-weight: 500; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA  (single CSV now)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, "final_customer_segments.csv"))
    return df

@st.cache_data
def build_dashboard_data(df):
    dashboard = pd.DataFrame()

    # ── Core model outputs already in CSV ──
    dashboard["churn_probability"]         = df["churn_probability"].values
    dashboard["churned"]                   = df["churned"].values

    # ── CLV: reverse log transform (CSV stores CLV_log = log(CLV)) ──
    dashboard["CLV"]                       = np.exp(df["CLV_log"].values)

    # ── Member features ──
    dashboard["Salary"]                    = df["Salary"].values
    dashboard["Tenure_Months"]             = df["Tenure_Months"].values
    dashboard["Total_Flights"]             = df["Total_Flights"].values
    dashboard["Months_Since_Last_Flight"]  = df["Months_Since_Last_Flight"].values
    dashboard["Redemption_Rate"]           = df["Redemption_Rate"].values
    dashboard["Zero_Activity_Months"]      = df["Zero_Activity_Months"].values
    dashboard["Total_Points_Acc"]          = df["Total_Points_Acc"].values

    # ── Cluster labels now come directly from the merged CSV ──
    dashboard["Cluster"] = df["cluster"].values

    # Segment names (verify against your actual cluster assignments)
    segment_map = {
        0: "Disengaged At-Risk Members",
        1: "Loyal Core Members",
        2: "High-Frequency New Flyers"
    }
    dashboard["Segment"] = dashboard["Cluster"].map(segment_map)

    # ── Risk tier ──
    def risk_tier(p):
        if p >= 0.5:   return "High Risk"
        elif p >= 0.2: return "Medium Risk"
        return "Low Risk"
    dashboard["Risk Tier"] = dashboard["churn_probability"].apply(risk_tier)

    # ── Recommended action ──
    action_map = {
        "High-Frequency New Flyers"   : "Points Activation Campaign",
        "Disengaged At-Risk Members"  : "Win-Back Campaign",
        "Loyal Core Members"          : "Loyalty Recognition"
    }
    dashboard["Recommended Action"] = dashboard["Segment"].map(action_map)

    # ── Loyalty card ──
    card_map = {0: "Star", 1: "Nova", 2: "Aurora"}
    dashboard["Loyalty Card"] = df["loyalty_card_encoded"].map(card_map)

    # ── Churn % ──
    dashboard["Churn Risk %"] = (dashboard["churn_probability"] * 100).round(1)

    return dashboard

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df_raw    = load_data()
    dashboard = build_dashboard_data(df_raw)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <p class="header-title">✈️ Airline Loyalty — Retention Intelligence</p>
    <p class="header-sub">Consulting & Analytics Club · IIT Guwahati · Summer Projects 2026</p>
    <span class="header-badge">XGBoost · AUC 0.9883</span>
    <span class="header-badge">K-Means · 3 Segments</span>
    <span class="header-badge">16,737 Members</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters")
st.sidebar.markdown("---")

seg_options  = ["All Segments"] + sorted(dashboard["Segment"].dropna().unique().tolist())
risk_options = ["All Risk Tiers", "High Risk", "Medium Risk", "Low Risk"]
card_options = ["All Tiers", "Star", "Nova", "Aurora"]

selected_seg  = st.sidebar.selectbox("Segment",      seg_options)
selected_risk = st.sidebar.selectbox("Risk Tier",    risk_options)
selected_card = st.sidebar.selectbox("Loyalty Card", card_options)
min_risk      = st.sidebar.slider("Min Churn Risk %", 0, 100, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97%    |
| Recall    | 94%    |
| Precision | 88%    |
| AUC-ROC   | 0.9883 |
""")

# Apply filters
filtered = dashboard.copy()
if selected_seg  != "All Segments":   filtered = filtered[filtered["Segment"]      == selected_seg]
if selected_risk != "All Risk Tiers": filtered = filtered[filtered["Risk Tier"]    == selected_risk]
if selected_card != "All Tiers":      filtered = filtered[filtered["Loyalty Card"] == selected_card]
filtered = filtered[filtered["Churn Risk %"] >= min_risk]

# ─────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────
total     = len(dashboard)
churned   = int(dashboard["churned"].sum())
high_risk = int((dashboard["Risk Tier"] == "High Risk").sum())
avg_risk  = dashboard["Churn Risk %"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Total Members</p>
        <p class="metric-value purple">{total:,}</p>
        <p class="metric-sub">In loyalty program</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Churned</p>
        <p class="metric-value coral">{churned:,}</p>
        <p class="metric-sub">{churned/total*100:.1f}% of members</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">High Risk</p>
        <p class="metric-value amber">{high_risk:,}</p>
        <p class="metric-sub">Need immediate action</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Avg Churn Risk</p>
        <p class="metric-value blue">{avg_risk:.1f}%</p>
        <p class="metric-sub">Across all members</p>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Showing</p>
        <p class="metric-value green">{len(filtered):,}</p>
        <p class="metric-sub">After filters applied</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Customer Risk Table",
    "🎯 Segment Analysis",
    "📈 Visual Insights",
    "💡 Recommended Actions"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — CUSTOMER RISK TABLE
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Customer Risk Table</p>', unsafe_allow_html=True)
    st.markdown(f"Showing **{len(filtered):,}** customers · Sorted by churn risk (highest first)")

    display_cols = {
        "Segment"                  : "Segment",
        "Risk Tier"                : "Risk Tier",
        "Churn Risk %"             : "Churn Risk %",
        "Loyalty Card"             : "Loyalty Card",
        "Tenure_Months"            : "Tenure (Months)",
        "Total_Flights"            : "Total Flights",
        "Months_Since_Last_Flight" : "Months Since Last Flight",
        "Redemption_Rate"          : "Redemption Rate",
        "CLV"                      : "CLV",
        "Recommended Action"       : "Recommended Action"
    }

    table_df = filtered[list(display_cols.keys())].rename(columns=display_cols)
    table_df = table_df.sort_values("Churn Risk %", ascending=False).reset_index(drop=True)
    table_df["Redemption Rate"] = table_df["Redemption Rate"].round(3)
    table_df["CLV"]             = table_df["CLV"].round(0).astype(int)

    st.dataframe(
        table_df,
        use_container_width=True,
        height=450,
        column_config={
            "Churn Risk %": st.column_config.ProgressColumn(
                "Churn Risk %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Redemption Rate": st.column_config.NumberColumn(
                "Redemption Rate", format="%.3f"
            ),
            "CLV": st.column_config.NumberColumn(
                "CLV (CAD)", format="$%d"
            )
        }
    )

    csv = table_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download filtered list as CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Segment Analysis</p>', unsafe_allow_html=True)

    # Compute live segment stats from data
    seg_stats = dashboard.groupby("Segment").agg(
        members    = ("churned", "count"),
        churn_rate = ("churned", "mean"),
        tenure     = ("Tenure_Months", "mean"),
        flights    = ("Total_Flights", "mean"),
        redemp     = ("Redemption_Rate", "mean"),
        high_risk  = ("Risk Tier", lambda x: (x == "High Risk").sum())
    ).reset_index()

    seg_meta = {
        "High-Frequency New Flyers": {
            "color": "#6366f1", "icon": "🚀",
            "insight": "Most active flyers but zero program engagement. Fly frequently without redeeming points — transactional, not loyal yet."
        },
        "Disengaged At-Risk Members": {
            "color": "#ef4444", "icon": "⚠️",
            "insight": "Inactive for most of their tenure. High redemption rate is the key exit signal — customers drain their points before leaving."
        },
        "Loyal Core Members": {
            "color": "#10b981", "icon": "⭐",
            "insight": "Backbone of the program. Longest tenured, steady flyers, very low churn. The high-risk members in this segment are the most valuable to investigate immediately."
        }
    }

    for _, row in seg_stats.iterrows():
        seg_name = row["Segment"]
        meta     = seg_meta.get(seg_name, {"color": "#64748b", "icon": "•", "insight": ""})
        pct      = row["members"] / total * 100

        with st.expander(
            f"{meta['icon']} {seg_name} — {int(row['members']):,} members ({pct:.1f}%)",
            expanded=True
        ):
            col_a, col_b, col_c = st.columns([2, 2, 3])

            with col_a:
                st.markdown(f"""
                <div style="background:{meta['color']}15; border-left:3px solid {meta['color']};
                            padding:12px 16px; border-radius:8px;">
                    <p style="margin:0 0 8px; font-size:0.75rem; font-weight:600;
                              text-transform:uppercase; color:{meta['color']}">Key Stats</p>
                    <div style="font-size:0.85rem; color:#374151; line-height:1.8">
                        <b>Churn rate:</b> {row['churn_rate']*100:.1f}%<br>
                        <b>Avg tenure:</b> {row['tenure']:.1f} months<br>
                        <b>Avg flights:</b> {row['flights']:.2f} total<br>
                        <b>Avg redemption:</b> {row['redemp']:.2f}
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0;
                            padding:12px 16px; border-radius:8px;">
                    <p style="margin:0 0 8px; font-size:0.75rem; font-weight:600;
                              text-transform:uppercase; color:#64748b">Risk Breakdown</p>
                    <div style="font-size:0.85rem; color:#374151; line-height:1.8">
                        <span style="color:#ef4444; font-weight:600">● High Risk:</span>
                        {int(row['high_risk']):,} members<br>
                        <span style="color:#94a3b8; font-size:0.78rem">Need immediate intervention</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_c:
                st.markdown(f"""
                <div style="background:#fffbeb; border:1px solid #fde68a;
                            padding:12px 16px; border-radius:8px; height:100%">
                    <p style="margin:0 0 6px; font-size:0.75rem; font-weight:600;
                              text-transform:uppercase; color:#92400e">Key Insight</p>
                    <p style="margin:0; font-size:0.85rem; color:#374151; line-height:1.6">
                        {meta['insight']}
                    </p>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Segment × Risk Tier Cross Table</p>',
                unsafe_allow_html=True)
    cross = pd.crosstab(dashboard["Segment"], dashboard["Risk Tier"], margins=True)
    st.dataframe(cross, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — VISUAL INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Visual Insights</p>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        tier_counts = dashboard["Risk Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        tier_order = ["High Risk", "Medium Risk", "Low Risk"]
        tier_counts["Risk Tier"] = pd.Categorical(
            tier_counts["Risk Tier"], categories=tier_order, ordered=True)
        tier_counts = tier_counts.sort_values("Risk Tier")

        fig1 = px.bar(
            tier_counts, x="Risk Tier", y="Count", color="Risk Tier",
            color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
            title="Customer Risk Tier Distribution", text="Count"
        )
        fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig1.update_layout(showlegend=False, plot_bgcolor='white',
                           paper_bgcolor='white', title_font_size=14, height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with row1_col2:
        seg_counts = dashboard["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig2 = px.pie(
            seg_counts, values="Count", names="Segment",
            title="Customer Segment Distribution",
            color_discrete_sequence=["#6366f1", "#ef4444", "#10b981"],
            hole=0.4
        )
        fig2.update_layout(height=350, title_font_size=14, paper_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        churn_by_seg = dashboard.groupby("Segment")["churned"].mean().mul(100).round(1).reset_index()
        churn_by_seg.columns = ["Segment", "Churn Rate %"]

        fig3 = px.bar(
            churn_by_seg.sort_values("Churn Rate %", ascending=True),
            x="Churn Rate %", y="Segment", orientation="h",
            title="Churn Rate by Segment",
            color="Churn Rate %",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            text="Churn Rate %"
        )
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3.update_layout(coloraxis_showscale=False, plot_bgcolor='white',
                           paper_bgcolor='white', title_font_size=14, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        fig4 = px.histogram(
            dashboard, x="churn_probability", nbins=50,
            title="Churn Probability Distribution",
            color_discrete_sequence=["#6366f1"],
            labels={"churn_probability": "Churn Probability"}
        )
        fig4.add_vline(x=0.5, line_dash="dash", line_color="#ef4444",
                       annotation_text="High Risk threshold (0.5)")
        fig4.add_vline(x=0.2, line_dash="dash", line_color="#f59e0b",
                       annotation_text="Medium Risk (0.2)")
        fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           title_font_size=14, height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-header">Flight Activity vs Tenure by Segment</p>',
                unsafe_allow_html=True)

    fig5 = px.scatter(
        dashboard.sample(min(3000, len(dashboard)), random_state=42),
        x="Tenure_Months", y="Total_Flights",
        color="Segment",
        color_discrete_map={
            "High-Frequency New Flyers"  : "#6366f1",
            "Disengaged At-Risk Members" : "#ef4444",
            "Loyal Core Members"         : "#10b981"
        },
        opacity=0.5,
        size="churn_probability",
        size_max=12,
        title="Tenure vs Total Flights (bubble size = churn probability)",
        labels={
            "Tenure_Months" : "Membership Tenure (Months)",
            "Total_Flights" : "Total Flights (2017-2018)"
        }
    )
    fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                       title_font_size=14, height=420)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDED ACTIONS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Retention Playbook — Recommended Actions by Segment</p>',
                unsafe_allow_html=True)

    st.markdown("""
    > Every recommendation below states **who** receives it, **when** it triggers,
    > **what** the offer is, **what channel** it goes through, and
    > **what metric** proves it is working.
    """)

    # Pull live counts from data
    dis  = dashboard[dashboard["Segment"] == "Disengaged At-Risk Members"]
    newf = dashboard[dashboard["Segment"] == "High-Frequency New Flyers"]
    core = dashboard[dashboard["Segment"] == "Loyal Core Members"]

    dis_total  = len(dis);  dis_hr  = int((dis["Risk Tier"]  == "High Risk").sum())
    newf_total = len(newf); newf_hr = int((newf["Risk Tier"] == "High Risk").sum())
    core_total = len(core); core_hr = int((core["Risk Tier"] == "High Risk").sum())

    dis_churn  = dis["churned"].mean() * 100
    newf_churn = newf["churned"].mean() * 100
    core_churn = core["churned"].mean() * 100

    st.markdown(f"""
    <div class="action-card" style="border-color:#ef4444">
        <p class="action-title" style="color:#ef4444">⚠️ Disengaged At-Risk Members</p>
        <p class="action-count">{dis_total:,} members total · {dis_hr:,} High Risk · Churn rate {dis_churn:.1f}%</p>
        <div class="action-row"><span class="action-icon">🎯</span>
            <span><span class="action-label">Trigger:</span>
            Redemption rate crosses 0.75 AND no flight in 6+ months</span></div>
        <div class="action-row"><span class="action-icon">📧</span>
            <span><span class="action-label">High CLV members:</span>
            Personal email from named airline rep + status tier upgrade for 6 months
            if they book one flight within 60 days</span></div>
        <div class="action-row"><span class="action-icon">🤖</span>
            <span><span class="action-label">Low CLV members:</span>
            Automated 3-email sequence over 30 days · Bonus points = one free
            domestic flight if booked within 45 days</span></div>
        <div class="action-row"><span class="action-icon">📊</span>
            <span><span class="action-label">Success metric:</span>
            Re-engagement rate — % of contacted members who book at least one
            flight within 60 days · Target: recover 15% = {int(dis_hr*0.15):,} customers</span></div>
        <div class="action-row"><span class="action-icon">💡</span>
            <span><span class="action-label">Key insight:</span>
            Redemption rate 0.97 is the exit signal — intervene before points
            balance reaches zero while customer still has something to stay for</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="action-card" style="border-color:#6366f1">
        <p class="action-title" style="color:#6366f1">🚀 High-Frequency New Flyers</p>
        <p class="action-count">{newf_total:,} members total · {newf_hr:,} High Risk · Churn rate {newf_churn:.1f}%</p>
        <div class="action-row"><span class="action-icon">🎯</span>
            <span><span class="action-label">Trigger:</span>
            Enrolled 6+ months · 10+ total flights · Zero redemptions ever</span></div>
        <div class="action-row"><span class="action-icon">📧</span>
            <span><span class="action-label">Action:</span>
            Personalised email showing current points balance with concrete translation —
            "You have 45,000 points — that is a free return flight to Vancouver"</span></div>
        <div class="action-row"><span class="action-icon">🎁</span>
            <span><span class="action-label">Offer:</span>
            Bonus 5,000 points if they make their first redemption within 30 days</span></div>
        <div class="action-row"><span class="action-icon">📱</span>
            <span><span class="action-label">Channel:</span>
            Email + in-app notification · Onboarding call for 10+ flight members</span></div>
        <div class="action-row"><span class="action-icon">📊</span>
            <span><span class="action-label">Success metric:</span>
            First redemption rate within 30 days · Target: 60% of {newf_total:,} members
            make their first redemption</span></div>
        <div class="action-row"><span class="action-icon">💡</span>
            <span><span class="action-label">Key insight:</span>
            Redemption rate = 0.00 means zero emotional connection to the program.
            First redemption is the loyalty inflection point — once redeemed,
            churn probability drops sharply</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="action-card" style="border-color:#10b981">
        <p class="action-title" style="color:#10b981">⭐ Loyal Core Members</p>
        <p class="action-count">{core_total:,} members total · {core_hr:,} High Risk · Churn rate {core_churn:.1f}%</p>
        <div class="action-row"><span class="action-icon">🎯</span>
            <span><span class="action-label">Trigger (High Risk {core_hr:,} only):</span>
            No flight in 4+ months — anomaly for this segment, act immediately</span></div>
        <div class="action-row"><span class="action-icon">📧</span>
            <span><span class="action-label">Action for {core_hr:,} High Risk:</span>
            Personalised recognition email acknowledging tenure + complimentary
            tier upgrade for 3 months, no booking required</span></div>
        <div class="action-row"><span class="action-icon">📞</span>
            <span><span class="action-label">Aurora card holders:</span>
            Direct call from customer relations — zero tolerance for losing
            long-tenured Aurora members</span></div>
        <div class="action-row"><span class="action-icon">🎪</span>
            <span><span class="action-label">Broader segment ({core_total:,}):</span>
            Annual loyalty recognition communication + early access to seat sales
            and new routes before general public</span></div>
        <div class="action-row"><span class="action-icon">📊</span>
            <span><span class="action-label">Success metric:</span>
            Churn rate stays below 3% · 80% of {core_hr:,} High Risk re-engage
            within 60 days of outreach</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Business recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">3 Business Recommendations</p>',
                unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:1.3rem;
                    border:1px solid #e2e8f0; height:100%">
            <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.08em; color:#6366f1; margin:0 0 6px">Recommendation 1</p>
            <p style="font-size:0.95rem; font-weight:600; color:#1e293b; margin:0 0 10px">
                Shift to Predictive Retention</p>
            <p style="font-size:0.82rem; color:#374151; line-height:1.6; margin:0">
                Target {high_risk:,} customers ({high_risk/total*100:.0f}%) instead of all {total:,}.
                Model identifies High Risk with 92.3% accuracy.
                Retention costs 5–7× less than acquisition.
            </p>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div style="background:white; border-radius:12px; padding:1.3rem;
                    border:1px solid #e2e8f0; height:100%">
            <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.08em; color:#ef4444; margin:0 0 6px">Recommendation 2</p>
            <p style="font-size:0.95rem; font-weight:600; color:#1e293b; margin:0 0 10px">
                Use Redemption Rate as Early Warning</p>
            <p style="font-size:0.82rem; color:#374151; line-height:1.6; margin:0">
                Alert when redemption rate crosses 0.75 in a rolling 3-month window.
                Disengaged members show a 0.97 rate — intervene before points reach zero.
            </p>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:1.3rem;
                    border:1px solid #e2e8f0; height:100%">
            <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.08em; color:#10b981; margin:0 0 6px">Recommendation 3</p>
            <p style="font-size:0.95rem; font-weight:600; color:#1e293b; margin:0 0 10px">
                Activate New Members Within 6 Months</p>
            <p style="font-size:0.82rem; color:#374151; line-height:1.6; margin:0">
                {newf_hr/newf_total*100:.0f}% of new frequent flyers are already high risk.
                First redemption incentive costs ~CAD 45 per member.
                Applied to {newf_total:,} members = under CAD {int(newf_total*45/1000)*1000:,} total investment.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding:1rem;
                border-top:1px solid #e2e8f0">
        Airline Loyalty Retention Intelligence ·
        Consulting & Analytics Club, IIT Guwahati ·
        Summer Projects 2026 ·
        XGBoost AUC 0.9883 · K-Means Silhouette 0.2931
    </div>
    """, unsafe_allow_html=True)
