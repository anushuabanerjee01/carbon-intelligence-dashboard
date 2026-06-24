"""
Carbon Intelligence Dashboard
Author: Anushua Banerjee
MS Business Analytics | CSUEB
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Intelligence Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a2f 0%, #0d2a1f 100%);
        border: 1px solid #2d6a4f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #52b788; }
    .metric-label { font-size: 0.85rem; color: #95d5b2; margin-top: 4px; }
    .metric-delta { font-size: 0.9rem; margin-top: 6px; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #52b788;
        border-left: 4px solid #52b788;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        background: #1a2e1a;
    }
    .stTabs [aria-selected="true"] { background: #2d6a4f !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
VEHICLE_TYPES   = ['Electric', 'Hybrid', 'Motorcycle', 'Petrol', 'Diesel', 'Public Transport']
BUILDING_TYPES  = ['Apartment', 'Commercial', 'House', 'Office']
FEATURES        = [
    'Energy_Usage_kWh', 'Transportation_Distance_km', 'Smart_Appliance_Usage_hours',
    'Renewable_Energy_Usage_percent', 'Temperature_C', 'Humidity_percent',
    'Vehicle_Type_Encoded', 'Building_Type_Encoded',
    'Effective_Energy_Usage', 'Distance_Vehicle_Interaction'
]
FEATURE_LABELS  = {
    'Energy_Usage_kWh': 'Energy Usage',
    'Transportation_Distance_km': 'Transport Distance',
    'Smart_Appliance_Usage_hours': 'Smart Appliance Usage',
    'Renewable_Energy_Usage_percent': 'Renewable Energy %',
    'Temperature_C': 'Temperature',
    'Humidity_percent': 'Humidity',
    'Vehicle_Type_Encoded': 'Vehicle Type',
    'Building_Type_Encoded': 'Building Type',
    'Effective_Energy_Usage': 'Effective Energy Usage',
    'Distance_Vehicle_Interaction': 'Distance × Vehicle'
}

# ── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n=10000, seed=42):
    """Generate synthetic dataset matching original notebook distribution."""
    rng = np.random.default_rng(seed)
    vehicle_types  = rng.choice(VEHICLE_TYPES, n)
    building_types = rng.choice(BUILDING_TYPES, n)

    vehicle_enc = LabelEncoder().fit(VEHICLE_TYPES)
    building_enc = LabelEncoder().fit(BUILDING_TYPES)

    energy    = rng.uniform(50, 500, n)
    transport = rng.uniform(5, 200, n)
    appliance = rng.uniform(0, 12, n)
    renewable = rng.uniform(0, 100, n)
    temp      = rng.uniform(-5, 40, n)
    humidity  = rng.uniform(20, 90, n)
    veh_enc   = vehicle_enc.transform(vehicle_types).astype(float)
    bld_enc   = building_enc.transform(building_types).astype(float)
    eff_energy = energy * (1 - renewable / 100)
    dist_veh   = transport * veh_enc

    carbon = (
        0.45 * energy
        + 0.30 * transport
        + 0.08 * appliance
        - 0.12 * renewable
        + 0.05 * temp
        + 0.03 * humidity
        + 2.5  * veh_enc
        + 1.2  * bld_enc
        + 0.15 * eff_energy
        + 0.02 * dist_veh
        + rng.normal(0, 8, n)
    )

    return pd.DataFrame({
        'Energy_Usage_kWh': energy,
        'Transportation_Distance_km': transport,
        'Smart_Appliance_Usage_hours': appliance,
        'Renewable_Energy_Usage_percent': renewable,
        'Temperature_C': temp,
        'Humidity_percent': humidity,
        'Vehicle_Type': vehicle_types,
        'Building_Type': building_types,
        'Vehicle_Type_Encoded': veh_enc,
        'Building_Type_Encoded': bld_enc,
        'Effective_Energy_Usage': eff_energy,
        'Distance_Vehicle_Interaction': dist_veh,
        'Carbon_Emission_kgCO2': carbon,
    })

@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df['Carbon_Emission_kgCO2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42).fit(X_train, y_train)

    metrics = {
        'lr': {
            'r2':  r2_score(y_test, lr.predict(X_test)),
            'mae': mean_absolute_error(y_test, lr.predict(X_test)),
            'rmse': np.sqrt(mean_squared_error(y_test, lr.predict(X_test))),
        },
        'rf': {
            'r2':  r2_score(y_test, rf.predict(X_test)),
            'mae': mean_absolute_error(y_test, rf.predict(X_test)),
            'rmse': np.sqrt(mean_squared_error(y_test, rf.predict(X_test))),
        },
    }
    importance = pd.DataFrame({
        'feature': [FEATURE_LABELS.get(f, f) for f in FEATURES],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return lr, rf, metrics, importance, X_test, y_test

@st.cache_resource
def get_encoders():
    ve = LabelEncoder().fit(VEHICLE_TYPES)
    be = LabelEncoder().fit(BUILDING_TYPES)
    return ve, be

def emission_category(val):
    if val < 50:   return "🟢 Low",   "#52b788"
    if val < 100:  return "🟡 Medium", "#ffd166"
    if val < 150:  return "🟠 High",   "#f4a261"
    return "🔴 Very High", "#ef233c"

# ── Load everything ──────────────────────────────────────────────────────────
df                              = generate_synthetic_data()
lr, rf, metrics, importance, X_test, y_test = train_models(df)
vehicle_encoder, building_encoder           = get_encoders()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/leaf.png", width=60)
    st.title("🌿 Carbon Intelligence")
    st.caption("Predict · Simulate · Act")
    st.divider()

    st.subheader("⚙️ Input Parameters")
    energy     = st.slider("⚡ Energy Usage (kWh)",           50.0,  500.0, 250.0, 10.0)
    transport  = st.slider("🚗 Transport Distance (km)",       5.0,  200.0,  80.0,  5.0)
    appliance  = st.slider("🏠 Smart Appliance Usage (hrs)",   0.0,   12.0,   4.0,  0.5)
    renewable  = st.slider("☀️ Renewable Energy (%)",          0.0,  100.0,  30.0,  5.0)
    temp       = st.slider("🌡️ Temperature (°C)",             -5.0,   40.0,  22.0,  1.0)
    humidity   = st.slider("💧 Humidity (%)",                  20.0,   90.0,  55.0,  5.0)
    vehicle    = st.selectbox("🚙 Vehicle Type",   VEHICLE_TYPES)
    building   = st.selectbox("🏢 Building Type",  BUILDING_TYPES)

    st.divider()
    model_choice = st.radio("🤖 Model", ["Random Forest", "Linear Regression"], index=0)

# ── Prediction ───────────────────────────────────────────────────────────────
veh_enc_val = float(vehicle_encoder.transform([vehicle])[0])
bld_enc_val = float(building_encoder.transform([building])[0])
eff_energy  = energy * (1 - renewable / 100)
dist_veh    = transport * veh_enc_val

input_row = pd.DataFrame([{
    'Energy_Usage_kWh':             energy,
    'Transportation_Distance_km':   transport,
    'Smart_Appliance_Usage_hours':  appliance,
    'Renewable_Energy_Usage_percent': renewable,
    'Temperature_C':                temp,
    'Humidity_percent':             humidity,
    'Vehicle_Type_Encoded':         veh_enc_val,
    'Building_Type_Encoded':        bld_enc_val,
    'Effective_Energy_Usage':       eff_energy,
    'Distance_Vehicle_Interaction': dist_veh,
}])

model      = rf if model_choice == "Random Forest" else lr
prediction = float(model.predict(input_row)[0])
cat_label, cat_color = emission_category(prediction)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🌿 Carbon Intelligence Dashboard")
st.caption("IoT-powered carbon emission prediction and sustainability analytics")
st.divider()

# ── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prediction:.1f}</div>
        <div class="metric-label">Predicted Emissions (kgCO₂)</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{cat_color}">{cat_label}</div>
        <div class="metric-label">Emission Category</div>
    </div>""", unsafe_allow_html=True)

avg_emission  = df['Carbon_Emission_kgCO2'].mean()
delta_vs_avg  = prediction - avg_emission
delta_symbol  = "▲" if delta_vs_avg > 0 else "▼"
delta_color   = "#ef233c" if delta_vs_avg > 0 else "#52b788"

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{delta_color}">{delta_symbol} {abs(delta_vs_avg):.1f}</div>
        <div class="metric-label">vs. Dataset Average ({avg_emission:.1f} kgCO₂)</div>
    </div>""", unsafe_allow_html=True)

with k4:
    best_r2 = max(metrics['lr']['r2'], metrics['rf']['r2'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{best_r2:.3f}</div>
        <div class="metric-label">Best Model R² Score</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Prediction", "🔬 Scenario Simulator", "📈 Insights"])

# ─── TAB 1: OVERVIEW ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Avg Emission",   f"{avg_emission:.1f} kgCO₂")
    c3.metric("Max Emission",   f"{df['Carbon_Emission_kgCO2'].max():.1f} kgCO₂")
    c4.metric("Min Emission",   f"{df['Carbon_Emission_kgCO2'].min():.1f} kgCO₂")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Emission Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Carbon_Emission_kgCO2', nbins=60,
                           color_discrete_sequence=['#52b788'],
                           labels={'Carbon_Emission_kgCO2': 'Carbon Emission (kgCO₂)'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#c9d6df', showlegend=False)
        fig.update_traces(marker_line_color='#2d6a4f', marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Emissions by Vehicle Type</div>', unsafe_allow_html=True)
        vehicle_avg = df.groupby('Vehicle_Type')['Carbon_Emission_kgCO2'].mean().reset_index()
        vehicle_avg.columns = ['Vehicle Type', 'Avg Emission (kgCO₂)']
        fig2 = px.bar(vehicle_avg.sort_values('Avg Emission (kgCO₂)', ascending=True),
                      x='Avg Emission (kgCO₂)', y='Vehicle Type', orientation='h',
                      color='Avg Emission (kgCO₂)', color_continuous_scale='Greens')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#c9d6df', coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = ['Energy_Usage_kWh', 'Transportation_Distance_km', 'Smart_Appliance_Usage_hours',
                 'Renewable_Energy_Usage_percent', 'Temperature_C', 'Humidity_percent',
                 'Carbon_Emission_kgCO2']
    corr = df[corr_cols].corr()
    labels = ['Energy', 'Transport', 'Appliance', 'Renewable%', 'Temp', 'Humidity', 'Carbon Emission']
    fig3 = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='#c9d6df', height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ─── TAB 2: PREDICTION ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([1, 1])

    with p1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={'reference': avg_emission, 'valueformat': '.1f'},
            title={'text': f"Predicted Emission (kgCO₂)<br><span style='font-size:0.8em;color:{cat_color}'>{cat_label}</span>"},
            gauge={
                'axis': {'range': [0, 300]},
                'bar': {'color': cat_color},
                'steps': [
                    {'range': [0, 50],   'color': '#1b4332'},
                    {'range': [50, 100], 'color': '#2d6a4f'},
                    {'range': [100, 150],'color': '#b7791f'},
                    {'range': [150, 300],'color': '#7f1d1d'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 3}, 'value': avg_emission},
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#c9d6df', height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with p2:
        st.markdown("**📋 Your Input Summary**")
        input_summary = pd.DataFrame({
            'Parameter': ['Energy Usage', 'Transport Distance', 'Smart Appliance', 'Renewable Energy',
                          'Temperature', 'Humidity', 'Vehicle Type', 'Building Type'],
            'Value': [f"{energy:.0f} kWh", f"{transport:.0f} km", f"{appliance:.1f} hrs",
                      f"{renewable:.0f}%", f"{temp:.1f}°C", f"{humidity:.0f}%",
                      vehicle, building]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LR R²",   f"{metrics['lr']['r2']:.3f}")
    m2.metric("LR MAE",  f"{metrics['lr']['mae']:.2f}")
    m3.metric("LR RMSE", f"{metrics['lr']['rmse']:.2f}")
    m4.metric("RF R²",   f"{metrics['rf']['r2']:.3f}")
    m5.metric("RF MAE",  f"{metrics['rf']['mae']:.2f}")
    m6.metric("RF RMSE", f"{metrics['rf']['rmse']:.2f}")

    st.markdown('<div class="section-header">Actual vs Predicted (Random Forest)</div>', unsafe_allow_html=True)
    y_pred_test = rf.predict(X_test)
    sample_n = min(300, len(y_test))
    idx = np.random.choice(len(y_test), sample_n, replace=False)
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test.iloc[idx], y=y_pred_test[idx],
                                  mode='markers', marker=dict(color='#52b788', opacity=0.5, size=5),
                                  name='Predictions'))
    mn, mx = y_test.min(), y_test.max()
    fig_avp.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                  line=dict(color='#ef233c', dash='dash', width=2), name='Perfect fit'))
    fig_avp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#c9d6df',
                           xaxis_title='Actual (kgCO₂)', yaxis_title='Predicted (kgCO₂)', height=380)
    st.plotly_chart(fig_avp, use_container_width=True)

# ─── TAB 3: SCENARIO SIMULATOR ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">What-If Scenario Analysis</div>', unsafe_allow_html=True)
    st.caption("Simulate how behavioral changes reduce your carbon footprint.")

    s1, s2, s3 = st.columns(3)
    with s1:
        reduce_energy    = st.slider("⚡ Reduce Energy by (%)",      0, 80, 20, 5)
    with s2:
        reduce_transport = st.slider("🚗 Reduce Transport by (%)",   0, 80, 20, 5)
    with s3:
        increase_renew   = st.slider("☀️ Increase Renewable to (%)", renewable, 100.0, min(renewable + 20, 100.0), 5.0)

    # Scenario prediction
    s_energy    = energy    * (1 - reduce_energy    / 100)
    s_transport = transport * (1 - reduce_transport / 100)
    s_renewable = increase_renew
    s_eff       = s_energy * (1 - s_renewable / 100)
    s_dist_veh  = s_transport * veh_enc_val

    scenario_row = pd.DataFrame([{
        'Energy_Usage_kWh':             s_energy,
        'Transportation_Distance_km':   s_transport,
        'Smart_Appliance_Usage_hours':  appliance,
        'Renewable_Energy_Usage_percent': s_renewable,
        'Temperature_C':                temp,
        'Humidity_percent':             humidity,
        'Vehicle_Type_Encoded':         veh_enc_val,
        'Building_Type_Encoded':        bld_enc_val,
        'Effective_Energy_Usage':       s_eff,
        'Distance_Vehicle_Interaction': s_dist_veh,
    }])

    scenario_pred = float(model.predict(scenario_row)[0])
    reduction     = prediction - scenario_pred
    pct_reduction = (reduction / prediction) * 100 if prediction > 0 else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Current Emissions",  f"{prediction:.1f} kgCO₂")
    sc2.metric("Scenario Emissions", f"{scenario_pred:.1f} kgCO₂", delta=f"-{reduction:.1f} kgCO₂")
    sc3.metric("Reduction",          f"{pct_reduction:.1f}%")

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        name="Emission Impact",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Current", f"Energy −{reduce_energy}%", f"Transport −{reduce_transport}%",
           f"Renewable +{increase_renew-renewable:.0f}%", "Scenario"],
        y=[prediction,
           -(prediction - float(rf.predict(pd.DataFrame([{**input_row.iloc[0].to_dict(),
               'Energy_Usage_kWh': s_energy,
               'Effective_Energy_Usage': s_energy*(1-renewable/100)}]))[0])),
           -(prediction - float(rf.predict(pd.DataFrame([{**input_row.iloc[0].to_dict(),
               'Transportation_Distance_km': s_transport,
               'Distance_Vehicle_Interaction': s_transport*veh_enc_val}]))[0])),
           -(prediction - float(rf.predict(pd.DataFrame([{**input_row.iloc[0].to_dict(),
               'Renewable_Energy_Usage_percent': s_renewable,
               'Effective_Energy_Usage': energy*(1-s_renewable/100)}]))[0])),
           scenario_pred],
        connector={"line": {"color": "#2d6a4f"}},
        increasing={"marker": {"color": "#ef233c"}},
        decreasing={"marker": {"color": "#52b788"}},
        totals={"marker": {"color": "#ffd166"}},
    ))
    fig_wf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#c9d6df', height=400,
                          title="Emission Reduction Waterfall")
    st.plotly_chart(fig_wf, use_container_width=True)

    # Recommendations
    st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if energy > 300:
        recs.append("🔌 Your energy usage is high. Switch to LED lighting and energy-efficient appliances to cut 15–25%.")
    if transport > 100 and vehicle in ['Petrol', 'Diesel']:
        recs.append("🚌 Consider switching to public transport or an EV — this could reduce transport emissions by 40–60%.")
    if renewable < 30:
        recs.append("☀️ Increasing your renewable energy share to 50%+ can significantly lower your effective carbon footprint.")
    if appliance > 8:
        recs.append("🏠 Smart appliance usage above 8 hrs/day adds up. Schedule high-energy tasks during off-peak hours.")
    if not recs:
        recs.append("✅ You're doing great! Your inputs suggest a relatively low carbon profile. Keep it up!")

    for r in recs:
        st.info(r)

# ─── TAB 4: INSIGHTS ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

    fig_imp = px.bar(importance, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Greens',
                     labels={'importance': 'Importance Score', 'feature': 'Feature'})
    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#c9d6df', coloraxis_showscale=False,
                           yaxis={'categoryorder': 'total ascending'}, height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

    i1, i2 = st.columns(2)

    with i1:
        st.markdown('<div class="section-header">Energy Usage vs Emissions</div>', unsafe_allow_html=True)
        sample = df.sample(500, random_state=1)
        fig_sc = px.scatter(sample, x='Energy_Usage_kWh', y='Carbon_Emission_kgCO2',
                            color='Vehicle_Type', opacity=0.6,
                            labels={'Energy_Usage_kWh': 'Energy Usage (kWh)',
                                    'Carbon_Emission_kgCO2': 'Carbon Emission (kgCO₂)'})
        fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#c9d6df', height=350)
        st.plotly_chart(fig_sc, use_container_width=True)

    with i2:
        st.markdown('<div class="section-header">Renewable Energy Impact</div>', unsafe_allow_html=True)
        fig_ren = px.scatter(sample, x='Renewable_Energy_Usage_percent', y='Carbon_Emission_kgCO2',
                             color='Building_Type', opacity=0.6,
                             labels={'Renewable_Energy_Usage_percent': 'Renewable Energy (%)',
                                     'Carbon_Emission_kgCO2': 'Carbon Emission (kgCO₂)'})
        fig_ren.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#c9d6df', height=350)
        st.plotly_chart(fig_ren, use_container_width=True)

    st.markdown('<div class="section-header">Emissions by Building Type</div>', unsafe_allow_html=True)
    fig_box = px.box(df, x='Building_Type', y='Carbon_Emission_kgCO2',
                     color='Building_Type',
                     labels={'Carbon_Emission_kgCO2': 'Carbon Emission (kgCO₂)', 'Building_Type': ''})
    fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#c9d6df', showlegend=False, height=350)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built by **Anushua Banerjee** · MS Business Analytics, CSUEB · [LinkedIn](https://linkedin.com/in/anushuabanerjee) · Powered by Streamlit + Scikit-learn + Plotly")
