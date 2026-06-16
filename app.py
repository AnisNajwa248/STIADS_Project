# ============================================================
# PREDICTIVE MAINTENANCE - STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1f4e79, #2e86c1);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }
    .header-banner h1 {
        color: white;
        font-size: 2.2em;
        margin: 0;
        font-weight: 700;
    }
    .header-banner p {
        color: #aed6f1;
        font-size: 1em;
        margin: 8px 0 0 0;
    }

    /* Section cards */
    .section-card {
        background-color: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Range note boxes */
    .range-note {
        background-color: #1e2a3a;
        border-left: 3px solid #2e86c1;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 0.78em;
        color: #aed6f1;
        margin-top: 2px;
        margin-bottom: 8px;
    }

    /* Result cards */
    .result-card {
        background-color: #1a1d24;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3139;
    }
    .result-value {
        font-size: 2em;
        font-weight: 700;
        margin: 5px 0;
    }
    .result-label {
        color: #7f8c8d;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Threshold table */
    .threshold-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85em;
    }
    .threshold-table th {
        background-color: #1f4e79;
        color: white;
        padding: 8px 12px;
        text-align: left;
    }
    .threshold-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #2d3139;
        color: #ecf0f1;
    }
    .threshold-table tr:hover td {
        background-color: #1e2a3a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model   = joblib.load('rf_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

model, encoder = load_model()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-banner">
    <h1>⚙️ Predictive Maintenance System</h1>
    <p>AI4I 2020 Dataset &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp; 98.85% Accuracy &nbsp;|&nbsp; MAXD 5213 Applied Data Science</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT — Two main columns
# ============================================================
left_col, right_col = st.columns([1.2, 1], gap="large")

# ============================================================
# LEFT COLUMN — Inputs
# ============================================================
with left_col:
    st.markdown("### 🔧 Machine Sensor Readings")
    st.caption("Adjust the sliders to match the machine's current sensor readings, then click Predict.")

    # --- Machine Type ---
    machine_type = st.selectbox(
        "🏭 Machine Type",
        options=["L (Low Quality)", "M (Medium Quality)", "H (High Quality)"],
        help="Machine manufacturing quality grade"
    )
    st.markdown("""<div class='range-note'>
        📌 Low Quality machines fail at 3.92% &nbsp;|&nbsp;
        Medium at 2.77% &nbsp;|&nbsp;
        High Quality at 2.09%
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        torque = st.slider("🔩 Torque (Nm)", 0.0, 80.0, 40.0, 0.1)
        st.markdown("""<div class='range-note'>
            ✅ Safe: &lt; 50 Nm &nbsp;|&nbsp;
            🟡 Moderate: 50–60 Nm &nbsp;|&nbsp;
            🔴 High: &gt; 60 Nm
        </div>""", unsafe_allow_html=True)

        tool_wear = st.slider("🔨 Tool Wear (mins)", 0, 300, 100, 1)
        st.markdown("""<div class='range-note'>
            ✅ Safe: &lt; 150 mins &nbsp;|&nbsp;
            🟡 Elevated: 150–200 mins &nbsp;|&nbsp;
            🔴 Critical: &gt; 200 mins
        </div>""", unsafe_allow_html=True)

        air_temp = st.slider("🌡️ Air Temperature (°C)", 20.0, 35.0, 26.0, 0.1)
        st.markdown("""<div class='range-note'>
            ✅ Safe: &lt; 28°C &nbsp;|&nbsp;
            🟡 Warning: 28–30°C &nbsp;|&nbsp;
            🔴 Critical: &gt; 30°C &nbsp;|&nbsp; Dataset avg: 26.85°C
        </div>""", unsafe_allow_html=True)

    with col2:
        rpm = st.slider("⚙️ Rotational Speed (RPM)", 1000, 3000, 1500, 10)
        st.markdown("""<div class='range-note'>
            📌 Normal range: 1168–2886 RPM &nbsp;|&nbsp;
            Abnormal RPM linked to Power Failure
        </div>""", unsafe_allow_html=True)

        process_temp = st.slider("🌡️ Process Temperature (°C)", 30.0, 50.0, 36.0, 0.1)
        st.markdown("""<div class='range-note'>
            ✅ Safe: &lt; 38°C &nbsp;|&nbsp;
            🟡 Warning: 38–42°C &nbsp;|&nbsp;
            🔴 Critical: &gt; 42°C &nbsp;|&nbsp; High temp increases thermal stress
        </div>""", unsafe_allow_html=True)

        temp_diff = process_temp - air_temp
        st.metric("🔁 Temp Difference (auto)", f"{temp_diff:.1f} °C",
                  help="Process Temp minus Air Temp — calculated automatically")
        st.markdown("""<div class='range-note'>
            ✅ Safe: &lt; 10°C &nbsp;|&nbsp;
            🟡 Warning: 10–12°C &nbsp;|&nbsp;
            🔴 Critical: &gt; 12°C &nbsp;|&nbsp; Large difference indicates thermal stress
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- Predict Button ---
    predict_btn = st.button("🔍 PREDICT FAILURE RISK",
                            use_container_width=True,
                            type="primary")

# ============================================================
# RIGHT COLUMN — Info + Results
# ============================================================
with right_col:

    # --- Sensor Threshold Reference ---
    st.markdown("### 📋 Sensor Threshold Reference")
    st.markdown("""
    <table class='threshold-table'>
        <tr>
            <th>Sensor</th>
            <th>Safe</th>
            <th>Warning</th>
            <th>Critical</th>
        </tr>
        <tr>
            <td>Torque</td>
            <td>🟢 &lt; 50 Nm</td>
            <td>🟡 50–60 Nm</td>
            <td>🔴 &gt; 60 Nm</td>
        </tr>
        <tr>
            <td>Tool Wear</td>
            <td>🟢 &lt; 150 mins</td>
            <td>🟡 150–200 mins</td>
            <td>🔴 &gt; 200 mins</td>
        </tr>
        <tr>
            <td>RPM</td>
            <td>🟢 1168–2886</td>
            <td>🟡 Outside range</td>
            <td>🔴 Extreme values</td>
        </tr>
        <tr>
            <td>Air Temp</td>
            <td>🟢 &lt; 28°C</td>
            <td>🟡 28–30°C</td>
            <td>🔴 &gt; 30°C</td>
        </tr>
        <tr>
            <td>Process Temp</td>
            <td>🟢 &lt; 38°C</td>
            <td>🟡 38–42°C</td>
            <td>🔴 &gt; 42°C</td>
        </tr>
        <tr>
            <td>Temp Diff</td>
            <td>🟢 &lt; 10°C</td>
            <td>🟡 10–12°C</td>
            <td>🔴 &gt; 12°C</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # PREDICTION RESULTS
    # ============================================================
    if predict_btn:

        # --- Prepare input ---
        type_map  = {"L (Low Quality)": "L",
                     "M (Medium Quality)": "M",
                     "H (High Quality)": "H"}
        type_code = type_map[machine_type]
        type_enc  = encoder.transform([type_code])[0]
        power     = torque * rpm * (2 * np.pi / 60)

        input_data = pd.DataFrame([[
            type_enc, air_temp, process_temp,
            rpm, torque, tool_wear,
            temp_diff, power
        ]], columns=[
            'Type_enc', 'Air_Temp_C', 'Process_Temp_C',
            'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'Temp_Diff', 'Power'
        ])

        # --- Predict ---
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # --- Risk Tier ---
        if tool_wear > 200 or torque > 60:
            risk_tier  = "🔴 High Risk"
            risk_color = "#e74c3c"
        elif tool_wear > 150 or torque > 50:
            risk_tier  = "🟡 Medium Risk"
            risk_color = "#f39c12"
        else:
            risk_tier  = "🟢 Low Risk"
            risk_color = "#2ecc71"

        # --- Probability bar color ---
        if probability >= 60:
            prob_color = "#e74c3c"
        elif probability >= 30:
            prob_color = "#f39c12"
        else:
            prob_color = "#2ecc71"

        st.markdown("### 📊 Prediction Result")

        # --- Main result banner ---
        if prediction == 1:
            st.error("❌ MACHINE FAILURE DETECTED — Immediate action required!")
        else:
            st.success("✅ MACHINE IS HEALTHY — Operating within normal parameters")

        # --- 3 metric cards ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-label'>Failure Probability</div>
                <div class='result-value' style='color:{prob_color}'>{probability:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-label'>Risk Tier</div>
                <div class='result-value' style='font-size:1.2em'>{risk_tier}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            status = "Failed" if prediction == 1 else "Healthy"
            s_color = "#e74c3c" if prediction == 1 else "#2ecc71"
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-label'>Status</div>
                <div class='result-value' style='color:{s_color}'>{status}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Sensor Summary Table ---
        st.markdown("#### 📋 Sensor Reading Summary")
        summary = pd.DataFrame({
            "Sensor"  : ["Machine Type", "Torque", "Tool Wear",
                         "Rotational Speed", "Air Temp",
                         "Process Temp", "Temp Diff"],
            "Value"   : [machine_type,
                         f"{torque} Nm", f"{tool_wear} mins",
                         f"{rpm} RPM", f"{air_temp} °C",
                         f"{process_temp} °C", f"{temp_diff:.1f} °C"],
            "Status"  : [
                "⚠️ Low Quality"    if type_code == "L" else
                "🟡 Medium Quality" if type_code == "M" else "✅ High Quality",
                "🔴 High"    if torque > 60 else
                "🟡 Moderate" if torque > 50 else "✅ Normal",
                "🔴 Critical"  if tool_wear > 200 else
                "🟡 Elevated"  if tool_wear > 150 else "✅ Normal",
                "✅ Normal"    if 1168 < rpm < 2886 else "⚠️ Abnormal",
                "✅ Normal"    if air_temp <= 28 else
                "🟡 Warning"   if air_temp <= 30 else "🔴 Critical",
                "✅ Normal"    if process_temp <= 38 else
                "🟡 Warning"   if process_temp <= 42 else "🔴 Critical",
                "✅ Normal"    if temp_diff <= 10 else
                "🟡 Warning"   if temp_diff <= 12 else "🔴 Critical"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.divider()

        # --- Recommendation ---
        st.markdown("#### 🛠️ Maintenance Recommendation")
        if prediction == 1:
            st.error("""
            **Immediate Action Required:**
            - 🔴 Stop machine operation immediately
            - 🔧 Schedule emergency maintenance within 24 hours
            - 🔩 Inspect and replace worn tools
            - 🌡️ Check cooling system for heat dissipation issues
            - 📋 Log failure details for future analysis
            """)
        elif risk_tier == "🔴 High Risk":
            st.warning("""
            **Priority Inspection Within 48 Hours:**
            - 🟡 Machine approaching failure threshold
            - 🔧 Schedule preventive maintenance soon
            - 🔩 Monitor tool wear — approaching critical level
            - 📊 Increase sensor monitoring frequency
            """)
        elif risk_tier == "🟡 Medium Risk":
            st.warning("""
            **Increased Monitoring Recommended:**
            - 🟡 Elevated risk indicators detected
            - 📊 Increase monitoring frequency
            - 🔧 Plan maintenance in next scheduled window
            """)
        else:
            st.success("""
            **Machine Operating Normally:**
            - 🟢 All sensors within safe range
            - 📊 Continue standard monitoring schedule
            - 🔧 Proceed with routine maintenance plan
            """)
    else:
        st.info("👈 Enter sensor readings on the left and click **PREDICT FAILURE RISK** to see results.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("MAXD 5213 Special Topics in Applied Data Science | AI4I 2020 Predictive Maintenance Dataset | Random Forest Classifier")
