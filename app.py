import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import plotly.express as px

# ---------------------------------------------------------
# TASK 1: SENSOR ANALYSIS & HEAT INDEX CALCULATION
# ---------------------------------------------------------
FIXED_THRESHOLD = 35.0  # Base temperature threshold in °C
CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad"]

def calculate_heat_index(temp, humidity):
    """Novelty 1: Calculates feel-like heat stress based on humidity"""
    return temp + 0.55 * (1 - (humidity / 100.0)) * (temp - 14.5)

st.set_page_config(page_title="Smart City Irrigation Research", layout="wide")
st.title("🏙️ Smart City Automated Heat Mitigation & Irrigation System")

# ---------------------------------------------------------
# TASK 2: DATA GENERATION & GATHERING
# ---------------------------------------------------------
@st.cache_data
def generate_sensor_data(samples=1000):
    np.random.seed(42)
    data = []
    for _ in range(samples):
        city = np.random.choice(CITIES)
        temp = np.round(np.random.uniform(20.0, 45.0), 1)
        humidity = np.round(np.random.uniform(30.0, 85.0), 1)
        heat_index = np.round(calculate_heat_index(temp, humidity), 1)
        # Rule: Trigger = 1 if Temp > 35°C OR Heat Index > 37°C
        trigger = 1 if (temp > FIXED_THRESHOLD or heat_index > 37.0) else 0
        data.append([city, temp, humidity, heat_index, trigger])
    return pd.DataFrame(data, columns=["City", "Temperature", "Humidity", "Heat_Index", "Trigger_Action"])

dataset = generate_sensor_data()

# ---------------------------------------------------------
# TASK 3: DATA PROCESSING BY ML (DECISION TREE)
# ---------------------------------------------------------
X = dataset[["Temperature", "Humidity", "Heat_Index"]]
y = dataset["Trigger_Action"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# TASK 4: TRIGGERING & PERFORMANCE EVALUATION
# ---------------------------------------------------------
st.header("📊 Task 4: Machine Learning Model Evaluation")
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
col2.metric("Precision Score", f"{precision_score(y_test, y_pred) * 100:.2f}%")
col3.metric("Recall Score", f"{recall_score(y_test, y_pred) * 100:.2f}%")

with st.expander("View Confusion Matrix"):
    st.write(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# TASK 5: ACTION OUTCOMES & REAL-TIME DASHBOARD
# ---------------------------------------------------------
st.divider()
st.header("🚿 Task 5: Live City Monitoring & Dynamic Water Allocation")

if st.button("🔄 Fetch Live Multi-City Telemetry"):
    results = []
    city_cols = st.columns(3)
    
    for idx, city in enumerate(CITIES):
        live_temp = np.round(np.random.uniform(25.0, 42.0), 1)
        live_hum = np.round(np.random.uniform(35.0, 80.0), 1)
        live_hi = np.round(calculate_heat_index(live_temp, live_hum), 1)
        
        prediction = model.predict([[live_temp, live_hum, live_hi]])[0]
        
        if prediction == 1:
            priority = "HIGH" if live_hi > 40.0 else "MEDIUM"
            water_duration = "15 Mins" if priority == "HIGH" else "8 Mins"
            status = "💦 ACTIVE"
        else:
            priority = "LOW"
            water_duration = "0 Mins"
            status = "🟢 NORMAL (OFF)"
            
        with city_cols[idx % 3]:
            st.metric(label=f"City: {city}", value=f"{live_temp} °C", delta=status, delta_color="off" if prediction==1 else "normal")
            st.caption(f"Humidity: {live_hum}% | Heat Index: {live_hi}°C")
            st.caption(f"Cooling Priority: {priority} ({water_duration})")
        
        results.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "City": city,
            "Temp (°C)": live_temp,
            "Humidity (%)": live_hum,
            "Heat Index (°C)": live_hi,
            "Status": status,
            "Priority": priority,
            "Spray Duration": water_duration
        })
        
    st.subheader("📝 Action Logs")
    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    
    fig = px.bar(
        res_df, x="City", y="Heat Index (°C)", color="Priority",
        title="City Heat Index vs Priority Sprinkler Allocation",
        color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    )
    fig.add_hline(y=FIXED_THRESHOLD, line_dash="dot", annotation_text="Base Threshold (35°C)")
    st.plotly_chart(fig, use_container_width=True)