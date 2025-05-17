import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

# ------------------------ Title ------------------------
st.title("🌫️ Real-Time Air Quality Dashboard")

# ------------------------ Load Data ------------------------
df = pd.read_csv("final_dataset_air.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

# ------------------------ Preprocessing ------------------------
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = df[['NowCast Conc.', 'Raw Conc.']].values
y = df[['AQI']].values

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ------------------------ AQI Trend ------------------------
st.subheader("📈 AQI Trend Over Time")
fig1 = px.line(df.sort_values("Date"), x='Date', y='AQI', title='AQI Over Time')
st.plotly_chart(fig1)

# ------------------------ Control Chart ------------------------
st.subheader("📏 AQI Control Chart")
aqi = df['AQI']
mean = aqi.mean()
std = aqi.std()
ucl = mean + 3 * std
lcl = mean - 3 * std

fig2, ax = plt.subplots()
ax.plot(df['Date'], aqi, label="AQI")
ax.axhline(mean, color='green', linestyle='--', label="Mean")
ax.axhline(ucl, color='red', linestyle='--', label="UCL")
ax.axhline(lcl, color='red', linestyle='--', label="LCL")
ax.legend()
st.pyplot(fig2)

# ------------------------ User Input ------------------------
st.subheader("🎯 Predict AQI")

model_option = st.selectbox("Choose a Model:", ["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "RNN", "CNN"])


nowcast = st.slider("NowCast Conc.", float(df['NowCast Conc.'].min()), float(df['NowCast Conc.'].max()), 100.0)
raw = st.slider("Raw Conc.", float(df['Raw Conc.'].min()), float(df['Raw Conc.'].max()), 100.0)

# ------------------------ Prepare Sequence ------------------------
X_seq = X_scaled[-23:]  # last 23 for sequence
new_input_scaled = scaler_x.transform([[nowcast, raw]])
seq_input = np.vstack([X_seq, new_input_scaled]).reshape(1, 24, 2)

# ------------------------ Load Model & Predict ------------------------
model_path = {
    "LSTM": "model/lstm_model.h5",
    "GRU": "model/gru_model.h5",
    "BiLSTM": "model/bilstm_model.h5",
    "CNN-LSTM": "model/cnn_lstm_model.h5",
    "RNN": "model/rnn_model.h5",
    "CNN": "model/cnn_model.h5"

}

try:
    model = load_model(model_path[model_option], compile=False)
    y_pred_scaled = model.predict(seq_input)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
except Exception as e:
    st.error(f"Error loading model or prediction failed: {e}")
    y_pred = 0.8 * nowcast + 0.2 * raw  # fallback dummy formula
    st.info("🔁 Fallback formula used.")

# ------------------------ Output ------------------------
st.success(f"🔮 Predicted AQI: {y_pred:.2f}")

# ------------------------ Category ------------------------
def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

category = get_category(y_pred)
st.markdown(f"### 🟡 AQI Category: `{category}`")

# ------------------------ AQI Category Distribution ------------------------
if 'AQI Category' in df.columns:
    st.subheader("📊 AQI Category Distribution")
    st.bar_chart(df['AQI Category'].value_counts())

# ------------------------ AQI Heatmap ------------------------
st.subheader("🔥 Hourly AQI Heatmap")
if 'Hour' not in df.columns:
    df['Hour'] = df['Date'].dt.hour

pivot = df.pivot_table(index=df['Date'].dt.date, columns='Hour', values='AQI', aggfunc='mean')
fig3 = px.imshow(pivot, labels=dict(x="Hour", y="Date", color="AQI"), aspect="auto")
st.plotly_chart(fig3)
