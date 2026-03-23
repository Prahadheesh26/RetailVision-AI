import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# MUST BE FIRST
st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

st.title("🛒 Retail AI Dashboard")

df = pd.read_csv("customer_data.csv")

st.write("### Raw Data")
st.dataframe(df)

# KPIs
total_visitors = df["PersonID"].nunique()
total_records = len(df)

st.metric("Total Visitors", total_visitors)
st.metric("Total Records", total_records)

zone_counts = df["Zone"].value_counts()

st.bar_chart(zone_counts)

st.write("### Zone Popularity")

fig, ax = plt.subplots()
zone_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.write("### Insights")

most_zone = zone_counts.idxmax()
st.success(f"Most visited zone is {most_zone}")

time_spent = df.groupby("PersonID")["Frame"].count()

st.write("### Avg Time Spent")
st.write(time_spent.head())

from statsmodels.tsa.arima.model import ARIMA

# Count visitors per frame
footfall = df.groupby("Frame")["PersonID"].nunique()

# Fit model
model = ARIMA(footfall, order=(2,1,2))
model_fit = model.fit()

# Predict next 20 frames
forecast = model_fit.forecast(steps=20)

st.write("### 🔮 Footfall Forecast")
st.line_chart(forecast)

st.write("### 📊 Footfall Trend")
st.line_chart(footfall)

st.write("### 💡 AI Insights")

peak_frame = footfall.idxmax()
peak_value = footfall.max()

st.success(f"Peak customer activity at frame {peak_frame} with {peak_value} visitors")

if peak_value > 5:
    st.warning("High crowd detected – Increase staff during peak hours")
else:
    st.info("Normal customer flow")