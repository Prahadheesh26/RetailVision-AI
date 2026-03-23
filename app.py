import streamlit as st
import pandas as pd


st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

st.title("🛒 Retail AI Dashboard")

df = pd.read_csv("customer_data.csv")

st.write("### Raw Data")
st.dataframe(df)


total_visitors = df["PersonID"].nunique()
total_records = len(df)

st.metric("Total Visitors", total_visitors)
st.metric("Total Records", total_records)

zone_counts = df["Zone"].value_counts()

st.bar_chart(zone_counts)

st.write("### Zone Popularity")

st.write("### Insights")

most_zone = zone_counts.idxmax()
st.success(f"Most visited zone is {most_zone}")

time_spent = df.groupby("PersonID")["Frame"].count()

st.write("### Avg Time Spent")
st.write(time_spent.head())


# Count visitors per frame
footfall = df.groupby("Frame")["PersonID"].nunique()

st.write("### 📊 Footfall Trend")
st.line_chart(footfall)

# 🔮 Simple Forecast (Moving Average Method)
forecast_steps = 20

# Calculate rolling mean
rolling_mean = footfall.rolling(window=5).mean()

# Take last value and extend
last_value = rolling_mean.dropna().iloc[-1]

forecast = [last_value] * forecast_steps

# Create future index
future_index = range(footfall.index.max()+1, footfall.index.max()+1+forecast_steps)

forecast_series = pd.Series(forecast, index=future_index)

st.write("### 🔮 Footfall Forecast")
st.line_chart(forecast_series)
st.write("### 💡 AI Insights")

peak_frame = footfall.idxmax()
peak_value = footfall.max()

st.success(f"Peak customer activity at frame {peak_frame} with {peak_value} visitors")

if peak_value > 5:
    st.warning("High crowd detected – Increase staff during peak hours")
else:
    st.info("Normal customer flow")