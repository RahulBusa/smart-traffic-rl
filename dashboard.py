import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Smart Traffic Dashboard", layout="centered")
st.title("🚦 Smart Traffic – Live Status")

chart_placeholder = st.empty()
info_placeholder = st.empty()

while True:
    try:
        data = requests.get("http://127.0.0.1:5000/status", timeout=1).json()

        counts = data["counts"]
        green = data["green_lane"]
        remaining = data["green_remaining"]

        df = pd.DataFrame({
            "Road": ["North", "East", "South", "West"],
            "Vehicles": [counts["N"], counts["E"], counts["S"], counts["W"]]
        })

        with info_placeholder.container():
            st.metric("🟢 Green Lane", green)
            st.metric("⏱ Remaining Time (s)", remaining)

        with chart_placeholder.container():
            st.subheader("Vehicle Count per Road")
            st.bar_chart(df.set_index("Road"))

    except Exception as e:
        st.error(f"Connection error: {e}")

    time.sleep(1)
