# traffic_dashboard.py
import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(layout="wide", page_title="Traffic Control Dashboard")

st.title("Smart Traffic Control — Live Dashboard")

# placeholders
col1, col2 = st.columns([2,3])

with col1:
    st.subheader("Intersection Status")
    phase_box = st.empty()
    countdown_box = st.empty()
    metrics_box = st.empty()
    serve_table = st.empty()

with col2:
    st.subheader("Lane Densities & Charts")
    counts_cols = st.columns(4)
    pN = counts_cols[0].empty()
    pE = counts_cols[1].empty()
    pS = counts_cols[2].empty()
    pW = counts_cols[3].empty()

    chart_placeholder = st.empty()
    history_len = st.slider("History points", min_value=20, max_value=500, value=120, step=20)

# log area
st.subheader("Recent Decisions")
log_placeholder = st.empty()

# data storage in session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, N,E,S,W,phase,green,start,end

REFRESH_SEC = 1.0

def get_status():
    try:
        r = requests.get(API_BASE + "/get_status", timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

# main update loop
while True:
    status = get_status()
    if status is None:
        st.warning("No connection to controller API. Make sure controller_api.py is running.")
        time.sleep(1.5)
        continue

    counts = status.get("counts", {"N":0,"E":0,"S":0,"W":0})
    last_decision = status.get("last_decision", {})
    ts = datetime.fromtimestamp(status.get("timestamp", time.time())).strftime("%H:%M:%S")

    # update small counters
    pN.metric("North (N)", counts.get("N",0))
    pE.metric("East  (E)", counts.get("E",0))
    pS.metric("South (S)", counts.get("S",0))
    pW.metric("West  (W)", counts.get("W",0))

    # show current phase & countdown
    phase = last_decision.get("phase", "—")
    green = last_decision.get("green", 0) or 0
    start = last_decision.get("start_time")
    end = last_decision.get("end_time")
    now_ts = time.time()
    remaining = max(0, int(end - now_ts)) if end else 0

    phase_box.markdown(f"**Current Phase:**  `{phase}`")
    countdown_box.markdown(f"**Time left:**  `{remaining} s`")

    # summary metrics
    avg_queue = (counts.get("N",0)+counts.get("E",0)+counts.get("S",0)+counts.get("W",0))/4.0
    metrics_box.markdown(f"- **Avg queue**: `{avg_queue:.2f}`\n- **Last update**: `{ts}`")

    # record into history (only when new decision or every tick)
    rec = {"time": ts, "N": counts.get("N",0), "E": counts.get("E",0), "S": counts.get("S",0), "W": counts.get("W",0),
           "phase": phase, "green": green, "remaining": remaining}
    st.session_state.history.append(rec)
    # cap history length
    if len(st.session_state.history) > 2000:
        st.session_state.history = st.session_state.history[-2000:]

    # build DataFrame for charts from last history_len
    df = pd.DataFrame(st.session_state.history[-history_len:])
    df.index = range(len(df))
    if not df.empty:
        chart_df = df[["N","E","S","W"]].rename(columns={"N":"North","E":"East","S":"South","W":"West"})
        chart_placeholder.line_chart(chart_df)

    # show last 10 decisions as table
    recent = df[["time","phase","green","N","E","S","W"]].tail(10).iloc[::-1]
    serve_table.table(recent)

    # log area (text)
    log_lines = []
    latest = st.session_state.history[-6:]
    for r in latest[::-1]:
        log_lines.append(f"{r['time']} | serve {r['phase']} for {r['green']}s | counts N{r['N']} E{r['E']} S{r['S']} W{r['W']}")
    log_placeholder.markdown("\n".join(log_lines))

    # wait before next poll
    time.sleep(REFRESH_SEC)
    # streamlit requires an explicit rerun for while-loops; but sleep + loop works
    # Breaking condition if user stops the app in the browser is handled by the Streamlit run environment.
