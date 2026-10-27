import re
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="WhatsApp Behavioural Profiling", page_icon="🛡️", layout="wide")
st.title("🛡️ WhatsApp Behavioural Profiling")
st.caption("Experimental behavioural screening — an alert is not proof of account takeover.")

FEATURES = [
    "avg_message_length",
    "messages_per_hour",
    "writing_style_similarity",
    "contact_change_ratio",
    "unusual_hour_ratio",
]
RF_PATH = "whatsapp_takeover_rf_model.joblib"
ISO_PATH = "whatsapp_isolation_forest_model.joblib"

@st.cache_resource
def load_models():
    return joblib.load(RF_PATH), joblib.load(ISO_PATH)

def parse_timestamp(value):
    value = value.strip()
    formats = [
        "%d/%m/%Y, %I:%M %p", "%d/%m/%y, %I:%M %p",
        "%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M",
        "%m/%d/%Y, %I:%M %p", "%m/%d/%y, %I:%M %p",
        "%m/%d/%Y, %H:%M", "%m/%d/%y, %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None

def parse_whatsapp_txt(raw_text):
    # Supports common Android/iOS export styles; timestamps are interpreted locally.
    patterns = [
        re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\]\s*([^:]+?):\s?(.*)$"),
        re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+?):\s?(.*)$"),
    ]
    rows, current = [], None
    for line in raw_text.splitlines():
        match = None
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                break
        if match:
            if current:
                rows.append(current)
            ts = parse_timestamp(match.group(1))
            if ts:
                current = {"timestamp": ts, "sender": match.group(2).strip(), "message": match.group(3).strip()}
            else:
                current = None
        elif current and line.strip():
            current["message"] += "\n" + line.strip()
    if current:
        rows.append(current)
    return pd.DataFrame(rows, columns=["timestamp", "sender", "message"])

def chat_to_features(messages):
    if messages.empty:
        raise ValueError("No timestamped sender messages were parsed. Check the export format.")
    records = []
    for sender, group in messages.groupby("sender"):
        group = group.sort_values("timestamp")
        lengths = group["message"].astype(str).str.len()
        span_hours = max((group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600, 1/60)
        # This is a crude proxy, not a validated measure of writing identity.
        cv = float(lengths.std(ddof=0) / lengths.mean()) if lengths.mean() else 1.0
        records.append({
            "sender": sender,
            "message_count": int(len(group)),
            "avg_message_length": float(lengths.mean()) if len(lengths) else 0.0,
            "messages_per_hour": float(len(group) / span_hours),
            "writing_style_similarity": float(np.clip(1 - cv, 0, 1)),
            # WhatsApp text exports do not provide contact-change history.
            "contact_change_ratio": 0.0,
            "unusual_hour_ratio": float(group["timestamp"].dt.hour.isin(list(range(0,6)) + [22,23]).mean()),
        })
    return pd.DataFrame(records)

def prepare_csv(df):
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError("CSV is missing required model features: " + ", ".join(missing) +
                         ". A raw WhatsApp export CSV usually needs feature preparation first.")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("One or more required feature values are missing or non-numeric.")
    return df

try:
    rf_model, iso_model = load_models()
except Exception as exc:
    st.error("The model files could not be loaded. Put both .joblib files beside app.py in the GitHub repository.")
    st.code(f"Required files:\n{RF_PATH}\n{ISO_PATH}")
    st.stop()

st.info("Use a WhatsApp exported .txt chat, or a prepared .csv containing the five model features. Results are experimental because the models were trained on synthetic data.")
uploaded = st.file_uploader("Upload WhatsApp export (.txt) or prepared feature CSV (.csv)", type=["txt", "csv"])
if uploaded:
    try:
        if uploaded.name.lower().endswith(".txt"):
            raw = uploaded.getvalue().decode("utf-8-sig", errors="replace")
            messages = parse_whatsapp_txt(raw)
            if messages.empty:
                st.error("No supported timestamped messages were found. Try exporting the chat without media and check the timestamp format.")
                st.stop()
            features_df = chat_to_features(messages)
            st.write(f"Parsed **{len(messages)} messages** from **{features_df['sender'].nunique()} senders**.")
        else:
            features_df = prepare_csv(pd.read_csv(uploaded))
        X = features_df[FEATURES].copy()
        rf_pred = rf_model.predict(X).astype(int)
        proba = rf_model.predict_proba(X)
        classes = list(rf_model.classes_)
        rf_score = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        iso_flag = (iso_model.predict(X) == -1).astype(int)
        anomaly_score = -iso_model.decision_function(X)
        # Conservative heuristic: both models must agree, and RF score >= 0.80.
        flagged = (rf_score >= 0.80) & (iso_flag == 1)
        results = features_df.copy()
        results["random_forest_prediction"] = rf_pred
        results["random_forest_score"] = np.round(rf_score, 4)
        results["isolation_forest_anomaly"] = iso_flag
        results["anomaly_score"] = np.round(anomaly_score, 4)
        results["screening_status"] = np.where(
            flagged, "Flagged for review (both models agree)",
            "No consensus alert (not proof of safety)"
        )
        a, b = st.columns(2)
        a.metric("Senders / rows screened", len(results))
        b.metric("Flagged for review", int(flagged.sum()))
        st.dataframe(results, use_container_width=True)
        st.download_button("Download screening results CSV", results.to_csv(index=False).encode("utf-8"),
                           file_name="screened_behavioural_sessions.csv", mime="text/csv")
        st.warning("A review alert is not proof of compromise. No alert is not proof of safety. The chat-derived features and thresholds have not been validated for real-world account takeover detection.")
    except Exception as exc:
        st.error(str(exc))

with st.expander("Privacy and limitations"):
    st.write("Process chats only with the appropriate consent. This app does not contact WhatsApp, verify account ownership, or establish that an account has been taken over. A text export cannot reveal reliable contact-change history, and the writing-style value is only a rough proxy.")
