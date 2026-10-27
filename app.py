import re
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

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
DATA_PATH = "synthetic_labelled_sessions.csv"
RANDOM_STATE = 42


def train_models():
    """Retrain both models from the bundled synthetic CSV (same settings as the notebook)."""
    data = pd.read_csv(Path(__file__).parent / DATA_PATH)
    X, y, groups = data[FEATURES], data["label"].astype(int), data["user_id"]
    train_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE).split(X, y, groups=groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ]).fit(X_train, y_train)
    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(n_estimators=200, contamination=0.20, random_state=RANDOM_STATE)),
    ]).fit(X_train.loc[y_train == 0])
    return rf, iso


@st.cache_resource
def load_models():
    """Load the saved models. If they were built with a different scikit-learn
    version (or can't be read), retrain from the bundled CSV instead of crashing."""
    base = Path(__file__).parent
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat version-mismatch warnings as failures
            return joblib.load(base / RF_PATH), joblib.load(base / ISO_PATH), "saved"
    except Exception:
        rf, iso = train_models()
        return rf, iso, "retrained"

def parse_timestamp(value):
    """Parse common WhatsApp timestamps across Android/iOS export locales."""
    value = str(value).strip()
    # Normalize non-breaking/narrow spaces and invisible direction marks found in exports.
    value = re.sub(r"[\u00a0\u202f\u200e\u200f\u2066-\u2069]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().strip("[]")
    formats = [
        # Day/month and month/day, with 2- or 4-digit year; 12/24-hour clocks.
        "%d/%m/%Y, %I:%M:%S %p", "%d/%m/%y, %I:%M:%S %p",
        "%m/%d/%Y, %I:%M:%S %p", "%m/%d/%y, %I:%M:%S %p",
        "%d/%m/%Y, %I:%M %p", "%d/%m/%y, %I:%M %p",
        "%m/%d/%Y, %I:%M %p", "%m/%d/%y, %I:%M %p",
        "%d/%m/%Y, %H:%M:%S", "%d/%m/%y, %H:%M:%S",
        "%m/%d/%Y, %H:%M:%S", "%m/%d/%y, %H:%M:%S",
        "%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M",
        "%m/%d/%Y, %H:%M", "%m/%d/%y, %H:%M",
        # ISO-like and dot-separated dates seen in some localized exports.
        "%Y-%m-%d, %H:%M:%S", "%Y-%m-%d, %H:%M",
        "%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M",
        "%d.%m.%Y, %H:%M:%S", "%d.%m.%Y, %H:%M",
        "%d.%m.%y, %H:%M:%S", "%d.%m.%y, %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_txt(raw_text):
    """Parse common WhatsApp text exports, including bracketed iOS and Android formats.

    A new message begins with a date/time prefix. Continuation lines are appended to
    the previous message. System notifications without a sender are safely ignored.
    """
    # Date/time prefix; allow seconds, varying date separators, AM/PM, and Unicode spaces.
    date_part = r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})"
    time_part = r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?"
    patterns = [
        re.compile(r"^\[\s*(" + date_part + r",\s*" + time_part + r")\s*\]\s*(.*)$"),
        re.compile(r"^(" + date_part + r",\s*" + time_part + r")\s+-\s+(.*)$"),
    ]
    rows, current = [], None
    for original_line in raw_text.lstrip("\ufeff").splitlines():
        # Strip common invisible direction characters but preserve message contents otherwise.
        line = re.sub(r"[\u200e\u200f\u2066-\u2069]", "", original_line).strip("\r")
        match = None
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                break
        if match:
            if current:
                rows.append(current)
            ts = parse_timestamp(match.group(1))
            rest = match.group(2).strip()
            # Sender labels may contain spaces; split on the first colon followed by a space.
            sender_match = re.match(r"^(.+?):\s?(.*)$", rest)
            if ts and sender_match and sender_match.group(1).strip():
                current = {
                    "timestamp": ts,
                    "sender": sender_match.group(1).strip(),
                    "message": sender_match.group(2).strip(),
                }
            else:
                # Ignore WhatsApp system events (e.g. encryption notices) without a sender.
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
    rf_model, iso_model, model_source = load_models()
except Exception as exc:
    st.error("The models could not be loaded or trained. Make sure the .joblib files and synthetic_labelled_sessions.csv are beside app.py in the GitHub repository.")
    st.code(f"Required files:\n{RF_PATH}\n{ISO_PATH}\n{DATA_PATH}\n\nError: {exc}")
    st.stop()
if model_source == "retrained":
    st.caption("Models were retrained on startup from the bundled synthetic dataset (saved models were built with a different scikit-learn version).")

st.info("Use a WhatsApp exported .txt chat, or a prepared .csv containing the five model features. Results are experimental because the models were trained on synthetic data.")
uploaded = st.file_uploader("Upload WhatsApp export (.txt) or prepared feature CSV (.csv)", type=["txt", "csv"])
if uploaded:
    try:
        if uploaded.name.lower().endswith(".txt"):
            raw = uploaded.getvalue().decode("utf-8-sig", errors="replace")
            messages = parse_whatsapp_txt(raw)
            if messages.empty:
                st.error("The app could not recognize any timestamped sender messages in this .txt file.")
                st.caption("Supported examples: 12/06/2026, 9:30 PM - Name: Hello | [12/06/26, 9:30:12 AM] Name: Hello | 2026-06-12, 21:30 - Name: Hello. Export as plain text (.txt), not a PDF or copied chat screen.")
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
