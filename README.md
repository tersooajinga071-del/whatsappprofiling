# WhatsApp Behavioural Profiling — Hosting Package

## Included files
- `app.py` — Streamlit web app. Accepts WhatsApp `.txt` exports or prepared feature `.csv` files.
- `requirements.txt` — dependencies for Streamlit Community Cloud.
- `whatsapp_takeover_rf_model.joblib` — Random Forest model.
- `whatsapp_isolation_forest_model.joblib` — Isolation Forest model.
- `synthetic_labelled_sessions.csv` — simulated teaching dataset used to train the models.
- `WhatsApp_Behavioural_Profiling_Colab_reduced_false_alerts.ipynb` — Colab notebook for training and experimentation.

## Deploy with Streamlit Community Cloud
1. Download and extract this ZIP.
2. Open your GitHub repository (for example, `whatsappprofiling`).
3. Upload **all six files** to the repository root: `app.py`, `requirements.txt`, both `.joblib` files, `synthetic_labelled_sessions.csv`, and the notebook (the notebook is optional for hosting).
4. Commit the files.
5. Go to https://share.streamlit.io/ and create an app from that GitHub repository.
6. Select your repository and branch, set **Main file path** to `app.py`, then deploy.
7. If deployment fails, open the app's Logs and inspect the first error message.

## Upload formats
- WhatsApp `.txt` export: common timestamped export formats are parsed into approximate per-sender features.
- `.csv`: must already contain all five feature columns: `avg_message_length`, `messages_per_hour`, `writing_style_similarity`, `contact_change_ratio`, `unusual_hour_ratio`.

## Important scientific limitation
The models are trained on synthetic data, not verified real account-takeover examples. The alert rule requires both models to agree and a Random Forest score of at least 0.80. This is a conservative heuristic, not a validated real-world threshold. An alert is not proof of compromise, and no alert is not proof of safety. Real-world performance remains unknown and requires representative labelled evaluation data. A WhatsApp text export cannot reliably reveal contact-change history, and writing-style similarity is only a rough proxy.


## Important deployment fix for Streamlit Community Cloud

The earlier deployment logs showed Python 3.14.9 while this project pins older pandas, NumPy, and scikit-learn versions. Use **Python 3.12** for this package.

1. In the current Streamlit app, note the GitHub repository (`whatsappprofiling`), branch (`main`), and entrypoint (`app.py`).
2. Streamlit Community Cloud does not let you change an existing app's Python version in place. Delete the current app from Streamlit Community Cloud (this does not delete your GitHub repository).
3. Create/deploy the app again from the same repository and branch.
4. Before deploying, open **Advanced settings** and select **Python 3.12**.
5. Confirm the app entrypoint is `app.py`, then deploy.
6. Keep `requirements.txt` in the repository root beside `app.py`. It must be named exactly `requirements.txt`.
7. If another dependency manager file exists at the app path (for example `uv.lock`, `Pipfile`, `environment.yml`, or `pyproject.toml`), make sure it is not unintentionally taking precedence over `requirements.txt`; Community Cloud uses one dependency file.
8. Watch the build logs. They should show Python 3.12 and successful installation of joblib and scikit-learn.

The code and model files are included in this archive, but no package can guarantee successful deployment until the Cloud environment and logs confirm it.
