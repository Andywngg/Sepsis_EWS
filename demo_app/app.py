from __future__ import annotations

# PURPOSE: Interactive Streamlit demo for exploring per-patient sepsis risk predictions.
#
# WHAT IT DOES:
#   - Select any test patient from the sidebar
#   - Plots predicted sepsis probability over time (risk trajectory)
#   - Shows when the first alert fires vs when sepsis actually onset
#   - Threshold slider updates alerts in real time
#   - Replay slider: scrub hour-by-hour to see which measurements changed most
#   - Export risk plot (PNG) or full report (PDF)
#   - QR code lets judges scan and open the demo on their phones
#
# RUN: streamlit run demo_app/app.py

import io
import json
import random
import socket
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sepsis_ews.data import load_patient_df, build_features, compute_onset_hour, compute_quality
from sepsis_ews.utils import apply_alert_policy, compute_prediction_utility

try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False
try:
    from fpdf import FPDF, XPos, YPos
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


@st.cache_resource  # runs once and caches — prevents reloading the 40MB model on every click
def load_model(weights_path: Path, medians_path: Path, metrics_path: Path):
    bundle = joblib.load(weights_path)
    med = json.loads(medians_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return bundle["model"], bundle["scaler"], np.array(med["medians"], dtype=float), metrics


@st.cache_data  # caches per argument — same patient path returns instantly on second load
def list_patients(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.glob("*.psv")])


@st.cache_data
def load_patient(path: Path, feature_set: str):
    df = load_patient_df(path)
    labels = df["SepsisLabel"].values.astype(int)
    feats, _ = build_features(df, feature_set=feature_set, patient_normalize=False)
    onset = compute_onset_hour(labels)
    quality = compute_quality(df)
    return df, feats, labels, onset, quality


def _load_case_studies(case_dir: Path) -> dict:
    path = case_dir / "case_studies.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _load_policy(policy_path: Path) -> dict:
    if not policy_path.exists():
        return {}
    return json.loads(policy_path.read_text(encoding="utf-8"))

def _nearest_policy_row(rows: list[dict], threshold: float) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r.get("threshold", 0.0)) - threshold))

def _top_deltas(df_values: np.ndarray, cols: list[str], hour: int, top_k: int = 6):
    if hour <= 0:
        return []
    prev = df_values[hour - 1]
    curr = df_values[hour]
    deltas = curr - prev
    out = []
    for i, col in enumerate(cols):
        if col == "SepsisLabel":
            continue
        if np.isnan(curr[i]) and np.isnan(prev[i]):
            continue
        out.append((col, curr[i], deltas[i]))
    out.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else -1, reverse=True)
    return out[:top_k]

def _build_risk_plot(
    probs: np.ndarray,
    threshold: float,
    onset: int | None,
    first_alert: int | None,
    baseline_probs: np.ndarray | None,
    baseline_threshold: float | None,
):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.plot(probs, label="Risk")
    if baseline_probs is not None:
        ax.plot(baseline_probs, color="gray", linestyle="--", label="Baseline risk")
    ax.axhline(threshold, color="orange", linestyle="--", label="Threshold")
    if baseline_threshold is not None:
        ax.axhline(baseline_threshold, color="gray", linestyle=":", label="Baseline threshold")
    if onset is not None:
        ax.axvline(onset, color="red", linestyle="--", label="Onset")
    if first_alert is not None:
        ax.axvline(first_alert, color="green", linestyle="--", label="First alert")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Risk")
    ax.legend(loc="upper right", fontsize=6)
    plt.tight_layout()
    return fig


def _compute_patient_utility(labels: np.ndarray, preds: np.ndarray) -> float:
    # Official utility normalization, applied to a single patient.
    dt_early = -12
    dt_optimal = -6
    dt_late = 3

    observed = compute_prediction_utility(labels, preds)

    best_preds = np.zeros_like(labels)
    if np.any(labels):
        t_sepsis = int(np.argmax(labels)) - dt_optimal
        start = max(0, t_sepsis + dt_early)
        end = min(t_sepsis + dt_late + 1, len(labels))
        best_preds[start:end] = 1
    best = compute_prediction_utility(labels, best_preds)

    inaction = compute_prediction_utility(labels, np.zeros_like(labels))
    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)

def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def _build_pdf_report(
    patient_id: str,
    hours: int,
    onset: int | None,
    first_alert: int | None,
    lead_time: str,
    threshold: float,
    alert_k: int,
    metrics: dict,
    policy_row: dict | None,
    deltas: list[tuple],
    plot_png: bytes,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Sepsis Early-Warning Demo Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Patient: {patient_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Hours: {hours}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Onset hour: {onset if onset is not None else 'None'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"First alert hour: {first_alert if first_alert is not None else 'None'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Lead time (hrs): {lead_time}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Alert Policy", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Threshold: {threshold:.2f} | Consecutive alerts (k): {alert_k}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if policy_row:
        pdf.cell(
            0,
            6,
            f"Utility: {policy_row['utility']:.3f} | Early detection: {policy_row['early_detection_rate']:.3f} | "
            f"False alert: {policy_row['false_alert_rate']:.3f} | Alerts/day: {policy_row['alerts_per_patient_day']:.2f}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    pdf.ln(2)
    if metrics:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Model Snapshot", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        auroc = metrics.get("metrics", {}).get("auroc", 0.0)
        auprc = metrics.get("metrics", {}).get("auprc", 0.0)
        util = metrics.get("official_utility", metrics.get("utility_score", 0.0))
        alerts = metrics.get("alert_burden", {}).get("alerts_per_patient_day", 0.0)
        pdf.cell(
            0,
            6,
            f"AUROC: {auroc:.3f} | AUPRC: {auprc:.3f} | Utility: {util:.3f} | Alerts/day: {alerts:.2f}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    pdf.ln(2)
    img_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(plot_png)
            img_path = tmp.name
        pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
        pdf.ln(50)
    finally:
        if img_path:
            try:
                Path(img_path).unlink(missing_ok=True)
            except Exception:
                pass
    if deltas:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Top 1-hour Signal Changes", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for name, val, delta in deltas:
            val_str = "nan" if np.isnan(val) else f"{val:.3f}"
            delta_str = "nan" if np.isnan(delta) else f"{delta:+.3f}"
            pdf.cell(
                0,
                5,
                f"{name}: {val_str} (d {delta_str})",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode("latin1")
    return bytes(pdf_bytes)

def _get_local_ip() -> str | None:
    # Find the machine's local WiFi IP so the QR code points to the right address.
    # Trick: open a UDP socket toward Google DNS — the OS picks the network interface,
    # then we read which IP was assigned. No data is actually sent.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Sepsis Early-Warning Demo", layout="wide")
    st.title("Sepsis Early-Warning Demo")
    st.caption("Interactive replay of patient risk over time with alert policies.")

    data_dir = Path(
        st.sidebar.text_input("Data directory", str(ROOT / "data" / "train"))
    )
    weights_path = Path(
        st.sidebar.text_input("Model weights", str(ROOT / "outputs" / "utility" / "model.joblib"))
    )
    medians_path = Path(
        st.sidebar.text_input("Medians file", str(ROOT / "outputs" / "utility" / "medians.json"))
    )
    metrics_path = Path(
        st.sidebar.text_input("Metrics file", str(ROOT / "outputs" / "eval_full" / "metrics.json"))
    )
    case_dir = Path(
        st.sidebar.text_input("Case study folder", str(ROOT / "outputs" / "case_studies"))
    )
    policy_path = Path(
        st.sidebar.text_input("Policy tradeoff file", str(ROOT / "outputs" / "policy_full" / "policy_analysis.json"))
    )
    show_baseline = st.sidebar.checkbox("Show baseline comparison", value=True)
    baseline_weights = Path(
        st.sidebar.text_input("Baseline weights", str(ROOT / "outputs" / "baseline" / "model.joblib"))
    )
    baseline_medians = Path(
        st.sidebar.text_input("Baseline medians", str(ROOT / "outputs" / "baseline" / "medians.json"))
    )
    baseline_metrics_path = Path(
        st.sidebar.text_input("Baseline metrics", str(ROOT / "outputs" / "baseline" / "metrics.json"))
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Share Demo")
    local_ip = _get_local_ip()
    default_url = "http://localhost:8501"
    if local_ip:
        default_url = f"http://{local_ip}:8501"
    share_url = st.sidebar.text_input("Demo URL", default_url)
    st.sidebar.caption("Tip: run with --server.address 0.0.0.0 to share on your Wi-Fi.")
    if QR_AVAILABLE:
        qr_img = qrcode.make(share_url)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        st.sidebar.image(buf.getvalue(), caption="Scan to open demo")
        st.sidebar.download_button(
            "Download QR",
            data=buf.getvalue(),
            file_name="sepsis_demo_qr.png",
            mime="image/png",
        )
    else:
        st.sidebar.info("Install requirements_demo.txt to enable QR code.")

    if not data_dir.exists():
        st.error("Data directory not found.")
        return

    files = list_patients(data_dir)
    if not files:
        st.error("No patient files found.")
        return

    cases = _load_case_studies(case_dir)
    if cases:
        st.subheader("Quick Scenarios")
        cols = st.columns(3)
        mapping = [
            ("early_detection", "Early detection"),
            ("late_detection", "Late detection"),
            ("false_alarm", "False alarm"),
        ]
        for col, (key, label) in zip(cols, mapping):
            info = cases.get(key)
            if not info:
                col.button(f"{label} (missing)", disabled=True)
                continue
            pid = info["patient_id"]
            if col.button(f"{label}: {pid}"):
                st.session_state["selected_patient"] = pid
                target_hour = info.get("first_alert_hour")
                if target_hour is None:
                    target_hour = info.get("onset_hour", 0)
                st.session_state["current_hour"] = max(int(target_hour) - 1, 0)
                st.rerun()
        st.markdown("---")

    filter_text = st.sidebar.text_input("Filter patient ID (optional)", "")
    if filter_text:
        filtered = [p for p in files if filter_text in p.stem]
    else:
        filtered = files

    st.sidebar.caption(f"{len(filtered)} patients match filter.")
    if len(filtered) > 500:
        st.sidebar.info("Showing a random sample of 500 patients. Use filter to narrow.")
        random.seed(42)
        filtered = random.sample(filtered, 500)
    # Ensure quick-scenario selection stays in the list even after sampling.
    selected_id = st.session_state.get("selected_patient")
    if selected_id:
        selected_path = next((p for p in files if p.stem == selected_id), None)
        if selected_path and selected_path not in filtered:
            filtered.append(selected_path)

    patient_ids = [p.stem for p in filtered]
    if "selected_patient" in st.session_state and st.session_state["selected_patient"] in patient_ids:
        st.session_state["patient_select"] = st.session_state["selected_patient"]
    if "patient_select" not in st.session_state:
        st.session_state["patient_select"] = filtered[0].stem
    patient_id = st.sidebar.selectbox("Patient", patient_ids, key="patient_select")
    matches = [p for p in filtered if p.stem == patient_id]
    patient_path = matches[0] if matches else filtered[0]
    # These sliders re-run the whole script on every change — alerts update instantly
    threshold = st.sidebar.slider("Alert threshold", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    alert_k = st.sidebar.slider("Consecutive alerts (k)", min_value=1, max_value=3, value=1, step=1)
    show_quality = st.sidebar.checkbox("Show data quality curve", value=True)

    model, scaler, medians, metrics = load_model(weights_path, medians_path, metrics_path)
    feature_set = metrics.get("feature_set", "enhanced")

    df, feats, labels, onset, quality = load_patient(patient_path, feature_set)
    baseline_probs = None
    baseline_threshold = None
    if show_baseline and baseline_weights.exists() and baseline_medians.exists():
        base_bundle = joblib.load(baseline_weights)
        base_med = json.loads(baseline_medians.read_text(encoding="utf-8"))
        base_medians = np.array(base_med["medians"], dtype=float)
        base_medians = np.where(np.isnan(base_medians), 0.0, base_medians)
        base_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8")) if baseline_metrics_path.exists() else {}
        base_feature_set = base_metrics.get("feature_set", "basic")
        base_feats, _ = build_features(df, feature_set=base_feature_set, patient_normalize=False)
        Xb = np.where(np.isnan(base_feats), base_medians, base_feats)
        Xb = base_bundle["scaler"].transform(Xb)
        baseline_probs = base_bundle["model"].predict_proba(Xb)[:, 1]
        baseline_threshold = float(base_metrics.get("best_threshold", 0.1))
    # Impute → scale → predict probabilities for the selected patient
    X = np.where(np.isnan(feats), medians, feats)
    X = scaler.transform(X)
    probs = model.predict_proba(X)[:, 1]  # one probability per hour

    # Apply alert policy: fire alert where prob >= threshold
    preds = apply_alert_policy(probs, threshold, alert_k=alert_k)
    alert_idx = np.where(preds == 1)[0]
    first_alert = int(alert_idx[0]) if len(alert_idx) else None
    patient_utility = _compute_patient_utility(labels, preds)

    st.subheader("Summary")
    cols = st.columns(5)
    cols[0].metric("Patient", patient_path.stem)
    cols[1].metric("Hours", len(df))
    cols[2].metric("Sepsis onset hour", onset if onset is not None else "None")
    cols[3].metric("First alert hour", first_alert if first_alert is not None else "None")
    if onset is not None and first_alert is not None:
        lead_time = onset - first_alert
        cols[4].metric("Lead time (hrs)", lead_time)
    else:
        lead_time = "n/a"
        cols[4].metric("Lead time (hrs)", lead_time)
    st.metric("Patient utility (normalized)", f"{patient_utility:.3f}")

    if metrics:
        st.subheader("Model Snapshot")
        snap = st.columns(4)
        snap[0].metric("AUROC", f"{metrics.get('metrics', {}).get('auroc', 0.0):.3f}")
        snap[1].metric("AUPRC", f"{metrics.get('metrics', {}).get('auprc', 0.0):.3f}")
        util_val = metrics.get("official_utility", metrics.get("utility_score", 0.0))
        snap[2].metric("Utility", f"{util_val:.3f}")
        snap[3].metric("Alerts/day", f"{metrics.get('alert_burden', {}).get('alerts_per_patient_day', 0.0):.2f}")

    # MAIN CHART: risk trajectory plot
    # Blue line = model probability, orange dashed = threshold, red = onset, green = first alert
    # Gap between green and red lines = lead time (hours of warning)
    st.subheader("Risk Over Time")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(probs, label="Risk")
    if baseline_probs is not None:
        ax.plot(baseline_probs, color="gray", linestyle="--", label="Baseline risk")
    ax.axhline(threshold, color="orange", linestyle="--", label="Threshold")
    if baseline_threshold is not None:
        ax.axhline(baseline_threshold, color="gray", linestyle=":", label="Baseline threshold")
    if onset is not None:
        ax.axvline(onset, color="red", linestyle="--", label="Onset")
    if first_alert is not None:
        ax.axvline(first_alert, color="green", linestyle="--", label="First alert")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Risk")
    ax.legend(loc="upper right")
    st.pyplot(fig)

    policy = _load_policy(policy_path)
    policy_row = None
    if policy and "rows" in policy:
        st.subheader("Threshold Tradeoff")
        rows = policy["rows"]
        thresholds = [r["threshold"] for r in rows]
        utilities = [r["utility"] for r in rows]
        alerts = [r["alerts_per_patient_day"] for r in rows]
        col_a, col_b = st.columns(2)
        fig_u, ax_u = plt.subplots(figsize=(4, 3))
        ax_u.plot(thresholds, utilities, marker="o")
        ax_u.axvline(threshold, color="orange", linestyle="--")
        ax_u.set_xlabel("Threshold")
        ax_u.set_ylabel("Utility")
        ax_u.set_title("Utility vs Threshold")
        col_a.pyplot(fig_u)
        fig_b, ax_b = plt.subplots(figsize=(4, 3))
        ax_b.plot(thresholds, alerts, marker="o", color="#4c78a8")
        ax_b.axvline(threshold, color="orange", linestyle="--")
        ax_b.set_xlabel("Threshold")
        ax_b.set_ylabel("Alerts per patient-day")
        ax_b.set_title("Alert Burden vs Threshold")
        col_b.pyplot(fig_b)

        nearest = _nearest_policy_row(rows, threshold)
        if nearest:
            policy_row = nearest
            st.subheader("Operational Summary (policy curve)")
            op = st.columns(4)
            op[0].metric("Utility", f"{nearest['utility']:.3f}")
            op[1].metric("Early detection", f"{nearest['early_detection_rate']:.3f}")
            op[2].metric("False alert", f"{nearest['false_alert_rate']:.3f}")
            op[3].metric("Alerts/day", f"{nearest['alerts_per_patient_day']:.2f}")

    if show_quality:
        st.subheader("Data Quality (1 - missingness)")
        fig_q, ax_q = plt.subplots(figsize=(8, 2))
        ax_q.plot(quality, color="purple")
        ax_q.set_xlabel("Hour")
        ax_q.set_ylabel("Quality")
        st.pyplot(fig_q)

    st.subheader("Replay")
    max_hour = max(len(df) - 1, 0)
    if "current_hour" not in st.session_state:
        st.session_state["current_hour"] = min(24, max_hour)
    if st.session_state["current_hour"] > max_hour:
        st.session_state["current_hour"] = max_hour
    current_hour = st.slider(
        "Current hour",
        min_value=0,
        max_value=max_hour,
        value=st.session_state["current_hour"],
        key="current_hour",
    )
    if current_hour > max_hour:
        current_hour = max_hour
    risk_now = probs[current_hour]
    alerted_so_far = bool(np.any(preds[: current_hour + 1] == 1))
    st.write(
        f"At hour {current_hour}: risk={risk_now:.3f} | alert triggered so far: {alerted_so_far}"
    )

    # FEATURE ATTRIBUTION: which measurements changed most in the last hour?
    # Sorts by absolute delta — largest changes rank first.
    st.subheader("Top Signal Changes (1-hour delta)")
    deltas = _top_deltas(df.values, list(df.columns), current_hour, top_k=6)
    if deltas:
        delta_df = {
            "Feature": [d[0] for d in deltas],
            "Value": [float(d[1]) if not np.isnan(d[1]) else np.nan for d in deltas],
            "Delta (1h)": [float(d[2]) if not np.isnan(d[2]) else np.nan for d in deltas],
        }
        st.dataframe(delta_df, use_container_width=True)
    else:
        st.caption("Not enough history to compute deltas.")

    st.subheader("Export Report")
    fig_export = _build_risk_plot(
        probs,
        threshold,
        onset,
        first_alert,
        baseline_probs,
        baseline_threshold,
    )
    plot_png = _fig_to_png_bytes(fig_export)
    st.download_button(
        "Download risk plot (PNG)",
        data=plot_png,
        file_name=f"{patient_path.stem}_risk_plot.png",
        mime="image/png",
    )
    if PDF_AVAILABLE:
        pdf_bytes = _build_pdf_report(
            patient_id=patient_path.stem,
            hours=len(df),
            onset=onset,
            first_alert=first_alert,
            lead_time=str(lead_time),
            threshold=threshold,
            alert_k=alert_k,
            metrics=metrics,
            policy_row=policy_row,
            deltas=deltas,
            plot_png=plot_png,
        )
        st.download_button(
            "Download 1-page report (PDF)",
            data=pdf_bytes,
            file_name=f"{patient_path.stem}_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Install fpdf2 to enable PDF export.")

    st.subheader("Raw Vitals/Labs (selected hour)")
    st.dataframe(df.iloc[[current_hour]].T, use_container_width=True)

    st.subheader("Case Studies")
    cases = _load_case_studies(case_dir)
    if cases:
        cols = st.columns(3)
        keys = ["early_detection", "late_detection", "false_alarm"]
        labels = ["Early detection", "Late detection", "False alarm"]
        for col, key, label in zip(cols, keys, labels):
            info = cases.get(key)
            if not info:
                continue
            pid = info["patient_id"]
            col.markdown(f"**{label}: {pid}**")
            img_path = case_dir / f"{pid}.png"
            if img_path.exists():
                # Pass bytes to avoid Windows path being treated as a URL by Streamlit.
                col.image(img_path.read_bytes(), use_column_width=True)
            col.caption(f"First alert: {info['first_alert_hour']}, Onset: {info['onset_hour']}")
    else:
        st.caption("Case studies not found. Run scripts/case_studies.py first.")


if __name__ == "__main__":
    main()
