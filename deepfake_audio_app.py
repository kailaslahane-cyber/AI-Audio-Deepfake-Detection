"""
╔══════════════════════════════════════════════════════════╗
║      Deepfake Audio Detector  —  Streamlit Web App       ║
║      Run:  streamlit run deepfake_audio_app.py           ║
╚══════════════════════════════════════════════════════════╝

Install dependencies first:
    pip install streamlit librosa numpy matplotlib scipy
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import datetime
import json
import tempfile
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark theme */
    .stApp { background-color: #0a1628; color: #f1f1f1; }

    /* Header */
    .main-title {
        font-size: 2.5rem; font-weight: 900;
        color: #ffffff; margin-bottom: 0;
    }
    .brand-cyan   { color: #00c8ff; }
    .brand-green  { color: #00e5b0; }
    .brand-red    { color: #ff4d6d; }
    .subtitle {
        color: #7a9bc7; font-size: 1rem;
        margin-top: 0.3rem; margin-bottom: 1.5rem;
    }

    /* Result boxes */
    .result-real {
        background: #092a1a;
        border: 2px solid #00e5b0;
        border-radius: 12px;
        padding: 24px 32px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-fake {
        background: #2a0a14;
        border: 2px solid #ff4d6d;
        border-radius: 12px;
        padding: 24px 32px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-icon  { font-size: 3.5rem; }
    .result-label { font-size: 1.8rem; font-weight: 900; margin-top: 8px; }
    .result-sub   { font-size: 0.95rem; color: #aaa; margin-top: 6px; }

    /* Feature card */
    .feat-card {
        background: #162952;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #00c8ff;
    }
    .feat-card.teal  { border-left-color: #00e5b0; }
    .feat-card.yellow{ border-left-color: #ffd166; }
    .feat-name  { font-size: 1.1rem; font-weight: 800; color: #00c8ff; }
    .feat-card.teal  .feat-name { color: #00e5b0; }
    .feat-card.yellow .feat-name{ color: #ffd166; }
    .feat-val   { font-size: 1.6rem; font-weight: 900; color: #f1f1f1; margin: 4px 0; }
    .feat-desc  { font-size: 0.82rem; color: #7a9bc7; }
    .feat-flag  { font-size: 0.8rem; font-weight: 700; margin-top: 6px; }

    /* Confidence bar container */
    .conf-wrap {
        background: #0f2044;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 1rem 0;
    }
    .conf-label { font-size: 0.85rem; color: #7a9bc7; margin-bottom: 6px; }
    .conf-title { font-size: 1rem; font-weight: 700; color: #f1f1f1; }

    /* Stat pill */
    .pill {
        display: inline-block;
        background: #0f2044;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #b8d0ef;
        margin: 3px;
    }

    /* File uploader override */
    [data-testid="stFileUploader"] {
        background: #162952 !important;
        border: 2px dashed #1e4080 !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #00c8ff !important;
        color: #0a1628 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #0096cc !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #162952 !important;
        color: #00c8ff !important;
        border: 1px solid #00c8ff !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100%;
    }

    hr { border-color: #1e3a6e !important; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(audio_path):
    """Load audio and extract all acoustic features."""
    y, sr = librosa.load(audio_path, sr=None)

    # Core detection features
    mfcc_matrix  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean    = float(np.mean(mfcc_matrix))
    mfcc_std     = float(np.std(mfcc_matrix))

    zcr_arr      = librosa.feature.zero_crossing_rate(y)
    zcr_mean     = float(np.mean(zcr_arr))

    spectral_arr = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_mean= float(np.mean(spectral_arr))

    # Additional features for richer analysis
    chroma       = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    rolloff      = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    bandwidth    = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rms_energy   = float(np.mean(librosa.feature.rms(y=y)))
    tempo_raw    = librosa.beat.beat_track(y=y, sr=sr)[0]
    tempo        = float(np.asarray(tempo_raw).flat[0])
    duration     = float(librosa.get_duration(y=y, sr=sr))

    return {
        "y": y, "sr": sr,
        "mfcc_mean":     mfcc_mean,
        "mfcc_std":      mfcc_std,
        "mfcc_matrix":   mfcc_matrix,
        "zcr_mean":      zcr_mean,
        "zcr_arr":       zcr_arr,
        "spectral_mean": spectral_mean,
        "spectral_arr":  spectral_arr,
        "chroma":        chroma,
        "rolloff":       rolloff,
        "bandwidth":     bandwidth,
        "rms_energy":    rms_energy,
        "tempo":         float(tempo),
        "duration":      duration,
        "sample_rate":   sr,
    }


# ─── Detection Logic ──────────────────────────────────────────────────────────

def classify(features):
    """
    Multi-feature scoring system.
    Each suspicious feature adds to the fake_score.
    Returns verdict, confidence %, and per-feature flags.
    """
    fake_score = 0
    max_score  = 5
    flags      = {}

    # 1. MFCC mean — deepfake audio tends to have unnaturally low values
    if features["mfcc_mean"] < -120:
        fake_score += 2
        flags["mfcc"]     = ("suspicious", f"{features['mfcc_mean']:.2f} < -120 threshold")
    elif features["mfcc_mean"] < -80:
        fake_score += 1
        flags["mfcc"]     = ("warning", f"{features['mfcc_mean']:.2f}  — slightly low")
    else:
        flags["mfcc"]     = ("ok", f"{features['mfcc_mean']:.2f}  — normal range")

    # 2. ZCR — synthesized speech has unusually high ZCR
    if features["zcr_mean"] > 0.1:
        fake_score += 2
        flags["zcr"]      = ("suspicious", f"{features['zcr_mean']:.4f} > 0.1 threshold")
    elif features["zcr_mean"] > 0.08:
        fake_score += 1
        flags["zcr"]      = ("warning", f"{features['zcr_mean']:.4f}  — slightly elevated")
    else:
        flags["zcr"]      = ("ok", f"{features['zcr_mean']:.4f}  — normal range")

    # 3. Spectral bandwidth — AI voices can have unusually narrow or wide band
    if features["bandwidth"] < 500 or features["bandwidth"] > 4000:
        fake_score += 1
        flags["bandwidth"]= ("warning", f"{features['bandwidth']:.1f} Hz  — abnormal bandwidth")
    else:
        flags["bandwidth"]= ("ok", f"{features['bandwidth']:.1f} Hz  — normal range")

    # 4. RMS energy — TTS audio often has very uniform / very low energy
    if features["rms_energy"] < 0.005:
        fake_score += 1
        flags["energy"]   = ("warning", f"{features['rms_energy']:.5f}  — unusually low energy")
    else:
        flags["energy"]   = ("ok", f"{features['rms_energy']:.5f}  — normal energy level")

    # Calculate confidence
    confidence = (fake_score / max_score) * 100
    confidence = min(confidence, 98)   # cap

    if fake_score >= 3:
        verdict    = "FAKE"
        conf_label = confidence
    elif fake_score == 2:
        verdict    = "SUSPICIOUS"
        conf_label = confidence
    else:
        verdict    = "REAL"
        conf_label = 100 - confidence

    return verdict, round(conf_label, 1), fake_score, max_score, flags


# ─── Visualizations ──────────────────────────────────────────────────────────

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("#0f2044")
    ax.set_facecolor("#0f2044")
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color="#00c8ff", linewidth=0.6, alpha=0.9)
    ax.fill_between(times, y, alpha=0.25, color="#00c8ff")
    ax.set_xlabel("Time (s)", color="#7a9bc7", fontsize=9)
    ax.set_ylabel("Amplitude", color="#7a9bc7", fontsize=9)
    ax.tick_params(colors="#7a9bc7", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a6e")
    ax.set_title("Waveform", color="#b8d0ef", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def plot_mfcc(mfcc_matrix, sr):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor("#0f2044")
    ax.set_facecolor("#0f2044")
    img = librosa.display.specshow(
        mfcc_matrix, sr=sr, x_axis="time", ax=ax,
        cmap="coolwarm"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f")
    ax.set_title("MFCC Heatmap", color="#b8d0ef", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", color="#7a9bc7", fontsize=9)
    ax.set_ylabel("MFCC Coefficient", color="#7a9bc7", fontsize=9)
    ax.tick_params(colors="#7a9bc7", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a6e")
    fig.tight_layout()
    return fig


def plot_zcr(zcr_arr, sr):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("#0f2044")
    ax.set_facecolor("#0f2044")
    frames = np.arange(zcr_arr.shape[1])
    times  = librosa.frames_to_time(frames, sr=sr)
    ax.plot(times, zcr_arr[0], color="#00e5b0", linewidth=0.8)
    ax.axhline(y=0.1, color="#ff4d6d", linewidth=1.2,
               linestyle="--", label="Fake threshold (0.1)")
    ax.legend(facecolor="#0f2044", edgecolor="#1e3a6e",
              labelcolor="#b8d0ef", fontsize=8)
    ax.set_xlabel("Time (s)", color="#7a9bc7", fontsize=9)
    ax.set_ylabel("ZCR", color="#7a9bc7", fontsize=9)
    ax.tick_params(colors="#7a9bc7", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a6e")
    ax.set_title("Zero Crossing Rate", color="#b8d0ef", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def plot_spectral(spectral_arr, sr):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("#0f2044")
    ax.set_facecolor("#0f2044")
    frames = np.arange(spectral_arr.shape[1])
    times  = librosa.frames_to_time(frames, sr=sr)
    ax.plot(times, spectral_arr[0], color="#ffd166", linewidth=0.8)
    ax.fill_between(times, spectral_arr[0], alpha=0.2, color="#ffd166")
    ax.set_xlabel("Time (s)", color="#7a9bc7", fontsize=9)
    ax.set_ylabel("Hz", color="#7a9bc7", fontsize=9)
    ax.tick_params(colors="#7a9bc7", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a6e")
    ax.set_title("Spectral Centroid", color="#b8d0ef", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ─── Report Generator ─────────────────────────────────────────────────────────

def generate_report(filename, features, verdict, confidence, fake_score, max_score, flags):
    report = {
        "file":       filename,
        "timestamp":  datetime.datetime.now().isoformat(),
        "verdict":    verdict,
        "confidence": f"{confidence}%",
        "fake_score": f"{fake_score}/{max_score}",
        "features": {
            "mfcc_mean":      round(features["mfcc_mean"], 4),
            "mfcc_std":       round(features["mfcc_std"], 4),
            "zcr_mean":       round(features["zcr_mean"], 6),
            "spectral_centroid": round(features["spectral_mean"], 2),
            "spectral_bandwidth": round(features["bandwidth"], 2),
            "spectral_rolloff":   round(features["rolloff"], 2),
            "chroma_mean":    round(features["chroma"], 4),
            "rms_energy":     round(features["rms_energy"], 6),
            "tempo_bpm":      round(features["tempo"], 2),
            "duration_sec":   round(features["duration"], 2),
            "sample_rate_hz": features["sample_rate"],
        },
        "flags": {k: {"status": v[0], "detail": v[1]} for k, v in flags.items()},
    }
    return json.dumps(report, indent=2)


# ─── UI ──────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0">
    <div class="main-title">🎙️ Deepfake Audio <span class="brand-cyan">Detector</span></div>
    <div class="subtitle">
        Upload a WAV file — the tool extracts acoustic features and detects
        AI-generated / synthesized audio using signal processing.
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Upload ──
uploaded_file = st.file_uploader(
    "Upload Audio File (.wav)",
    type=["wav"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    st.markdown(f"""
    <div style="margin: 0.5rem 0 1rem 0">
        <span class="pill">📁 {uploaded_file.name}</span>
        <span class="pill">💾 {uploaded_file.size / 1024:.1f} KB</span>
    </div>
    """, unsafe_allow_html=True)

    # Play the audio
    st.audio(uploaded_file, format="audio/wav")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Save to temp file (librosa needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── Extract & Classify ──
    with st.spinner("🔬 Analysing audio features..."):
        try:
            features                                        = extract_features(tmp_path)
            verdict, confidence, fake_score, max_score, flags = classify(features)
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    # ── RESULT BOX ──
    if verdict == "REAL":
        st.markdown(f"""
        <div class="result-real">
            <div class="result-icon">✅</div>
            <div class="result-label" style="color:#00e5b0">REAL AUDIO</div>
            <div class="result-sub">
                Confidence: <b>{confidence}%</b> real &nbsp;|&nbsp;
                Suspicion score: {fake_score}/{max_score}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif verdict == "SUSPICIOUS":
        st.markdown(f"""
        <div class="result-fake" style="border-color:#ffd166; background:#1a1500">
            <div class="result-icon">⚠️</div>
            <div class="result-label" style="color:#ffd166">SUSPICIOUS</div>
            <div class="result-sub">
                Some anomalies detected — may be processed or AI-generated &nbsp;|&nbsp;
                Score: {fake_score}/{max_score}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-fake">
            <div class="result-icon">❌</div>
            <div class="result-label" style="color:#ff4d6d">FAKE / DEEPFAKE AUDIO</div>
            <div class="result-sub">
                Confidence: <b>{confidence}%</b> fake &nbsp;|&nbsp;
                Suspicion score: {fake_score}/{max_score}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Feature Cards ──
    st.markdown("### 📊 Extracted Features")

    col1, col2, col3 = st.columns(3)

    # MFCC
    flag_mfcc   = flags["mfcc"]
    flag_icon   = "🔴" if flag_mfcc[0] == "suspicious" else ("🟡" if flag_mfcc[0] == "warning" else "🟢")
    flag_color  = "#ff4d6d" if flag_mfcc[0] == "suspicious" else ("#ffd166" if flag_mfcc[0] == "warning" else "#00e5b0")
    with col1:
        st.markdown(f"""
        <div class="feat-card">
            <div class="feat-name">MFCC</div>
            <div class="feat-desc">Mel-Frequency Cepstral Coefficients</div>
            <div class="feat-val">{features['mfcc_mean']:.3f}</div>
            <div class="feat-flag" style="color:{flag_color}">{flag_icon} {flag_mfcc[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    # ZCR
    flag_zcr    = flags["zcr"]
    flag_icon2  = "🔴" if flag_zcr[0] == "suspicious" else ("🟡" if flag_zcr[0] == "warning" else "🟢")
    flag_color2 = "#ff4d6d" if flag_zcr[0] == "suspicious" else ("#ffd166" if flag_zcr[0] == "warning" else "#00e5b0")
    with col2:
        st.markdown(f"""
        <div class="feat-card teal">
            <div class="feat-name">ZCR</div>
            <div class="feat-desc">Zero Crossing Rate</div>
            <div class="feat-val">{features['zcr_mean']:.5f}</div>
            <div class="feat-flag" style="color:{flag_color2}">{flag_icon2} {flag_zcr[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Spectral Centroid
    with col3:
        st.markdown(f"""
        <div class="feat-card yellow">
            <div class="feat-name">Spectral Centroid</div>
            <div class="feat-desc">Frequency brightness of audio</div>
            <div class="feat-val">{features['spectral_mean']:.1f} Hz</div>
            <div class="feat-flag" style="color:#7a9bc7">ℹ️ Used for display analysis</div>
        </div>
        """, unsafe_allow_html=True)

    # Extra stats
    st.markdown(f"""
    <div style="margin: 0.5rem 0 1.5rem 0">
        <span class="pill">⏱ Duration: {features['duration']:.2f}s</span>
        <span class="pill">🎵 Sample Rate: {features['sample_rate']} Hz</span>
        <span class="pill">⚡ RMS Energy: {features['rms_energy']:.5f}</span>
        <span class="pill">🎼 Tempo: {features['tempo']:.1f} BPM</span>
        <span class="pill">📡 Bandwidth: {features['bandwidth']:.1f} Hz</span>
        <span class="pill">🔄 Chroma: {features['chroma']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Visualizations ──
    st.markdown("### 📈 Audio Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waveform", "🔥 MFCC Heatmap", "📉 ZCR", "✨ Spectral Centroid"])

    with tab1:
        fig_w = plot_waveform(features["y"], features["sr"])
        st.pyplot(fig_w)
        plt.close(fig_w)

    with tab2:
        fig_m = plot_mfcc(features["mfcc_matrix"], features["sr"])
        st.pyplot(fig_m)
        plt.close(fig_m)

    with tab3:
        fig_z = plot_zcr(features["zcr_arr"], features["sr"])
        st.pyplot(fig_z)
        plt.close(fig_z)

    with tab4:
        fig_s = plot_spectral(features["spectral_arr"], features["sr"])
        st.pyplot(fig_s)
        plt.close(fig_s)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Download Report ──
    st.markdown("### 📄 Download Report")
    report_json = generate_report(
        uploaded_file.name, features,
        verdict, confidence, fake_score, max_score, flags
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name=f"audio_scan_{timestamp}.json",
        mime="application/json",
    )

# ─── No file yet ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; color:#1e3a6e; padding: 4rem 0;">
        <div style="font-size:5rem">🎙️</div>
        <div style="font-size:1.2rem; color:#4a6fa5; margin-top:1rem">
            Upload a <b style="color:#00c8ff">.WAV</b> file above to begin analysis
        </div>
        <div style="font-size:0.85rem; color:#2a4a7a; margin-top:0.6rem">
            Analyses: MFCC • Zero Crossing Rate • Spectral Centroid •
            Spectral Bandwidth • RMS Energy • Waveform Visualization
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works section ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ⚙️ How It Works")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="feat-card">
            <div class="feat-name">1. Upload WAV</div>
            <div class="feat-desc" style="margin-top:6px; font-size:0.9rem; color:#b8d0ef">
                Upload any .WAV audio file — speech, voice recording, or suspicious audio clip.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="feat-card teal">
            <div class="feat-name">2. Feature Extraction</div>
            <div class="feat-desc" style="margin-top:6px; font-size:0.9rem; color:#b8d0ef">
                Librosa extracts MFCC, ZCR, Spectral Centroid, Bandwidth, RMS Energy and more.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="feat-card yellow">
            <div class="feat-name">3. Detection & Report</div>
            <div class="feat-desc" style="margin-top:6px; font-size:0.9rem; color:#b8d0ef">
                Multi-feature scoring gives REAL / SUSPICIOUS / FAKE verdict with confidence score.
            </div>
        </div>
        """, unsafe_allow_html=True)
