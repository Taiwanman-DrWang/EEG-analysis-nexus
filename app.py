import streamlit as st
import pyedflib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft, welch
from scipy.interpolate import interp1d
from mne.time_frequency import psd_array_multitaper
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import tempfile
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="EEG NextGen Analytics V3.1", layout="wide")
st.title("🧠 Next-Generation Academic Anesthesia EEG Platform (v3.1)")
st.caption("Core Engine: Bilateral Analysis, Nyquist Defense, and Canonical Band Power Export")

# ==========================================
# 2. Sidebar Configuration
# ==========================================
st.sidebar.header("🎛️ Parameters")
fmin_filter = st.sidebar.slider("High-pass (Hz)", 0.1, 1.0, 0.2, 0.1)
fmax_filter = st.sidebar.slider("Low-pass (Hz)", 30, 100, 45, 5)
win_sec = st.sidebar.slider("Window Size (sec)", 1, 10, 3, 1)
step_sec = st.sidebar.slider("Step Size (sec)", 1, 5, 1, 1)
vmin = st.sidebar.slider("Min Power (dB)", -60, 0, -40, 5)
vmax = st.sidebar.slider("Max Power (dB)", -20, 30, 10, 5)
cmap = st.sidebar.selectbox("Colormap", ['jet', 'viridis', 'plasma', 'magma'])

# ==========================================
# 3. Helper Functions
# ==========================================
def extract_datetime(fname):
    match = re.match(r'EEG_(\d{6})_(\d{6})\.edf', fname)
    if match:
        try:
            dt = datetime.strptime(match.group(1) + match.group(2), "%y%m%d%H%M%S")
            return dt if dt.year <= datetime.now().year + 10 else datetime.strptime(f"19{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.now()

# ==========================================
# 4. Main Pipeline
# ==========================================
uploaded_files = st.file_uploader("Upload Sedline EDF files", type="edf", accept_multiple_files=True)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda f: extract_datetime(f.name))
    
    # 智能偵測導聯
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_files[0].read())
        uploaded_files[0].seek(0)
        tmp_filename = tmp_file.name
    
    try:
        f_in = pyedflib.EdfReader(tmp_filename)
        all_labels = f_in.getSignalLabels()
        f_in._close()
    except Exception:
        all_labels = ["Channel 0"]
    finally:
        if os.path.exists(tmp_filename): os.remove(tmp_filename)
        
    st.sidebar.markdown("---")
    # 自動尋找預設的雙側導聯
    default_targets = [l for l in all_labels if any(k in l.upper() for k in ['L1', 'R1', 'L2', 'R2'])]
    if not default_targets and all_labels: default_targets = [all_labels[0]]
    
    selected_channels = st.sidebar.multiselect("🧠 Select Channels for Bilateral Analysis", all_labels, default=default_targets)
    
    if not selected_channels:
        st.warning("Please select at least one channel in the sidebar.")
        st.stop()

    tab_raw, tab_fft, tab_stft, tab_fooof = st.tabs([
        "📈 1. Raw Signals", "📊 2. FFT Power (Bilateral)", "⚡ 3. STFT Spectrograms", "🚀 4. Export & FOOOF"
    ])
    
    ch_data_dict = {ch: [] for ch in selected_channels}
    fs = None
    dataset_start_time = extract_datetime(uploaded_files[0].name)
    
    # 資料讀取與濾波
    for fobj in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(fobj.read())
            tmp_filename = tmp_file.name
            
        try:
            f_edf = pyedflib.EdfReader(tmp_filename)
            for ch_name in selected_channels:
                idx = all_labels.index(ch_name)
                if fs is None: fs = f_edf.getSampleFrequency(idx)
                
                eeg = f_edf.readSignal(idx).astype(np.float32)
                
                # Artifact Rejection
                mean, std = np.mean(eeg), np.std(eeg)
                if std > 0:
                    mask = (eeg > mean - 3*std) & (eeg < mean + 3*std)
                    bad_idx, good_idx = np.where(~mask)[0], np.where(mask)[0]
                    if len(bad_idx) > 0 and len(good_idx) > 0:
                        interp_func = interp1d(good_idx, eeg[good_idx], bounds_error=False, fill_value="extrapolate")
                        eeg[bad_idx] = interp_func(bad_idx)
                        
                # Nyquist Defense
                nyq = fs / 2
                safe_fmin, safe_fmax = min(fmin_filter, nyq - 0.5), min(fmax_filter, nyq - 0.5)
                
                b_hp, a_hp = butter(3, safe_fmin / nyq, btype='high')
                b_lp, a_lp = butter(3, safe_fmax / nyq, btype='low')
                filtered = filtfilt(b_hp, a_hp, eeg)
                filtered = filtfilt(b_lp, a_lp, filtered)
                
                ch_data_dict[ch_name].append(filtered)
            f_edf._close()
        finally:
            if os.path.exists(tmp_filename): os.remove(tmp_filename)

    # 串接訊號
    full_signals = {ch: np.concatenate(ch_data_dict[ch]) for ch in selected_channels}
    total_duration = len(full_signals[selected_channels[0]]) / fs
    time_axis = np.arange(len(full_signals[selected_channels[0]])) / fs

    # ---------------------------
    # TAB 1: Raw Signals
    # ---------------------------
    with tab_raw:
        st.header("📈 Bilateral Preprocessed Waveforms")
        start_sec = st.slider("Select 10-sec Epoch (Start Time)", 0.0, float(total_duration - 10), 0.0, 1.0)
        
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 2.5 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        
        idx_start, idx_end = int(start_sec * fs), int((start_sec + 10) * fs)
        colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad', '#d35400']
        
        for i, ch in enumerate(selected_channels):
            axes[i].plot(time_axis[idx_start:idx_end], full_signals[ch][idx_start:idx_end], color=colors[i%len(colors)], lw=1)
            axes[i].set_ylabel(ch)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (Seconds)")
        st.pyplot(fig)

    # ---------------------------
    # TAB 2: Bilateral FFT
    # ---------------------------
    with tab_fft:
        st.header("📊 Global Welch PSD (Spatial Overlay)")
        fft_scale = st.radio("Y-Axis Scale:", ["Logarithmic (Log dB)", "Linear ($\mu V^2/Hz$)"], horizontal=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, ch in enumerate(selected_channels):
            f_welch, p_welch = welch(full_signals[ch], fs=fs, nperseg=int(fs * win_sec))
            
            if "Logarithmic" in fft_scale:
                ax.plot(f_welch, 10 * np.log10(p_welch + 1e-10), color=colors[i%len(colors)], lw=1.5, label=ch)
                ax.set_ylabel("Power Density (dB/Hz)")
            else:
                ax.plot(f_welch, p_welch, color=colors[i%len(colors)], lw=1.5, label=ch)
                ax.set_ylabel("Power ($\mu V^2/Hz$)")
                
        ax.set_xlim(safe_fmin, safe_fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # ---------------------------
    # TAB 3: Bilateral STFT & Export
    # ---------------------------
    with tab_stft:
        st.header("⚡ STFT Spectrogram Array")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 3.5 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        
        export_df_list = []
        
        # 建立要濃縮的頻段字典 (Canonical Frequency Bands)
        clinical_bands = {
            'Delta 0.5-4Hz': (0.5, 4),
            'Theta 4-8Hz': (4, 8),
            'Alpha 8-14Hz': (8, 14),
            'Beta 14-32Hz': (14, 32)
        }
        
        for i, ch in enumerate(selected_channels):
            f_stft, t_stft, Zxx = stft(full_signals[ch], fs=fs, nperseg=int(fs*win_sec), noverlap=int(fs*(win_sec-step_sec)))
            psd_stft = np.abs(Zxx)**2
            
            # 繪製原圖
            im = axes[i].imshow(10 * np.log10(psd_stft + 1e-10), aspect='auto', origin='lower',
                                extent=[0, total_duration, f_stft[0], f_stft[-1]], vmin=vmin, vmax=vmax, cmap=cmap)
            axes[i].set_ylim(safe_fmin, safe_fmax)
            axes[i].set_ylabel(f"{ch} Freq (Hz)")
            
            # ---------------------------
            # 科研頻段濃縮與匯出邏輯 (Band Power Aggregation)
            # ---------------------------
            band_power_dict = {'Channel': ch}
            for band_name, (f_low, f_high) in clinical_bands.items():
                # 找出落在該頻段內的所有頻率小數點索引
                idx_band = (f_stft >= f_low) & (f_stft < f_high)
                # 將這些細碎頻率的能量「垂直加總 (axis=0)」，濃縮為單一波段的總能量
                band_power_dict[band_name] = np.sum(psd_stft[idx_band, :], axis=0)
                
            df_ch = pd.DataFrame(band_power_dict)
            
            # 時間軸正規化：將複雜的中心秒數，強制轉為 1, 2, 3... 的相對紀元 (Epoch)
            df_ch.index = np.arange(1, len(t_stft) + 1)
            
            export_df_list.append(df_ch)
            
        axes[-1].set_xlabel("Time (Seconds)")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Power (dB)")
        st.pyplot(fig)
        
        # 整合匯出引擎 (Tidy Data)
        final_export_df = pd.concat(export_df_list)
        final_export_df.index.name = "Relative_Time (Seconds/Epochs)"
        csv_data = final_export_df.to_csv().encode('utf-8')
        
        st.divider()
        st.subheader("💾 Statistical Export (Clinical Bands)")
        st.download_button(
            label="📥 Download Multi-Channel Band Power Matrix (CSV)",
            data=csv_data,
            file_name="Bilateral_BandPower_Matrix.csv",
            mime="text/csv",
            help="Tidy Data Format: Rows are normalized relative time (1, 2, 3...). Columns are grouped into canonical clinical bands. Values represent the absolute raw sum power for that band."
        )
else:
    st.info("🔌 Pipeline Offline: Upload raw clinical EDF data to activate.")
