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
st.set_page_config(page_title="EEG NextGen Analytics V3.2", layout="wide")
st.title("🧠 Next-Generation Academic Anesthesia EEG Platform (v3.2)")
st.caption("Core Engine: Global ROI Navigator, Bilateral Analysis, Nyquist Defense, and Band Power Export")

# ==========================================
# 2. Sidebar Configuration
# ==========================================
st.sidebar.header("🎛️ Parameters")
fmin_filter = st.sidebar.slider("High-pass (Hz)", 0.1, 1.0, 0.2, 0.1)
fmax_filter = st.sidebar.slider("Low-pass (Hz)", 30, 100, 45, 5)
win_sec = st.sidebar.slider("Window Size (sec)", 1, 10, 3, 1, help="STFT/Welch resolution. Will automatically adapt if ROI is shorter.")
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

    # ==========================================
    # 全域 ROI 控制器 (Global ROI Navigator)
    # ==========================================
    st.markdown("---")
    st.subheader("⏱️ Global Region of Interest (ROI) Navigator")
    st.markdown("調整此區塊，將同步連動下方的 **Raw Signals**, **FFT**, 以及 **STFT Spectrograms** 與數據輸出。")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_sec = st.slider("Start Time (Seconds)", 0.0, float(max(0, total_duration - 3.0)), 0.0, 1.0)
    with col_t2:
        max_epoch = float(total_duration - start_sec)
        epoch_len = st.slider("Epoch Duration (Seconds)", 3.0, max(3.0, max_epoch), min(60.0, max_epoch), 1.0)
    
    # 計算 ROI 切片的陣列邊界
    idx_start = int(start_sec * fs)
    idx_end = min(int((start_sec + epoch_len) * fs), len(full_signals[selected_channels[0]]))
    actual_epoch_len = (idx_end - idx_start) / fs
    
    # 建立 ROI 專用的切割數據庫
    sliced_signals = {ch: full_signals[ch][idx_start:idx_end] for ch in selected_channels}
    sliced_time_axis = time_axis[idx_start:idx_end]
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad', '#d35400']

    # 建立多功能分頁
    tab_raw, tab_fft, tab_stft, tab_fooof = st.tabs([
        "📈 1. Raw Signals", "📊 2. Localized FFT Power", "⚡ 3. ROI STFT Spectrograms", "🚀 4. Export & FOOOF"
    ])

    # ---------------------------
    # TAB 1: Raw Signals (連動 ROI)
    # ---------------------------
    with tab_raw:
        st.header(f"📈 Bilateral Waveforms (ROI: {start_sec}s - {start_sec + actual_epoch_len}s)")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 2.5 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        
        for i, ch in enumerate(selected_channels):
            axes[i].plot(sliced_time_axis, sliced_signals[ch], color=colors[i%len(colors)], lw=1)
            axes[i].set_ylabel(ch)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Absolute Time (Seconds)")
        st.pyplot(fig)

    # ---------------------------
    # TAB 2: Localized Bilateral FFT (連動 ROI)
    # ---------------------------
    with tab_fft:
        st.header(f"📊 Localized Welch PSD (ROI: {start_sec}s - {start_sec + actual_epoch_len}s)")
        fft_scale = st.radio("Y-Axis Scale:", ["Logarithmic (Log dB)", "Linear ($\mu V^2/Hz$)"], horizontal=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, ch in enumerate(selected_channels):
            # 動態防禦：確保 Welch 分析視窗不會大於切片長度
            nperseg_val = min(int(fs * win_sec), len(sliced_signals[ch]))
            f_welch, p_welch = welch(sliced_signals[ch], fs=fs, nperseg=nperseg_val)
            
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
    # TAB 3: ROI STFT & Export (連動 ROI)
    # ---------------------------
    with tab_stft:
        st.header(f"⚡ STFT Spectrogram Array (ROI: {start_sec}s - {start_sec + actual_epoch_len}s)")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 3.5 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        
        export_df_list = []
        clinical_bands = {
            'Delta 0.5-4Hz': (0.5, 4),
            'Theta 4-8Hz': (4, 8),
            'Alpha 8-14Hz': (8, 14),
            'Beta 14-32Hz': (14, 32)
        }
        
        for i, ch in enumerate(selected_channels):
            # 動態防禦：計算相容的 window 與 overlap
            nperseg_val = min(int(fs * win_sec), len(sliced_signals[ch]))
            noverlap_val = int(fs * (win_sec - step_sec))
            if noverlap_val >= nperseg_val: noverlap_val = nperseg_val // 2
            
            f_stft, t_stft, Zxx = stft(sliced_signals[ch], fs=fs, nperseg=nperseg_val, noverlap=noverlap_val)
            psd_stft = np.abs(Zxx)**2
            
            # STFT 影像渲染 (X軸映射回真實時間範圍)
            im = axes[i].imshow(10 * np.log10(psd_stft + 1e-10), aspect='auto', origin='lower',
                                extent=[start_sec, start_sec + actual_epoch_len, f_stft[0], f_stft[-1]], 
                                vmin=vmin, vmax=vmax, cmap=cmap)
            axes[i].set_ylim(safe_fmin, safe_fmax)
            axes[i].set_ylabel(f"{ch} Freq (Hz)")
            
            # 頻段濃縮
            band_power_dict = {'Channel': ch}
            for band_name, (f_low, f_high) in clinical_bands.items():
                idx_band = (f_stft >= f_low) & (f_stft < f_high)
                band_power_dict[band_name] = np.sum(psd_stft[idx_band, :], axis=0)
                
            df_ch = pd.DataFrame(band_power_dict)
            df_ch.index = np.arange(1, len(t_stft) + 1)
            export_df_list.append(df_ch)
            
        axes[-1].set_xlabel("Absolute Time within Full Recording (Seconds)")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Power (dB)")
        st.pyplot(fig)
        
        # 整合匯出引擎 (Tidy Data) 專門針對目前的 ROI
        final_export_df = pd.concat(export_df_list)
        final_export_df.index.name = "ROI_Relative_Epochs"
        csv_data = final_export_df.to_csv().encode('utf-8')
        
        st.divider()
        st.subheader("💾 Export Focused Statistical Data (For current ROI only)")
        st.download_button(
            label="📥 Download ROI Band Power Matrix (CSV)",
            data=csv_data,
            file_name=f"ROI_{start_sec}s_to_{start_sec + actual_epoch_len}s_BandPower.csv",
            mime="text/csv",
            help="Exports data exclusively for the selected ROI slice. Perfect for comparing specific clinical events (e.g., pre-incision vs post-incision)."
        )
else:
    st.info("🔌 Pipeline Offline: Upload raw clinical EDF data to activate.")
