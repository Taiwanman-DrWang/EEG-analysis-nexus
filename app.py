import streamlit as st
import pyedflib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft, welch, hilbert, firwin
from scipy.ndimage import uniform_filter1d
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
st.set_page_config(page_title="EEG NextGen Analytics V4", layout="wide")
st.title("🧠 Next-Generation Academic Anesthesia EEG Platform (v4.0)")
st.caption("Core Engine: Global ROI Navigator, Bilateral FFT, and Advanced PAC (Tort's Modulation Index)")

# ==========================================
# 2. Sidebar Configuration
# ==========================================
st.sidebar.header("🎛️ Parameters")
fmin_filter = st.sidebar.slider("Global High-pass (Hz)", 0.1, 1.0, 0.2, 0.1)
fmax_filter = st.sidebar.slider("Global Low-pass (Hz)", 30, 100, 45, 5)
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
    default_targets = [l for l in all_labels if any(k in l.upper() for k in ['L1', 'R1', 'L2', 'R2'])]
    if not default_targets and all_labels: default_targets = [all_labels[0]]
    
    selected_channels = st.sidebar.multiselect("🧠 Select Channels for Analysis", all_labels, default=default_targets)
    
    if not selected_channels:
        st.warning("Please select at least one channel in the sidebar.")
        st.stop()
    
    ch_data_dict = {ch: [] for ch in selected_channels}
    fs = None
    
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
                        
                # Nyquist Defense Filter
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

    full_signals = {ch: np.concatenate(ch_data_dict[ch]) for ch in selected_channels}
    total_duration = len(full_signals[selected_channels[0]]) / fs
    time_axis = np.arange(len(full_signals[selected_channels[0]])) / fs

    # ==========================================
    # 全域 ROI 控制器
    # ==========================================
    st.markdown("---")
    st.subheader("⏱️ Global Region of Interest (ROI) Navigator")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_sec = st.slider("Start Time (Seconds)", 0.0, float(max(0, total_duration - 3.0)), 0.0, 1.0)
    with col_t2:
        max_epoch = float(total_duration - start_sec)
        epoch_len = st.slider("Epoch Duration (Seconds)", 3.0, max(3.0, max_epoch), min(60.0, max_epoch), 1.0)
    
    idx_start = int(start_sec * fs)
    idx_end = min(int((start_sec + epoch_len) * fs), len(full_signals[selected_channels[0]]))
    actual_epoch_len = (idx_end - idx_start) / fs
    
    sliced_signals = {ch: full_signals[ch][idx_start:idx_end] for ch in selected_channels}
    sliced_time_axis = time_axis[idx_start:idx_end]
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad', '#d35400']

    # 建立多功能分頁
    tab_raw, tab_fft, tab_stft, tab_pac = st.tabs([
        "📈 1. Raw Signals", "📊 2. Localized FFT", "⚡ 3. ROI Spectrograms", "🌊 4. PAC (Tort MI)"
    ])

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

    with tab_fft:
        st.header(f"📊 Localized Welch PSD (ROI: {start_sec}s - {start_sec + actual_epoch_len}s)")
        fft_scale = st.radio("Y-Axis Scale:", ["Logarithmic (Log dB)", "Linear ($\mu V^2/Hz$)"], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, ch in enumerate(selected_channels):
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

    with tab_stft:
        st.header(f"⚡ STFT Spectrogram Array (ROI: {start_sec}s - {start_sec + actual_epoch_len}s)")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 3.5 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        for i, ch in enumerate(selected_channels):
            nperseg_val = min(int(fs * win_sec), len(sliced_signals[ch]))
            noverlap_val = int(fs * (win_sec - step_sec))
            if noverlap_val >= nperseg_val: noverlap_val = nperseg_val // 2
            f_stft, t_stft, Zxx = stft(sliced_signals[ch], fs=fs, nperseg=nperseg_val, noverlap=noverlap_val)
            psd_stft = np.abs(Zxx)**2
            im = axes[i].imshow(10 * np.log10(psd_stft + 1e-10), aspect='auto', origin='lower',
                                extent=[start_sec, start_sec + actual_epoch_len, f_stft[0], f_stft[-1]], 
                                vmin=vmin, vmax=vmax, cmap=cmap)
            axes[i].set_ylim(safe_fmin, safe_fmax)
            axes[i].set_ylabel(f"{ch} Freq (Hz)")
        axes[-1].set_xlabel("Absolute Time (Seconds)")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Power (dB)")
        st.pyplot(fig)

# ---------------------------
    # TAB 4: Phase-Amplitude Coupling (PAC - Tort's MI)
    # ---------------------------
    with tab_pac:
        st.header("🌊 Cross-Frequency Phase-Amplitude Coupling (Tort's MI)")
        st.markdown(f"**Analyzing current ROI**: `{start_sec}s` to `{start_sec + actual_epoch_len}s`")
        
        pac_ch = st.selectbox("🎯 Select Target Channel for PAC Analysis", selected_channels)
        
        if st.button("🚀 Calculate PAC for current ROI"):
            with st.spinner("Hilbert Transform and Phase Binning in progress..."):
                raw_roi_sig = sliced_signals[pac_ch]
                
                # 1. 學術級 Nyquist 篩選防禦
                bands = {
                    'Delta': (0.5, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 14),
                    'Beta':  (14, 32)
                }
                filtered_sigs = {}
                nyq = fs / 2
                valid_bands = []
                
                for b_name, (low, high) in bands.items():
                    if high >= nyq - 0.5:
                        st.warning(f"🚫 **頻段失效警告**: {b_name} 頻段上限 ({high}Hz) 已逼近或超過資料物理極限 Nyquist ({nyq:.1f}Hz)。為確保科學嚴謹性，該頻段 PAC 運算已自動捨棄。")
                    else:
                        b, a = butter(3, [low/nyq, high/nyq], btype='bandpass')
                        filtered_sigs[b_name] = filtfilt(b, a, raw_roi_sig)
                        valid_bands.append(b_name)
                
                # 2. 希爾伯特轉換與振幅平滑
                phases, amps = {}, {}
                b_fir = firwin(13, 0.1) 
                
                for b_name, sig in filtered_sigs.items():
                    analytic = hilbert(sig)
                    phases[b_name] = np.angle(analytic)
                    amps[b_name] = filtfilt(b_fir, [1.0], np.abs(analytic))
                
                phase_deg = {b: np.round(np.mod(phases[b], 2*np.pi) * 180 / np.pi) for b in phases}
                
                # 3. 組合設定 (僅篩選科學有效的配對)
                all_pairs = [
                    ('Delta', 'Theta'), ('Delta', 'Alpha'), ('Delta', 'Beta'),
                    ('Theta', 'Alpha'), ('Theta', 'Beta'), ('Alpha', 'Beta')
                ]
                valid_pairs = [pair for pair in all_pairs if pair[0] in valid_bands and pair[1] in valid_bands]
                
                if not valid_pairs:
                    st.error("❌ 採樣率過低，沒有足夠的有效頻段組合可進行 PAC 運算。")
                    st.stop()
                
                mi_results, dist_results = {}, {}
                
                # 4. Tort's 演算法核心
                for p_band, a_band in valid_pairs:
                    ph = phase_deg[p_band]
                    am = amps[a_band]
                    
                    p_dist = np.zeros(72)
                    for k in range(1, 73):
                        idx = np.ceil(ph / 5) == k
                        if np.any(idx):
                            p_dist[k-1] = np.mean(am[idx])
                            
                    p_dist = np.nan_to_num(p_dist)
                    if np.sum(p_dist) > 0: p_dist = p_dist / np.sum(p_dist)
                        
                    P = uniform_filter1d(p_dist, size=13, mode='wrap')
                    if np.sum(P) > 0: P = P / np.sum(P)
                        
                    D = np.log(72) + np.sum(P * np.log(P + 1e-10))
                    MI = D / np.log(72)
                    
                    mi_results[f"{p_band}_{a_band}"] = MI
                    dist_results[f"{p_band}_{a_band}"] = P
                
                # 5. 動態渲染高解析度結果
                st.subheader(f"Coupling Distribution & MI (Channel: {pac_ch})")
                
                # 根據剩餘的有效組合數量動態生成畫布
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                x_axis = np.linspace(-360, 360, 144) 
                
                for i in range(6):
                    if i < len(valid_pairs):
                        pair_name = f"{valid_pairs[i][0]}_{valid_pairs[i][1]}"
                        MI_val = mi_results[pair_name]
                        P_double = np.tile(dist_results[pair_name], 2) 
                        
                        axes[i].plot(x_axis, P_double, color='#2c3e50', lw=2)
                        axes[i].fill_between(x_axis, P_double, np.min(P_double)*0.9, color='#3498db', alpha=0.3)
                        
                        axes[i].set_title(f"{pair_name}\nMI = {MI_val:.5f}", fontweight='bold', fontsize=12)
                        axes[i].set_xticks([-360, -180, 0, 180, 360])
                        axes[i].set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
                        axes[i].set_ylim(np.min(P_double)*0.9, np.max(P_double)*1.1)
                        axes[i].grid(True, linestyle='--', alpha=0.5)
                        
                        if i >= 3 or len(valid_pairs) <= 3: axes[i].set_xlabel("Phase")
                        if i % 3 == 0: axes[i].set_ylabel("Normalized Amplitude")
                    else:
                        axes[i].axis('off') # 隱藏沒有數據的子圖
                        
                plt.tight_layout()
                st.pyplot(fig)
                
                # 6. MI 數值動態 CSV 匯出引擎
                st.divider()
                st.subheader("💾 Export MI Values for current ROI")
                df_mi = pd.DataFrame([mi_results], index=[f"ROI_{start_sec}s_to_{start_sec+actual_epoch_len}s"])
                df_mi.index.name = "ROI_Window"
                
                csv_mi = df_mi.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download PAC Modulation Index (CSV)",
                    data=csv_mi,
                    file_name=f"PAC_MI_{pac_ch}_{start_sec}s.csv",
                    mime="text/csv",
                    help="Exports the Modulation Index (MI) for valid frequency pairs in the current ROI slice."
                )
else:
    st.info("🔌 Pipeline Offline: Upload raw clinical EDF data to activate.")
