import streamlit as st
import pyedflib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft, welch, hilbert, firwin
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import tempfile
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="EEG NextGen Analytics V4.3", layout="wide")
st.title("🧠 Next-Generation Academic Anesthesia EEG Platform (v4.3)")
st.caption("Core Engine: Synced Manual/Slider ROI Navigator, EDF Real-Time Clock Axis (HH:MM:SS), Bilateral FFT, and Adaptive PAC")

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
# 3. Helper Functions (EDF Header & Filename Time)
# ==========================================
def extract_datetime_from_filename(fname):
    match = re.match(r'EEG_(\d{6})_(\d{6})\.edf', fname)
    if match:
        try:
            dt = datetime.strptime(match.group(1) + match.group(2), "%y%m%d%H%M%S")
            return dt if dt.year <= datetime.now().year + 10 else datetime.strptime(f"19{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None

def get_edf_start_datetime(edf_reader, fname):
    """優先從 EDF 內部 Header 讀取真實錄製時間，若失敗則解析檔名"""
    try:
        dt = edf_reader.getStartdatetime()
        if isinstance(dt, datetime):
            return dt
    except Exception:
        pass
    fallback_dt = extract_datetime_from_filename(fname)
    return fallback_dt if fallback_dt is not None else datetime.now().replace(microsecond=0)

# ==========================================
# 4. Main Pipeline
# ==========================================
uploaded_files = st.file_uploader("Upload Sedline EDF files", type="edf", accept_multiple_files=True)

if uploaded_files:
    # 預先依檔名或上傳順序排列
    uploaded_files = sorted(
        uploaded_files,
        key=lambda f: extract_datetime_from_filename(f.name) or datetime.min
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_files[0].read())
        uploaded_files[0].seek(0)
        tmp_filename = tmp_file.name
    
    try:
        f_in = pyedflib.EdfReader(tmp_filename)
        all_labels = f_in.getSignalLabels()
        dataset_start_time = get_edf_start_datetime(f_in, uploaded_files[0].name)
        f_in._close()
    except Exception:
        all_labels = ["Channel 0"]
        dataset_start_time = datetime.now().replace(microsecond=0)
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        
    st.sidebar.markdown("---")
    default_targets = [l for l in all_labels if any(k in l.upper() for k in ['L1', 'R1', 'L2', 'R2'])]
    if not default_targets and all_labels:
        default_targets = [all_labels[0]]
    
    selected_channels = st.sidebar.multiselect("🧠 Select Channels for Analysis", all_labels, default=default_targets)
    
    if not selected_channels:
        st.warning("Please select at least one channel in the sidebar.")
        st.stop()
    
    ch_data_dict = {ch: [] for ch in selected_channels}
    fs = None
    
    for fobj in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(fobj.read())
            fobj.seek(0)
            tmp_filename = tmp_file.name
            
        try:
            f_edf = pyedflib.EdfReader(tmp_filename)
            for ch_name in selected_channels:
                idx = all_labels.index(ch_name)
                if fs is None:
                    fs = f_edf.getSampleFrequency(idx)
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
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    full_signals = {ch: np.concatenate(ch_data_dict[ch]) for ch in selected_channels}
    total_duration = float(len(full_signals[selected_channels[0]]) / fs)
    time_axis = np.arange(len(full_signals[selected_channels[0]])) / fs

    # ==========================================
    # 全域 ROI 控制器 (雙向綁定 Slider 與 手動輸入框)
    # ==========================================
    st.markdown("---")
    st.subheader("⏱️ Global Region of Interest (ROI) Navigator")

    max_start_val = round(max(0.0, total_duration - 3.0), 1)
    slider_max_start = max(0.5, max_start_val)

    # 初始化與邊界防護 (Start Time)
    if "roi_start" not in st.session_state:
        st.session_state.roi_start = 0.0
    if st.session_state.roi_start > max_start_val:
        st.session_state.roi_start = 0.0
    st.session_state.slider_start = st.session_state.roi_start
    st.session_state.num_start = st.session_state.roi_start

    def sync_start_from_slider():
        st.session_state.roi_start = st.session_state.slider_start
        st.session_state.num_start = st.session_state.slider_start

    def sync_start_from_num():
        st.session_state.roi_start = st.session_state.num_start
        st.session_state.slider_start = st.session_state.num_start

    col_s1, col_s2, col_e1, col_e2 = st.columns([2, 1, 2, 1])
    with col_s1:
        st.slider(
            "Start Time Slider (sec)",
            min_value=0.0,
            max_value=float(slider_max_start),
            step=0.5,
            key="slider_start",
            on_change=sync_start_from_slider
        )
    with col_s2:
        st.number_input(
            "⌨️ Start Time (sec)",
            min_value=0.0,
            max_value=float(slider_max_start),
            step=0.5,
            format="%.1f",
            key="num_start",
            on_change=sync_start_from_num
        )

    start_sec = float(st.session_state.roi_start)
    max_epoch_val = round(max(3.0, total_duration - start_sec), 1)
    slider_max_epoch = max(3.5, max_epoch_val)

    # 初始化與邊界防護 (Epoch Duration)
    if "roi_len" not in st.session_state:
        st.session_state.roi_len = min(60.0, max_epoch_val)
    if st.session_state.roi_len > max_epoch_val:
        st.session_state.roi_len = max_epoch_val
    if st.session_state.roi_len < 3.0:
        st.session_state.roi_len = 3.0
    st.session_state.slider_len = st.session_state.roi_len
    st.session_state.num_len = st.session_state.roi_len

    def sync_len_from_slider():
        st.session_state.roi_len = st.session_state.slider_len
        st.session_state.num_len = st.session_state.slider_len

    def sync_len_from_num():
        st.session_state.roi_len = st.session_state.num_len
        st.session_state.slider_len = st.session_state.num_len

    with col_e1:
        st.slider(
            "Epoch Duration Slider (sec)",
            min_value=3.0,
            max_value=float(slider_max_epoch),
            step=0.5,
            key="slider_len",
            on_change=sync_len_from_slider
        )
    with col_e2:
        st.number_input(
            "⌨️ Epoch Duration (sec)",
            min_value=3.0,
            max_value=float(slider_max_epoch),
            step=0.5,
            format="%.1f",
            key="num_len",
            on_change=sync_len_from_num
        )

    epoch_len = float(st.session_state.roi_len)

    # 計算切片範圍與真實時鐘時間
    idx_start = int(start_sec * fs)
    idx_end = min(int((start_sec + epoch_len) * fs), len(full_signals[selected_channels[0]]))
    actual_epoch_len = (idx_end - idx_start) / fs

    roi_clock_start = dataset_start_time + timedelta(seconds=start_sec)
    roi_clock_end = dataset_start_time + timedelta(seconds=start_sec + actual_epoch_len)
    rec_clock_end = dataset_start_time + timedelta(seconds=total_duration)

    st.info(
        f"🕒 **EDF Recording Start**: `{dataset_start_time.strftime('%Y-%m-%d %H:%M:%S')}` (Total: `{total_duration:.1f}s`, Ends: `{rec_clock_end.strftime('%H:%M:%S')}`)  |  "
        f"🎯 **Current ROI Window**: `{start_sec:.1f}s - {start_sec + actual_epoch_len:.1f}s` "
        f"➔ **Actual Clock Time**: `{roi_clock_start.strftime('%H:%M:%S')} - {roi_clock_end.strftime('%H:%M:%S')}`"
    )

    sliced_signals = {ch: full_signals[ch][idx_start:idx_end] for ch in selected_channels}
    sliced_time_axis = time_axis[idx_start:idx_end]
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad', '#d35400']

    # X 軸雙層標籤格式化器：同時顯示秒數與 (HH:MM:SS)
    def dual_time_formatter(x, pos):
        clock_str = (dataset_start_time + timedelta(seconds=float(x))).strftime('%H:%M:%S')
        return f"{x:.1f}s\n({clock_str})"

    # 建立多功能分頁
    tab_raw, tab_fft, tab_stft, tab_pac = st.tabs([
        "📈 1. Raw Signals", "📊 2. Localized FFT", "⚡ 3. ROI Spectrograms", "🌊 4. PAC (Tort MI)"
    ])

    # ---------------------------
    # TAB 1: Raw Signals
    # ---------------------------
    with tab_raw:
        st.header(f"📈 Bilateral Waveforms ({roi_clock_start.strftime('%H:%M:%S')} - {roi_clock_end.strftime('%H:%M:%S')})")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 2.6 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        for i, ch in enumerate(selected_channels):
            axes[i].plot(sliced_time_axis, sliced_signals[ch], color=colors[i % len(colors)], lw=1)
            axes[i].set_ylabel(f"{ch} (μV)")
            axes[i].grid(True, alpha=0.3)
        axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(dual_time_formatter))
        axes[-1].set_xlabel("Relative Time (Seconds) & Actual EDF Clock Time (HH:MM:SS)")
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------
    # TAB 2: Localized FFT
    # ---------------------------
    with tab_fft:
        st.header(f"📊 Localized Welch PSD ({roi_clock_start.strftime('%H:%M:%S')} - {roi_clock_end.strftime('%H:%M:%S')})")
        fft_scale = st.radio("Y-Axis Scale:", ["Logarithmic (Log dB)", "Linear (μV²/Hz)"], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, ch in enumerate(selected_channels):
            nperseg_val = min(int(fs * win_sec), len(sliced_signals[ch]))
            f_welch, p_welch = welch(sliced_signals[ch], fs=fs, nperseg=nperseg_val)
            if "Logarithmic" in fft_scale:
                ax.plot(f_welch, 10 * np.log10(p_welch + 1e-10), color=colors[i % len(colors)], lw=1.5, label=ch)
                ax.set_ylabel("Power Density (dB/Hz)")
            else:
                ax.plot(f_welch, p_welch, color=colors[i % len(colors)], lw=1.5, label=ch)
                ax.set_ylabel("Power (μV²/Hz)")
        ax.set_xlim(safe_fmin, safe_fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # ---------------------------
    # TAB 3: ROI Spectrograms & Band Export
    # ---------------------------
    with tab_stft:
        st.header(f"⚡ STFT Spectrogram Array ({roi_clock_start.strftime('%H:%M:%S')} - {roi_clock_end.strftime('%H:%M:%S')})")
        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 3.6 * len(selected_channels)), sharex=True)
        axes = np.atleast_1d(axes)
        
        export_df_list = []
        clinical_bands = {
            'Delta 0.5-4Hz': (0.5, 4),
            'Theta 4-8Hz': (4, 8),
            'Alpha 8-14Hz': (8, 14),
            'Beta 14-30Hz': (14, 30)
        }

        for i, ch in enumerate(selected_channels):
            nperseg_val = min(int(fs * win_sec), len(sliced_signals[ch]))
            noverlap_val = int(fs * (win_sec - step_sec))
            if noverlap_val >= nperseg_val:
                noverlap_val = nperseg_val // 2
            f_stft, t_stft, Zxx = stft(sliced_signals[ch], fs=fs, nperseg=nperseg_val, noverlap=noverlap_val)
            psd_stft = np.abs(Zxx)**2
            im = axes[i].imshow(
                10 * np.log10(psd_stft + 1e-10),
                aspect='auto',
                origin='lower',
                extent=[start_sec, start_sec + actual_epoch_len, f_stft[0], f_stft[-1]],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap
            )
            axes[i].set_ylim(safe_fmin, safe_fmax)
            axes[i].set_ylabel(f"{ch} Freq (Hz)")

            # 組裝包含相對秒數與真實時鐘時間的 Band Power 資料表
            abs_sec_array = np.round(start_sec + t_stft, 2)
            clock_time_array = [
                (dataset_start_time + timedelta(seconds=float(s))).strftime('%H:%M:%S')
                for s in abs_sec_array
            ]
            band_power_dict = {
                'Channel': ch,
                'Absolute_Time_Sec': abs_sec_array,
                'Actual_Time_HHMMSS': clock_time_array
            }
            for band_name, (f_low, f_high) in clinical_bands.items():
                idx_band = (f_stft >= f_low) & (f_stft < f_high)
                band_power_dict[band_name] = np.sum(psd_stft[idx_band, :], axis=0)

            df_ch = pd.DataFrame(band_power_dict)
            df_ch.index = np.arange(1, len(t_stft) + 1)
            export_df_list.append(df_ch)

        axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(dual_time_formatter))
        axes[-1].set_xlabel("Relative Time (Seconds) & Actual EDF Clock Time (HH:MM:SS)")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Power (dB)")
        st.pyplot(fig)

        final_export_df = pd.concat(export_df_list)
        final_export_df.index.name = "ROI_Relative_Epoch"
        csv_data = final_export_df.to_csv().encode('utf-8')

        st.divider()
        st.subheader("💾 Export Focused Statistical Data (With Clock Time HH:MM:SS)")
        st.download_button(
            label="📥 Download ROI Band Power Matrix (CSV)",
            data=csv_data,
            file_name=f"ROI_{start_sec:.1f}s_{roi_clock_start.strftime('%H%M%S')}_BandPower.csv",
            mime="text/csv"
        )

    # ---------------------------
    # TAB 4: Phase-Amplitude Coupling (PAC - Tort's MI)
    # ---------------------------
    with tab_pac:
        st.header("🌊 Cross-Frequency Phase-Amplitude Coupling (Tort's MI)")
        st.markdown(
            f"**Analyzing current ROI**: `{start_sec:.1f}s` to `{start_sec + actual_epoch_len:.1f}s` "
            f"(`{roi_clock_start.strftime('%H:%M:%S')} - {roi_clock_end.strftime('%H:%M:%S')}`)"
        )
        
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            pac_ch = st.selectbox("🎯 Select Target Channel", selected_channels)
        with col_p2:
            beta_max = st.number_input(
                "Beta Band Upper Limit (Hz)",
                min_value=20.0,
                max_value=50.0,
                value=30.0,
                step=1.0,
                help="Set to 30 Hz for files with 63Hz sampling rate to respect the Nyquist limit (~31.5Hz)."
            )
        
        if st.button("🚀 Calculate PAC for current ROI"):
            with st.spinner("Hilbert Transform and Phase Binning in progress..."):
                raw_roi_sig = sliced_signals[pac_ch]
                
                bands = {
                    'Delta': (0.5, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 14),
                    'Beta':  (14, beta_max)
                }
                filtered_sigs = {}
                nyq = fs / 2
                valid_bands = []
                
                for b_name, (low, high) in bands.items():
                    if high >= nyq - 0.5:
                        st.warning(f"🚫 **頻段失效警告**: {b_name} 頻段上限 ({high}Hz) 逼近物理極限 Nyquist ({nyq:.1f}Hz)，已捨棄。請調低上方 Beta 上限參數。")
                    else:
                        b, a = butter(3, [low / nyq, high / nyq], btype='bandpass')
                        filtered_sigs[b_name] = filtfilt(b, a, raw_roi_sig)
                        valid_bands.append(b_name)
                
                phases, amps = {}, {}
                b_fir = firwin(13, 0.1)
                
                for b_name, sig in filtered_sigs.items():
                    analytic = hilbert(sig)
                    phases[b_name] = np.angle(analytic)
                    amps[b_name] = filtfilt(b_fir, [1.0], np.abs(analytic))
                
                phase_deg = {b: np.round(np.mod(phases[b], 2 * np.pi) * 180 / np.pi) for b in phases}
                
                all_pairs = [
                    ('Delta', 'Theta'), ('Delta', 'Alpha'), ('Delta', 'Beta'),
                    ('Theta', 'Alpha'), ('Theta', 'Beta'), ('Alpha', 'Beta')
                ]
                valid_pairs = [pair for pair in all_pairs if pair[0] in valid_bands and pair[1] in valid_bands]
                
                if not valid_pairs:
                    st.error("❌ 採樣率過低，且 Beta 上限設定過高，無有效頻段可運算。")
                    st.stop()
                
                mi_results, dist_results = {}, {}
                
                for p_band, a_band in valid_pairs:
                    ph = phase_deg[p_band]
                    am = amps[a_band]
                    
                    p_dist = np.zeros(72)
                    for k in range(1, 73):
                        idx = np.ceil(ph / 5) == k
                        if np.any(idx):
                            p_dist[k - 1] = np.mean(am[idx])
                            
                    p_dist = np.nan_to_num(p_dist)
                    if np.sum(p_dist) > 0:
                        p_dist = p_dist / np.sum(p_dist)
                        
                    P = uniform_filter1d(p_dist, size=13, mode='wrap')
                    if np.sum(P) > 0:
                        P = P / np.sum(P)
                        
                    D = np.log(72) + np.sum(P * np.log(P + 1e-10))
                    MI = D / np.log(72)
                    
                    mi_results[f"{p_band}_{a_band}"] = MI
                    dist_results[f"{p_band}_{a_band}"] = P
                
                st.subheader(f"Coupling Distribution & MI (Channel: {pac_ch})")
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                x_axis = np.linspace(-360, 360, 144)
                
                for i in range(6):
                    if i < len(valid_pairs):
                        pair_name = f"{valid_pairs[i][0]}_{valid_pairs[i][1]}"
                        MI_val = mi_results[pair_name]
                        P_double = np.tile(dist_results[pair_name], 2)
                        
                        axes[i].plot(x_axis, P_double, color='#2c3e50', lw=2)
                        axes[i].fill_between(x_axis, P_double, np.min(P_double) * 0.9, color='#3498db', alpha=0.3)
                        
                        axes[i].set_title(f"{pair_name}\nMI = {MI_val:.5f}", fontweight='bold', fontsize=12)
                        axes[i].set_xticks([-360, -180, 0, 180, 360])
                        axes[i].set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
                        axes[i].set_ylim(np.min(P_double) * 0.9, np.max(P_double) * 1.1)
                        axes[i].grid(True, linestyle='--', alpha=0.5)
                        
                        if i >= 3 or len(valid_pairs) <= 3:
                            axes[i].set_xlabel("Phase")
                        if i % 3 == 0:
                            axes[i].set_ylabel("Normalized Amplitude")
                    else:
                        axes[i].axis('off')
                        
                plt.tight_layout()
                st.pyplot(fig)
                
                st.divider()
                st.subheader("💾 Export MI Values for current ROI")
                export_mi_row = {
                    'Channel': pac_ch,
                    'Start_Sec': start_sec,
                    'End_Sec': round(start_sec + actual_epoch_len, 1),
                    'Clock_Start_HHMMSS': roi_clock_start.strftime('%H:%M:%S'),
                    'Clock_End_HHMMSS': roi_clock_end.strftime('%H:%M:%S'),
                    **mi_results
                }
                df_mi = pd.DataFrame([export_mi_row])
                df_mi.index.name = "Record_ID"
                
                csv_mi = df_mi.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download PAC Modulation Index (CSV)",
                    data=csv_mi,
                    file_name=f"PAC_MI_{pac_ch}_{start_sec:.1f}s_{roi_clock_start.strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    help="Exports the Modulation Index (MI) with exact clock timestamps."
                )
else:
    st.info("🔌 Pipeline Offline: Upload raw clinical EDF data to activate.")
