import streamlit as st
import pyedflib
import numpy as np
from scipy.signal import butter, filtfilt, stft, welch
from scipy.interpolate import interp1d
from mne.time_frequency import psd_array_multitaper
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import tempfile
import os
import pandas as pd

# ==========================================
# 1. Page Configuration & Professional Branding
# ==========================================
st.set_page_config(page_title="EEG NextGen Analytics Platform v2.1", layout="wide")
st.title("🧠 Next-Generation Academic & Clinical Anesthesia EEG Analytics Platform (v2.1)")
st.caption("Core Engine: Automated Channel Discovery, Dual-Scale FFT, and Nyquist-Compliant Defensive Filtering Layer")

# ==========================================
# 2. Sidebar Configuration (Interactive Control Panel)
# ==========================================
st.sidebar.header("🎛️ Signal & Algorithm Parameters")

st.sidebar.subheader("🧼 Clinical Preprocessing Filter")
fmin_filter = st.sidebar.slider("High-pass Filter (Baseline Drift Removal, Hz)", 0.1, 1.0, 0.2, 0.1)
fmax_filter = st.sidebar.slider("Low-pass Filter (High-frequency Noise Removal, Hz)", 30, 100, 45, 5)

st.sidebar.subheader("🌊 Time-Frequency Windowing")
win_sec = st.sidebar.slider("Window Size (Seconds)", 1, 10, 3, 1)
step_sec = st.sidebar.slider("Step Size / Shift (Seconds)", 1, 5, 1, 1)

st.sidebar.subheader("🎨 Spectrogram Visual Rendering")
vmin = st.sidebar.slider("Spectrogram Min Power (dB)", -60, 0, -40, 5)
vmax = st.sidebar.slider("Spectrogram Max Power (dB)", -20, 30, 10, 5)
cmap = st.sidebar.selectbox("Colormap Selection", ['jet', 'viridis', 'plasma', 'magma', 'inferno'])

# ==========================================
# 3. Core Mathematical and Defensive Engineering Modules
# ==========================================
def extract_datetime_from_filename(fname):
    match = re.match(r'EEG_(\d{6})_(\d{6})\.edf', fname)
    if match:
        date_str, time_str = match.group(1), match.group(2)
        try:
            dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
            if dt.year > datetime.now().year + 10:
                dt = datetime.strptime(f"19{date_str}{time_str}", "%Y%m%d%H%M%S")
        except ValueError:
            return datetime.now()
        return dt
    return datetime.now()

def multitaper_dsa_sliding_mne(eeg, fs, win_sec, step_sec, fmin=1, fmax=45):
    n_win = int(win_sec * fs)
    n_step = int(step_sec * fs)
    n_total = len(eeg)
    starts = np.arange(0, n_total - n_win + 1, n_step)
    if len(starts) == 0:
        return None, None, None
    power_all, freqs_all = [], None
    for start in starts:
        segment = eeg[start:start + n_win]
        try:
            psd, freqs = psd_array_multitaper(
                segment, sfreq=fs, fmin=fmin, fmax=fmax,
                adaptive=False, normalization='full', verbose=False)
            if freqs_all is None: freqs_all = freqs
            power_all.append(psd)
        except Exception:
            continue
    if len(power_all) == 0:
        return None, None, None
    return np.stack(power_all, axis=1), freqs_all, (starts + n_win // 2) / fs

# ==========================================
# 4. File I/O and Dynamic Channel Interrogation
# ==========================================
uploaded_files = st.file_uploader("Upload one or more Sedline/Clinical EDF files", type="edf", accept_multiple_files=True)

if uploaded_files:
    # Sort files chronologically
    uploaded_files = sorted(uploaded_files, key=lambda f: extract_datetime_from_filename(f.name))
    
    # Interrogate the first file to extract clinical labels
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_files[0].read())
        uploaded_files[0].seek(0)  # Reset pointer
        tmp_filename = tmp_file.name
    
    try:
        f_inspect = pyedflib.EdfReader(tmp_filename)
        channel_labels = f_inspect.getSignalLabels()
        f_inspect._close()
    except Exception as e:
        st.error(f"Failed to parse EDF channel metadata: {e}")
        channel_labels = ["Channel 0"]
    finally:
        if os.path.exists(tmp_filename): os.remove(tmp_filename)
        
    # Render Channel Selector in Sidebar
    st.sidebar.markdown("---")
    selected_channel_label = st.sidebar.selectbox("🧠 Select Target Channel", channel_labels)
    target_channel_idx = channel_labels.index(selected_channel_label) if selected_channel_label in channel_labels else 0

    # Academic Layout Structure
    tab_raw, tab_fft, tab_stft, tab_dsa, tab_fooof, tab_pac_hht = st.tabs([
        "📈 1. Preprocessed Raw Waveform", 
        "📊 2. Static Power Spectrum (Dual-Scale FFT)", 
        "⚡ 3. Short-Time Fourier Transform (STFT)", 
        "🦅 4. Multitaper DSA (Legacy Pipeline)",
        "🚀 5. FOOOF (E/I Balance Fit) [R&D]",
        "🌊 6. PAC & HHT Advanced [R&D]"
    ])
    
    # Initialization
    all_filtered_eeg = []
    dataset_start_time = None
    cumulative_time_offset = 0
    powers_mne, freqs_mne, all_times_mne = [], None, []
    fs = None
    clamping_triggered = False  # Track if safety clamping occurs
    
    # Processing Pipeline Iteration
    for fobj in uploaded_files:
        fname = fobj.name
        file_datetime = extract_datetime_from_filename(fname)
        if dataset_start_time is None: dataset_start_time = file_datetime
            
        tmp_filename = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(fobj.read())
                tmp_filename = tmp_file.name
            
            f = pyedflib.EdfReader(tmp_filename)
            eeg = f.readSignal(target_channel_idx)  # Dynamic channel extraction
            fs = f.getSampleFrequency(target_channel_idx)
            f._close()
        except Exception as e:
            st.error(f"Error reading file {fname}: {e}")
            continue
        finally:
            if tmp_filename and os.path.exists(tmp_filename): os.remove(tmp_filename)
                
        # Defense Layer 1: Artifact Rejection & Interpolation
        eeg = eeg.astype(np.float32)
        mean, std = np.mean(eeg), np.std(eeg)
        if std > 0:
            mask = (eeg > mean - 3*std) & (eeg < mean + 3*std)
            bad_idx, good_idx = np.where(~mask)[0], np.where(mask)[0]
            if len(bad_idx) > 0 and len(good_idx) > 0:
                interp_func = interp1d(good_idx, eeg[good_idx], bounds_error=False, fill_value="extrapolate")
                eeg[bad_idx] = interp_func(bad_idx)
                
        # Defense Layer 2: Nyquist-Compliant Safety Clamping
        nyquist = fs / 2
        safe_fmin = min(fmin_filter, nyquist - 0.5)
        safe_fmax = min(fmax_filter, nyquist - 0.5)
        if safe_fmin >= safe_fmax:
            safe_fmin = max(0.1, safe_fmax - 0.5)
            
        if safe_fmax < fmax_filter or safe_fmin > fmin_filter:
            clamping_triggered = True
            
        # Phase-Preserved Zero-Phase Butterworth Filter Pipeline using Safe Cutoffs
        b_hp, a_hp = butter(3, safe_fmin / nyquist, btype='high')
        b_lp, a_lp = butter(3, safe_fmax / nyquist, btype='low')
        filtered = filtfilt(b_hp, a_hp, eeg)
        filtered = filtfilt(b_lp, a_lp, filtered)
        all_filtered_eeg.append(filtered)
        
        # Legacy Multitaper Generation using Safe Cutoffs
        power, cur_freqs, t_centers = multitaper_dsa_sliding_mne(filtered, fs, win_sec, step_sec, fmin=safe_fmin, fmax=safe_fmax)
        if power is not None:
            if freqs_mne is None: freqs_mne = cur_freqs
            duration = t_centers[-1] - t_centers[0] if len(t_centers) > 1 else 0
            actual_times = [dataset_start_time + timedelta(seconds=sec + cumulative_time_offset) for sec in t_centers]
            all_times_mne.extend(actual_times)
            cumulative_time_offset += duration + 1
            powers_mne.append(power)

    # Core Visualization Rendering Block
    if all_filtered_eeg:
        full_signal = np.concatenate(all_filtered_eeg)
        total_duration = len(full_signal) / fs
        time_axis_sec = np.arange(len(full_signal)) / fs
        
        # Display localized warning if defensive clamping was activated
        if clamping_triggered:
            st.warning(f"⚠️ **Nyquist Safety Clamping Active**: Your file's sampling rate is {fs} Hz (Nyquist: {fs/2} Hz). Cutoff parameters have been dynamically restricted to safe ranges (Highpass: {safe_fmin:.1f} Hz, Lowpass: {safe_fmax:.1f} Hz) to prevent filter system crashes.")

        # ------------------------------------------
        # TAB 1: Preprocessed Raw Waveform
        # ------------------------------------------
        with tab_raw:
            st.header(f"📈 Preprocessed Time-Series Waveform [{selected_channel_label}]")
            st.info(f"📊 Signal Successfully Concatenated! Total Duration: {total_duration:.1f} s | Native fs: {fs} Hz")
            start_plot_sec = st.slider("Select 10-Second Epoch (Start Time, Seconds)", 0.0, float(total_duration - 10), 0.0, 1.0)
            
            fig, ax = plt.subplots(figsize=(12, 3))
            idx_start, idx_end = int(start_plot_sec * fs), int((start_plot_sec + 10) * fs)
            ax.plot(time_axis_sec[idx_start:idx_end], full_signal[idx_start:idx_end], color='#2c3e50', alpha=0.8)
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Amplitude ($\mu V$)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # ------------------------------------------
        # TAB 2: Static Power Spectrum (Dual-Scale FFT & SEF95)
        # ------------------------------------------
        with tab_fft:
            st.header("📊 Welch Power Spectral Density & Phenomenological Metrics")
            
            fft_scale = st.radio("Select Y-Axis Power Scale Mode (Axis Transform):", ["Linear Scale ($\mu V^2/Hz$)", "Logarithmic Scale (Log dB) - Recommended for FOOOF R&D"], horizontal=True)
            
            # Welch Computation
            nperseg_welch = int(fs * win_sec)
            f_welch, p_welch = welch(full_signal, fs=fs, nperseg=nperseg_welch)
            
            # Clinical Derivative: Spectral Edge Frequency (SEF95) Computation via Safe Boundaries
            idx_band = (f_welch >= safe_fmin) & (f_welch <= safe_fmax)
            f_trimmed, p_trimmed = f_welch[idx_band], p_welch[idx_band]
            
            if len(p_trimmed) > 0:
                cum_power = np.cumsum(p_trimmed)
                total_power = cum_power[-1]
                sef95_idx = np.where(cum_power >= 0.95 * total_power)[0][0]
                sef95_val = f_trimmed[sef95_idx]
            else:
                sef95_val = safe_fmax
            
            # Secondary Feature: Frontal Alpha Synchronization Ratio (8-12Hz)
            idx_alpha = (f_welch >= 8) & (f_welch <= 12)
            total_band_power = np.sum(p_welch[idx_band]) if np.sum(p_welch[idx_band]) > 0 else 1e-10
            alpha_power_ratio = np.sum(p_welch[idx_alpha]) / total_band_power * 100
            
            # Academic KPI Display Panel
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("📊 Spectral Edge Frequency (SEF 95%)", f"{sef95_val:.2f} Hz", help="95% of total signal power is concentrated below this frequency threshold.")
            m_col2.metric("🌊 Frontal Alpha Synchronization Ratio (8-12 Hz)", f"{alpha_power_ratio:.1f} %", help="Quantifies the relative presence of coherent macrocircuit rhythms.")

            # Figure Export Engine
            fig, ax = plt.subplots(figsize=(10, 4))
            if "Logarithmic" in fft_scale:
                ax.plot(f_welch, 10 * np.log10(p_welch + 1e-10), color='#e74c3c', lw=2, label="Welch PSD")
                ax.axvline(sef95_val, color='#2c3e50', linestyle='--', lw=1.5, label=f'SEF95 ({sef95_val:.1f} Hz)')
                ax.set_ylabel("Power Density (dB/Hz)")
            else:
                ax.plot(f_welch, p_welch, color='#e74c3c', lw=2, label="Welch PSD")
                ax.axvline(sef95_val, color='#2c3e50', linestyle='--', lw=1.5, label=f'SEF95 ({sef95_val:.1f} Hz)')
                ax.set_ylabel("Power ($\mu V^2/Hz$)")
                
            ax.set_xlim(safe_fmin, safe_fmax)
            ax.set_xlabel("Frequency (Hz)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        # ------------------------------------------
        # TAB 3: Short-Time Fourier Transform (STFT)
        # ------------------------------------------
        with tab_stft:
            st.header("⚡ Short-Time Fourier Transform Spectrogram (STFT)")
            
            nperseg_stft = int(fs * win_sec)
            noverlap_stft = int(fs * (win_sec - step_sec))
            f_stft, t_stft, Zxx_stft = stft(full_signal, fs=fs, nperseg=nperseg_stft, noverlap=noverlap_stft)
            psd_stft = np.abs(Zxx_stft)**2
            
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(10 * np.log10(psd_stft + 1e-10), aspect='auto', origin='lower',
                           extent=[0, total_duration, f_stft[0], f_stft[-1]],
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_ylim(safe_fmin, safe_fmax)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (Seconds)")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            st.pyplot(fig)
            # --- STFT 數據匯出引擎 (Data Export Engine) ---
            st.divider()
            st.subheader("💾 Export STFT Matrix for Statistical Analysis")
            
            # 將矩陣轉置 (.T)，讓 Row = Time, Column = Frequency
            df_stft = pd.DataFrame(psd_stft.T, index=t_stft, columns=np.round(f_stft, 2))
            df_stft.index.name = "Time (Seconds)"
            
            # 轉換為 CSV 格式 (utf-8 編碼確保跨國系統不亂碼)
            csv_stft = df_stft.to_csv().encode('utf-8')
            
            st.download_button(
                label="📥 Download STFT Raw Power Matrix (CSV)",
                data=csv_stft,
                file_name="STFT_RawPower_Matrix.csv",
                mime="text/csv",
                help="Rows represent time windows (seconds), Columns represent frequency bins (Hz). Values are Raw Power (not dB) to facilitate accurate downstream statistical averaging and relative band power calculations."
            )
        # ------------------------------------------
        # TAB 4: Multitaper DSA (Legacy Pipeline)
        # ------------------------------------------
        with tab_dsa:
            st.header("🦅 Clinical-Grade Multitaper Density Spectral Array (DSA)")
            if powers_mne and all_times_mne and freqs_mne is not None:
                concat_power = np.concatenate(powers_mne, axis=1)
                extent = [mdates.date2num(all_times_mne[0]), mdates.date2num(all_times_mne[-1]), freqs_mne[0], freqs_mne[-1]]
                
                fig, ax = plt.subplots(figsize=(12, 4))
                im = ax.imshow(10 * np.log10(concat_power + 1e-10), aspect='auto', origin='lower', extent=extent,
                               vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Absolute Time')
                ax.set_ylim([safe_fmin, safe_fmax])
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=30)
                fig.colorbar(im, ax=ax, label='Power (dB)')
                st.pyplot(fig)
                # --- DSA 數據匯出引擎 (Data Export Engine) ---
                st.divider()
                st.subheader("💾 Export Multitaper DSA Matrix for Statistical Analysis")
                # 將絕對時間轉換為容易閱讀的字串格式
                time_strs = [t.strftime('%Y-%m-%d %H:%M:%S') for t in all_times_mne]
                # 建立 DataFrame，Row = Absolute Time, Column = Frequency
                df_dsa = pd.DataFrame(concat_power.T, index=time_strs, columns=np.round(freqs_mne, 2))
                df_dsa.index.name = "Absolute Time"
                csv_dsa = df_dsa.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Multitaper DSA Raw Power Matrix (CSV)",
                    data=csv_dsa,
                    file_name="Multitaper_DSA_RawPower_Matrix.csv",
                    mime="text/csv",
                    help="Rows represent absolute clinical time, Columns represent frequency bins (Hz). Values are Raw Power (not dB)."
                )        
            else:
                st.warning("Multitaper arrays could not be generated. Check temporal continuity metrics.")
        # ------------------------------------------
        # TAB 5 & 6: Research Structural Holes (Future Modules)
        # ------------------------------------------
        with tab_fooof:
            st.header("🚀 FOOOF (SpecParam) Aperiodic Parameterization Fitting Module")
            st.info("🔧 *R&D Pipeline Interface Allocated*. In late Stage 1, we will integrate the `specparam` library here to isolate the aperiodic slope from the periodic peak oscillators.")
            
        with tab_pac_hht:
            st.header("🌊 Phase-Amplitude Coupling (PAC) & Hilbert-Huang Transform (HHT)")
            st.info("🔧 *Advanced Breakthrough Pipeline Allocated*. This forms the core of our KPUM 2027 technical arsenal.")
else:
    st.info("🔌 Pipeline Offline: Upload raw clinical EDF data blocks above to activate the automated preprocessing and spectrum interrogation cluster.")
