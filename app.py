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

# ==========================================
# 1. 頁面基本設定與戰術主旨
# ==========================================
st.set_page_config(page_title="EEG Clinical Core Analytics", layout="wide")
st.title("🧠 次世代臨床麻醉與意識科學電生理分析平台")
st.caption("已整合前輩 EDF 讀取管線 | 拓展 FFT / STFT / Multitaper 多維度觀測站")

# ==========================================
# 2. 側邊欄：互動式參數控制 (臨床硬核參數)
# ==========================================
st.sidebar.header("🎛️ 訊號與演算參數設定")

st.sidebar.subheader("🧼 臨床預處理濾波")
fmin_filter = st.sidebar.slider("高通濾波 (切除基線飄移, Hz)", 0.1, 1.0, 0.2, 0.1)
fmax_filter = st.sidebar.slider("低通濾波 (切除高頻雜訊, Hz)", 30, 100, 45, 5)

st.sidebar.subheader("🌊 時頻分析時域窗 (Windowing)")
win_sec = st.sidebar.slider("分析視窗長度 (Window Size, 秒)", 1, 10, 3, 1)
step_sec = st.sidebar.slider("滑動步長 (Step Size, 秒)", 1, 5, 1, 1)

st.sidebar.subheader("🎨 視覺畫布渲染設定")
vmin = st.sidebar.slider("Power dB 最小值", -60, 0, -40, 5)
vmax = st.sidebar.slider("Power dB 最大值", -20, 30, 10, 5)
cmap = st.sidebar.selectbox("色彩映射 (Colormap)", ['jet', 'viridis', 'plasma', 'magma', 'inferno'])

# ==========================================
# 3. 核心工具函式 (繼承前輩的防禦性工程)
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
            if freqs_all is None:
                freqs_all = freqs
            power_all.append(psd)
        except Exception:
            continue
    if len(power_all) == 0:
        return None, None, None
    return np.stack(power_all, axis=1), freqs_all, (starts + n_win // 2) / fs

# ==========================================
# 4. 檔案上傳與核心管線驅動
# ==========================================
uploaded_files = st.file_uploader("請上傳一個或多個 Sedline EDF 檔案", type="edf", accept_multiple_files=True)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda f: extract_datetime_from_filename(f.name))
    
    # 建立多功能分頁，佈局未來學術結構洞
    tab_raw, tab_fft, tab_stft, tab_dsa, tab_fooof, tab_pac_hht = st.tabs([
        "📈 1. 原始清理時域訊號", 
        "📊 2. 靜態功率譜 (Welch FFT)", 
        "⚡ 3. 短時傅立葉時頻譜 (STFT)", 
        "🦅 4. 前輩多錐窗 (Multitaper DSA)",
        "🚀 5. FOOOF 興奮抑制平衡 (待研發)",
        "🌊 6. PAC & HHT 終極殺招 (待研發)"
    ])
    
    # 資料結構初始化
    all_filtered_eeg = []
    dataset_start_time = None
    cumulative_time_offset = 0
    powers_mne, freqs_mne, all_times_mne = [], None, []
    fs = None
    
    # 第一階段：循序讀取並清理所有 EDF 檔案
    for fobj in uploaded_files:
        fname = fobj.name
        file_datetime = extract_datetime_from_filename(fname)
        if dataset_start_time is None:
            dataset_start_time = file_datetime
            
        tmp_filename = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(fobj.read())
                tmp_filename = tmp_file.name
            
            f = pyedflib.EdfReader(tmp_filename)
            eeg = f.readSignal(0)  # 預設讀取第一通道
            fs = f.getSampleFrequency(0)
            f._close()
        except Exception as e:
            st.error(f"讀取檔案 {fname} 失敗: {e}")
            continue
        finally:
            if tmp_filename and os.path.exists(tmp_filename):
                os.remove(tmp_filename)
                
        # 異常突波剔除與內插 (Artifact Rejection)
        eeg = eeg.astype(np.float32)
        mean, std = np.mean(eeg), np.std(eeg)
        if std > 0:
            mask = (eeg > mean - 3*std) & (eeg < mean + 3*std)
            bad_idx, good_idx = np.where(~mask)[0], np.where(mask)[0]
            if len(bad_idx) > 0 and len(good_idx) > 0:
                interp_func = interp1d(good_idx, eeg[good_idx], bounds_error=False, fill_value="extrapolate")
                eeg[bad_idx] = interp_func(bad_idx)
                
        # 零相位高品質 Butterworth 帶通濾波
        b_hp, a_hp = butter(3, fmin_filter / (fs/2), btype='high')
        b_lp, a_lp = butter(3, fmax_filter / (fs/2), btype='low')
        filtered = filtfilt(b_hp, a_hp, eeg)
        filtered = filtfilt(b_lp, a_lp, filtered)
        all_filtered_eeg.append(filtered)
        
        # 計算前輩的 Multitaper DSA 資料結構
        power, cur_freqs, t_centers = multitaper_dsa_sliding_mne(filtered, fs, win_sec, step_sec, fmin=fmin_filter, fmax=fmax_filter)
        if power is not None:
            if freqs_mne is None: freqs_mne = cur_freqs
            duration = t_centers[-1] - t_centers[0] if len(t_centers) > 1 else 0
            actual_times = [dataset_start_time + timedelta(seconds=sec + cumulative_time_offset) for sec in t_centers]
            all_times_mne.extend(actual_times)
            cumulative_time_offset += duration + 1
            powers_mne.append(power)

    # 確保成功讀取並串接數據
    if all_filtered_eeg:
        full_signal = np.concatenate(all_filtered_eeg)
        total_duration = len(full_signal) / fs
        time_axis_sec = np.arange(len(full_signal)) / fs
        
        # ------------------------------------------
        # TAB 1: 原始清理時域訊號
        # ------------------------------------------
        with tab_raw:
            st.header("📈 預處理後的連續時域波形（第一通道）")
            st.info(f"📊 成功串接數據！採樣率: {fs} Hz | 總長度: {total_duration:.1f} 秒")
            start_plot_sec = st.slider("選擇觀察時間窗（啟始秒數）", 0.0, float(total_duration - 10), 0.0, 1.0)
            
            fig, ax = plt.subplots(figsize=(12, 3))
            idx_start = int(start_plot_sec * fs)
            idx_end = int((start_plot_sec + 10) * fs)
            ax.plot(time_axis_sec[idx_start:idx_end], full_signal[idx_start:idx_end], color='#2c3e50', alpha=0.8)
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Amplitude ($\mu V$)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # ------------------------------------------
        # TAB 2: 靜態功率譜 (Welch FFT)
        # ------------------------------------------
        with tab_fft:
            st.header("📊 傳統 Welch 方法功率譜密度 (PSD)")
            st.markdown("將整段手術大腦訊號壓縮，觀察整體功率分布。可用來檢視麻醉「枯水期」的 Slow 與 Alpha 雙峰。")
            
            # 使用 Welch 方法計算整體頻譜
            nperseg_welch = int(fs * win_sec)
            f_welch, p_welch = welch(full_signal, fs=fs, nperseg=nperseg_welch)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(f_welch, 10 * np.log10(p_welch), color='#e74c3c', lw=2)
            ax.set_xlim(fmin_filter, fmax_filter)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Density (dB/Hz)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        # ------------------------------------------
        # TAB 3: 短時傅立葉時頻譜 (STFT)
        # ------------------------------------------
        with tab_stft:
            st.header("⚡ 短時傅立葉轉換時頻譜 (STFT Spectrogram)")
            st.markdown("動態觀測大腦頻率隨時間發生的相變，能敏銳捕捉傷害感受突波（疼痛刺激）導致的「山洪爆發」。")
            
            nperseg_stft = int(fs * win_sec)
            noverlap_stft = int(fs * (win_sec - step_sec))
            f_stft, t_stft, Zxx_stft = stft(full_signal, fs=fs, nperseg=nperseg_stft, noverlap=noverlap_stft)
            psd_stft = np.abs(Zxx_stft)**2
            
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(10 * np.log10(psd_stft + 1e-10), aspect='auto', origin='lower',
                           extent=[0, total_duration, f_stft[0], f_stft[-1]],
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_ylim(fmin_filter, fmax_filter)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (Seconds)")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            st.pyplot(fig)

        # ------------------------------------------
        # TAB 4: 前輩多錐窗 (Multitaper DSA)
        # ------------------------------------------
        with tab_dsa:
            st.header("🦅 前輩的臨床標準 Multitaper DSA 串接頻譜")
            if powers_mne and all_times_mne and freqs_mne is not None:
                concat_power = np.concatenate(powers_mne, axis=1)
                extent = [mdates.date2num(all_times_mne[0]), mdates.date2num(all_times_mne[-1]), freqs_mne[0], freqs_mne[-1]]
                
                fig, ax = plt.subplots(figsize=(12, 4))
                im = ax.imshow(10 * np.log10(concat_power + 1e-10), aspect='auto', origin='lower', extent=extent,
                               vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Absolute Time')
                ax.set_ylim([fmin_filter, fmax_filter])
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=30)
                fig.colorbar(im, ax=ax, label='Power (dB)')
                st.pyplot(fig)
            else:
                st.warning("DSA 數據處理未完成")
                
        # ------------------------------------------
        # TAB 5 & 6: 預留未來特種修煉接口
        # ------------------------------------------
        with tab_fooof:
            st.header("🚀 FOOOF (SpecParam) 週期與非週期分離模組")
            st.info("🔧 *已預留未來接口*。第 1～2 個月後期，我們將在此處將 Welch 計算出的 PSD 傳入 `specparam`，一鍵擬合非週期性斜率 $\chi$。")
            
        with tab_pac_hht:
            st.header("🌊 PAC & HHT 跨頻率非穩態終極殺招")
            st.info("🔧 *已預留未來接口*。這將是我們在京都行（2027）之前的終極武器：利用 HHT 剔除突發疼痛的偽影，重新定義次世代 PAC 指標。")
else:
    st.info("🔌 請在上方上傳手術室錄製的 EDF 腦波檔案，系統將自動啟動科研級數據管線。")
