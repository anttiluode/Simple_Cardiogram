"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SIMPLE CARDIOGRAM · High-Speed Acoustic Monitor                         ║
║  PerceptionLab     · Live Threshold Tracking & Fast Rendering            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Features:                                                               ║
║   • VISUAL LIMITER – Dynamic yellow line shows exactly where beats fire  ║
║   • HIGH SPEED – Optimized FFTs, strict decimation, and lighter graphics ║
║   • CALIBRATION RACK – Dial in the detection math for your own body      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import scipy.signal
import threading
import warnings
from collections import deque

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
BG       = "#060810"
PANEL    = "#0c1220"
SIDEBAR  = "#0a1018"
BORDER   = "#1a2840"
CYAN     = "#00e5ff"
GREEN    = "#00ff88"
RED      = "#ff2244"
YELLOW   = "#ffd600"
ORANGE   = "#ff7000"
TXT      = "#c8d8f0"
TXT2     = "#6080a0"

FONT_MONO  = ("Consolas", 9)
FONT_HEAD  = ("Consolas", 11, "bold")
FONT_HUGE  = ("Consolas", 28, "bold")
FONT_LARGE = ("Consolas", 16, "bold")

# ─────────────────────────────────────────────────────────────────
#  SONIFICATION ENGINE
# ─────────────────────────────────────────────────────────────────
class SonificationEngine:
    def __init__(self, sample_rate, output_device=None):
        self.fs = sample_rate
        self.output_device = output_device
        self.output_stream = None
        self.audio_queue = deque(maxlen=10)
        self.is_playing = False
        self.lock = threading.Lock()
        self.volume = 0.8
        self._start_output_stream()

    def _start_output_stream(self):
        try:
            self.output_stream = sd.OutputStream(
                device=self.output_device,
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=self._audio_callback
            )
            self.output_stream.start()
            self.is_playing = True
        except Exception as e:
            print(f"Output stream error: {e}")
            self.is_playing = False

    def _audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            if len(self.audio_queue) > 0:
                data = self.audio_queue.popleft()
                if len(data) >= frames:
                    outdata[:] = data[:frames].reshape(-1, 1)
                else:
                    outdata[:len(data)] = data.reshape(-1, 1)
                    outdata[len(data):] = 0
            else:
                outdata.fill(0)

    def process_audio(self, audio_chunk):
        if not self.is_playing or self.volume == 0:
            return
        
        nyq = self.fs / 2
        sos = scipy.signal.butter(3, [15/nyq, 150/nyq], btype='band', output='sos')
        processed = scipy.signal.sosfilt(sos, audio_chunk)
        
        mx = np.max(np.abs(processed))
        if mx > 0:
            processed = (processed / mx) * self.volume
            
        with self.lock:
            self.audio_queue.append(processed.astype(np.float32))

    def stop(self):
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
        self.is_playing = False

# ─────────────────────────────────────────────────────────────────
#  DEVICE PICKER
# ─────────────────────────────────────────────────────────────────
class DevicePicker(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("SIMPLE CARDIOGRAM · Select Input")
        self.configure(bg=BG)
        self.geometry("720x480")
        self.resizable(False, False)
        self.result = None

        devices = sd.query_devices()
        input_devs = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

        tk.Label(self, text="SIMPLE CARDIOGRAM", fg=RED, bg=BG, font=("Consolas", 18, "bold")).pack(pady=(20, 2))
        tk.Label(self, text="Select your audio interface / contact microphone", fg=TXT2, bg=BG, font=FONT_MONO).pack(pady=(0, 10))

        lf = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        lf.pack(fill=tk.BOTH, expand=True, padx=24, pady=4)

        sc = tk.Scrollbar(lf, bg=PANEL)
        sc.pack(side=tk.RIGHT, fill=tk.Y)

        self.lb = tk.Listbox(lf, bg=PANEL, fg=TXT, font=FONT_MONO, selectbackground=RED, selectforeground=BG,
                             activestyle='none', highlightthickness=0, yscrollcommand=sc.set, borderwidth=0)
        self.lb.pack(fill=tk.BOTH, expand=True)
        sc.config(command=self.lb.yview)

        self.dev_ids = []
        default_dev = sd.default.device[0]
        default_idx = None
        for i, (did, d) in enumerate(input_devs):
            mark = "★ DEFAULT  " if did == default_dev else "           "
            lbl = f" {mark}[{did:2d}]  {d['name'][:52]:<52}  {int(d['default_samplerate'])} Hz"
            self.lb.insert(tk.END, lbl)
            self.dev_ids.append(did)
            if did == default_dev: default_idx = i
        if default_idx is not None:
            self.lb.selection_set(default_idx)
            self.lb.see(default_idx)

        sr_frame = tk.Frame(self, bg=BG)
        sr_frame.pack(fill=tk.X, padx=24, pady=8)
        
        tk.Label(sr_frame, text="Sample Rate:", fg=TXT2, bg=BG, font=FONT_MONO).pack(side=tk.LEFT)
        self.sr_var = tk.StringVar(value="48000")
        for sr in ["22050", "44100", "48000"]:
            tk.Radiobutton(sr_frame, text=sr, variable=self.sr_var, value=sr, fg=CYAN, bg=BG, selectcolor=BG, activebackground=BG, font=FONT_MONO).pack(side=tk.LEFT, padx=8)

        tk.Label(sr_frame, text="Output:", fg=TXT2, bg=BG, font=FONT_MONO).pack(side=tk.LEFT, padx=(20, 4))
        
        self._out_devs = []
        self._out_names = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                self._out_devs.append(i)
                self._out_names.append(f"[{i}] {d['name'][:28]}")
                
        self.out_var = tk.StringVar(value=self._out_names[0] if self._out_names else "Default")
        ttk.Combobox(sr_frame, textvariable=self.out_var, values=self._out_names, width=26, state="readonly").pack(side=tk.LEFT, padx=4)

        btn = tk.Frame(self, bg=BG)
        btn.pack(pady=15)
        tk.Button(btn, text="▶  START MONITORING", bg=RED, fg=BG, font=("Consolas", 12, "bold"), padx=20, pady=8, relief=tk.FLAT, cursor="hand2", command=self._confirm).pack()

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._use_default)

    def _confirm(self):
        sel = self.lb.curselection()
        if not sel: return
        did = self.dev_ids[sel[0]]
        fs  = int(self.sr_var.get())
        
        selected_out = self.out_var.get()
        if selected_out in self._out_names:
            out = self._out_devs[self._out_names.index(selected_out)]
        else:
            out = None
            
        self.result = (did, fs, out)
        self.destroy()

    def _use_default(self):
        self.result = (sd.default.device[0], 48000, sd.default.device[1])
        self.destroy()

# ─────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
class SimpleCardiogram:
    def __init__(self, root, device_id, fs, output_device):
        self.root = root
        self.device_id = device_id
        self.fs = fs
        
        self.root.title(f"SIMPLE CARDIOGRAM  ·  {fs} Hz")
        self.root.geometry("1400x850")
        self.root.configure(bg=BG)

        # Buffer holds 5 seconds of audio
        self.buf_sec = 5.0
        self.buf_len = int(fs * self.buf_sec)
        self.audio_buf = np.zeros(self.buf_len, dtype=np.float32)
        self._alock = threading.Lock()
        self._apos = 0

        self.sonifier = SonificationEngine(fs, output_device)
        self.is_running = True
        self.stream = None

        # Calibration Variables
        self.sonify_vol_var = tk.DoubleVar(value=0.8)
        self.env_smooth_var = tk.DoubleVar(value=6.0) # Hz LPF
        self.max_bpm_var = tk.IntVar(value=120)       # Prevents double-counting S1/S2
        self.squelch_var = tk.DoubleVar(value=0.8)    # Peak height multiplier

        # OPTIMIZATION: Lighter Plot Setup
        self.wfall_rows = 100 # Reduced from 150 for faster plotting
        self.f_max = 150.0  
        self.n_fft = 2048     # Halved from 4096 for massive speed boost
        self.hz_per_bin = (self.fs / 2.0) / (self.n_fft // 2)
        self.target_bins = int(self.f_max / self.hz_per_bin)
        
        self.wfall_buf = deque([np.full(self.target_bins, -80.0) for _ in range(self.wfall_rows)], maxlen=self.wfall_rows)

        self._build_ui()
        self._start_audio()

    def _build_ui(self):
        # ── Calibration Rack (Top Bar)
        top = tk.Frame(self.root, bg=PANEL, height=50)
        top.pack(fill=tk.X, side=tk.TOP)
        
        self.btn_pause = tk.Button(top, text="⏸  PAUSE", bg=RED, fg="#fff", font=FONT_HEAD, padx=10, relief=tk.FLAT, cursor="hand2", command=self._toggle_stream)
        self.btn_pause.pack(side=tk.LEFT, padx=10, pady=8)

        def make_slider(parent, label, var, from_, to, res, width=120):
            frame = tk.Frame(parent, bg=PANEL)
            frame.pack(side=tk.LEFT, padx=15)
            tk.Label(frame, text=label, fg=TXT2, bg=PANEL, font=FONT_MONO).pack(side=tk.TOP, anchor="w")
            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill=tk.X)
            tk.Scale(row, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, resolution=res, length=width, bg=PANEL, fg=CYAN, troughcolor=BORDER, highlightthickness=0, showvalue=False, command=self._force_update).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, fg=CYAN, bg=PANEL, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=5)

        make_slider(top, "🎧 Volume", self.sonify_vol_var, 0.0, 1.0, 0.05)
        
        tk.Frame(top, bg=BORDER, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        tk.Label(top, text="⚙️ CALIBRATION:", fg=YELLOW, bg=PANEL, font=FONT_HEAD).pack(side=tk.LEFT, padx=10)
        
        make_slider(top, "Envelope Smooth (Hz)", self.env_smooth_var, 1.0, 15.0, 0.5, 140)
        make_slider(top, "Max BPM Limit", self.max_bpm_var, 40, 220, 1, 140)
        make_slider(top, "Peak Threshold", self.squelch_var, 0.1, 3.0, 0.05, 140)

        # ── Main Content Area
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill=tk.BOTH, expand=True)

        # ── Sidebar (Metrics)
        self.sidebar = tk.Frame(content, bg=SIDEBAR, width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        def make_metric(parent, label_text, color):
            f = tk.Frame(parent, bg=SIDEBAR)
            f.pack(fill=tk.X, pady=(20, 5), padx=15)
            tk.Label(f, text=label_text, fg=TXT2, bg=SIDEBAR, font=FONT_HEAD, anchor="w").pack(fill=tk.X)
            val_lbl = tk.Label(f, text="---", fg=color, bg=SIDEBAR, font=FONT_HUGE, anchor="w")
            val_lbl.pack(fill=tk.X)
            return val_lbl

        tk.Label(self.sidebar, text="❤️ CARDIAC VITALS", fg=RED, bg=SIDEBAR, font=FONT_LARGE).pack(pady=(25, 10))
        tk.Frame(self.sidebar, bg=BORDER, height=2).pack(fill=tk.X, padx=15)

        self.bpm_lbl = make_metric(self.sidebar, "HEART RATE (BPM)", RED)
        self.hrv_lbl = make_metric(self.sidebar, "HRV / SDNN (ms)", CYAN)
        
        rf = tk.Frame(self.sidebar, bg=SIDEBAR)
        rf.pack(fill=tk.X, pady=(20, 5), padx=15)
        tk.Label(rf, text="RHYTHM ANALYSIS", fg=TXT2, bg=SIDEBAR, font=FONT_HEAD, anchor="w").pack(fill=tk.X)
        self.rhythm_lbl = tk.Label(rf, text="Analyzing...", fg=YELLOW, bg=SIDEBAR, font=("Consolas", 12, "bold"), anchor="w", wraplength=260, justify=tk.LEFT)
        self.rhythm_lbl.pack(fill=tk.X, pady=5)

        help_txt = ("CALIBRATION GUIDE:\n\n"
                    "1. Smooth (Hz): Smears 'lub-dub'\n"
                    "   into 1 wave. Lower to fix\n"
                    "   double-counting (try ~4Hz).\n\n"
                    "2. Max BPM: Sets speed limit.\n"
                    "   Set 20 BPM above true rate.\n\n"
                    "3. Threshold: Push slider until\n"
                    "   the yellow line sits just\n"
                    "   below your pulse peaks.")
        tk.Label(self.sidebar, text=help_txt, fg=TXT2, bg=SIDEBAR, font=("Consolas", 9), justify=tk.LEFT).pack(side=tk.BOTTOM, pady=20, padx=15, anchor="w")

        # ── Plots
        self.plot_frame = tk.Frame(content, bg=BG)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.fig = plt.Figure(figsize=(10, 8), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06, hspace=0.25)
        gs = GridSpec(2, 1, figure=self.fig, height_ratios=[1.5, 1])

        # Waveform Plot
        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_wave.set_facecolor(BG)
        self.ax_wave.set_title("5-Second Acoustic Mechanical Trace", color=CYAN, fontsize=11, loc='left')
        self.ax_wave.tick_params(colors=TXT2, labelsize=8)
        for sp in self.ax_wave.spines.values(): sp.set_color(BORDER)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xlim(0, 5.0)

        self.line_raw, = self.ax_wave.plot([], [], color=TXT2, lw=1.0, alpha=0.6)
        self.line_env, = self.ax_wave.plot([], [], color=CYAN, lw=2.0)
        self.fill_env = None
        
        # THE LIMITER LINE
        self.line_thresh = self.ax_wave.axhline(-99, color=YELLOW, ls='--', lw=1.5, alpha=0.8)
        self.scatter_peaks = self.ax_wave.scatter([], [], color=RED, s=60, zorder=5)

        # Waterfall Plot (0-150Hz)
        self.ax_wfall = self.fig.add_subplot(gs[1, 0])
        self.ax_wfall.set_facecolor(BG)
        self.ax_wfall.set_title("Low-Frequency Spectrogram (0 - 150 Hz)", color=CYAN, fontsize=11, loc='left')
        self.ax_wfall.tick_params(colors=TXT2, labelsize=8)
        for sp in self.ax_wfall.spines.values(): sp.set_color(BORDER)
        
        self.img_wfall = self.ax_wfall.imshow(
            np.zeros((self.wfall_rows, self.target_bins)),
            cmap='inferno', aspect='auto', origin='upper',
            extent=[0, self.f_max, 0, self.wfall_rows],
            vmin=-60, vmax=0
        )
        self.ax_wfall.set_xlabel("Frequency (Hz)", color=TXT2, fontsize=9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _force_update(self, val):
        self.sonifier.volume = float(self.sonify_vol_var.get())

    def _audio_cb(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._alock:
            space = len(mono)
            pos   = self._apos
            if pos + space <= self.buf_len:
                self.audio_buf[pos:pos+space] = mono
            else:
                first = self.buf_len - pos
                self.audio_buf[pos:]   = mono[:first]
                self.audio_buf[:space-first] = mono[first:]
            self._apos = (pos + space) % self.buf_len

    def _start_audio(self):
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.fs,
                callback=self._audio_cb,
                dtype='float32',
                blocksize=1024
            )
            self.stream.start()
            self._update_loop()
        except Exception as e:
            messagebox.showerror("Audio Error", str(e))

    def _toggle_stream(self):
        if self.is_running:
            self.stream.stop()
            self.is_running = False
            self.btn_pause.config(text="▶  RESUME", bg=GREEN, fg=BG)
            self.sonifier.stop()
        else:
            self.stream.start()
            self.is_running = True
            self.btn_pause.config(text="⏸  PAUSE", bg=RED, fg="#fff")
            self.sonifier._start_output_stream()
            self._update_loop()

    def _update_loop(self):
        if not self.is_running:
            return

        with self._alock:
            pos = self._apos
            if pos >= self.buf_len:
                raw = self.audio_buf.copy()
            else:
                raw = np.concatenate([self.audio_buf[pos:], self.audio_buf[:pos]])

        # ── 1. Process for Sonification & Waterfall (short chunk)
        # Using 2048 for faster FFTs
        short_chunk = raw[-2048:] 
        self.sonifier.process_audio(short_chunk)
        
        win = np.hanning(len(short_chunk))
        spec_full = 20 * np.log10(np.abs(np.fft.rfft(short_chunk * win, n=self.n_fft)) + 1e-9)
        self.wfall_buf.append(spec_full[:self.target_bins])
        self.img_wfall.set_data(np.array(self.wfall_buf))

        # ── 2. Cardio Analysis (Full 5-second window)
        nyq = self.fs / 2
        sos_bp = scipy.signal.butter(3, [15/nyq, 100/nyq], btype='band', output='sos')
        filtered = scipy.signal.sosfilt(sos_bp, raw)

        # Dynamic Envelope Extraction based on slider
        rectified = np.abs(filtered)
        lpf_hz = self.env_smooth_var.get()
        sos_lp = scipy.signal.butter(2, lpf_hz/nyq, btype='low', output='sos')
        env = scipy.signal.sosfilt(sos_lp, rectified)

        max_env = np.max(env) if np.max(env) > 0 else 1
        disp_raw = np.clip((filtered / max_env) * 0.5, -1, 1)
        disp_env = np.clip(env / max_env, 0, 1)

        # Dynamic Peak Detection based on sliders
        # We calculate the median baseline to create a robust dynamic threshold
        baseline = np.median(disp_env)
        threshold = baseline + (np.std(disp_env) * self.squelch_var.get())
        
        # Max BPM dictates the minimum distance between peaks
        max_bpm = max(40, self.max_bpm_var.get())
        min_distance_samples = int(self.fs * (60.0 / max_bpm))
        
        peaks, _ = scipy.signal.find_peaks(disp_env, height=threshold, distance=min_distance_samples)

        # ── 3. Update Metrics
        if len(peaks) >= 2:
            ibi_sec = np.diff(peaks) / self.fs
            bpm = 60.0 / np.median(ibi_sec)
            hrv = np.std(ibi_sec) * 1000

            self.bpm_lbl.config(text=f"{bpm:.0f}")
            self.bpm_lbl.config(fg=YELLOW if bpm < 50 else (ORANGE if bpm > 100 else RED))
            self.hrv_lbl.config(text=f"{hrv:.0f}")

            if bpm < 50: rhythm, col = "Bradycardia (Slow)", YELLOW
            elif bpm > 100: rhythm, col = "Tachycardia (Fast)", ORANGE
            elif hrv > 80: rhythm, col = "High Variability", RED
            else: rhythm, col = "Normal Sinus Rhythm", GREEN
            self.rhythm_lbl.config(text=rhythm, fg=col)
        else:
            self.bpm_lbl.config(text="---", fg=TXT2)
            self.hrv_lbl.config(text="---", fg=TXT2)
            self.rhythm_lbl.config(text="Searching for pulse...", fg=TXT2)

        # ── 4. Update Waveform Plot (Highly Optimized)
        # Force decimation to a maximum of 500 points for hyper-fast rendering
        ds_factor = max(1, len(disp_env) // 500)
        t_axis = np.linspace(0, 5.0, len(disp_env[::ds_factor]))
        
        self.line_raw.set_data(t_axis, disp_raw[::ds_factor])
        self.line_env.set_data(t_axis, disp_env[::ds_factor])
        
        # Update Limiter Line
        self.line_thresh.set_ydata([threshold, threshold])
        
        if self.fill_env: self.fill_env.remove()
        self.fill_env = self.ax_wave.fill_between(t_axis, 0, disp_env[::ds_factor], color=CYAN, alpha=0.2)

        if len(peaks) > 0:
            peak_times = peaks / self.fs
            peak_vals = disp_env[peaks]
            self.scatter_peaks.set_offsets(np.c_[peak_times, peak_vals])
        else:
            self.scatter_peaks.set_offsets(np.c_[-1, -1])

        self.canvas.draw_idle()
        # Fast refresh rate
        self.root.after(30, self._update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    picker = DevicePicker(root)
    root.wait_window(picker)

    if picker.result is None:
        root.destroy()
    else:
        device_id, fs, output_device = picker.result
        root.deiconify()
        app = SimpleCardiogram(root, device_id, fs, output_device)

        def _on_close():
            app.is_running = False
            if app.stream:
                try: app.stream.stop(); app.stream.close()
                except Exception: pass
            app.sonifier.stop()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        root.mainloop()