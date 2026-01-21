#v00readerV1.py
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import numpy as np
import pyproj  # for coordinate transformations
from pyproj import Transformer
import pandas as pd
from typing import Optional, Literal, Union, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numpy import sinc
from pyhht.emd import EMD
from scipy.signal import hilbert
import re
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from sklearn.decomposition import FastICA
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from tkinter import ttk
import pywt
import cv2
from skimage import feature




# import matplotlib.pyplot as plt


# from scipy.signal import hilbert


# =============================================================================
#                           GPR2DLoader Class
# =============================================================================
class GPR2DLoader:
    """
    GPR data loader class
    Supports .v00, .d00, .dt
    Auto-loads HDR and GEOX/GEC when available
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================
    def __init__(self):

        # ---------------- Core data ----------------
        self.data: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.xyz: Optional[pd.DataFrame] = None

        # ---------------- File info ----------------
        self.base_path: Optional[str] = None
        self.line_name: Optional[str] = None
        self.data_type: Optional[str] = None
        self.file_path: Optional[str] = None
        self.file_size: Optional[int] = None
        self.header_size: Optional[int] = None

        # ---------------- Data geometry ----------------
        self.samples_per_trace: Optional[int] = None
        self.num_traces: Optional[int] = None
        self.shape: Optional[tuple] = None
        self.dtype: Optional[np.dtype] = None

        # ---------------- HDR parameters ----------------
        self.hdr_info: Optional[Dict] = None
        self._sample_interval_s: Optional[float] = None
        self._velocity: Optional[float] = None
        self._x_cell_m: Optional[float] = None
        self._num_traces_hdr: Optional[int] = None
        self._samples_per_trace_hdr: Optional[int] = None

        # ---------------- GEO / GPS ----------------
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None
        self.coord_type: Optional[str] = None
        self.y_coordinate: Optional[float] = None

        # ---------------- Processing history ----------------
        self.process_history: list[str] = []

    # =========================================================
    # PROCESS TRACKING
    # =========================================================
    def add_process(self, tag: str):
        if tag not in self.process_history:
            self.process_history.append(tag)

    # =========================================================
    # DATA LOADING
    # =========================================================
    def load_data(
        self,
        base_path: str,
        line_name: str,
        data_type: Literal["v00", "d00", "dt"] = "v00",
        samples_per_trace: Optional[int] = None,
        num_traces: Optional[int] = None,
        verbose: bool = True
    ) -> np.ndarray:

        self.base_path = base_path
        self.line_name = line_name
        self.data_type = data_type.lower()

        self._load_hdr_00_auto(base_path, line_name)

        filename = f"{line_name}.{data_type.upper()}"
        self.file_path = os.path.join(base_path, filename)

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(self.file_path)

        self.file_size = os.path.getsize(self.file_path)

        if verbose:
            print(f"Loading {self.file_path}")

        if self.data_type == "v00":
            self.data = self._load_v00(self.file_path, samples_per_trace, num_traces)
        elif self.data_type in ("d00", "dt"):
            self.data = self._load_d00(self.file_path, samples_per_trace, num_traces)
        else:
            raise ValueError("Unsupported data type")

        self._load_geox_auto(base_path, line_name)

        if self.hdr_info:
            self.get_hdr_parameters()
            self.depth = self.get_depth_axis()

        self.process_history = ["RAW"]
        return self.data

    # =========================================================
    # V00 / D00 / DT LOADERS
    # =========================================================
    def _load_v00(self, file_path, samples_per_trace, num_traces):

        dtype = np.int16
        self.dtype = dtype
        dtype_size = np.dtype(dtype).itemsize

        if samples_per_trace is None:
            samples_per_trace = 512

        if num_traces is None:
            est_header = 1024
            est_data = self.file_size - est_header
            num_traces = est_data // (samples_per_trace * dtype_size)

        expected_data_size = samples_per_trace * num_traces * dtype_size
        self.header_size = self.file_size - expected_data_size

        with open(file_path, "rb") as f:
            f.seek(self.header_size)
            raw = np.fromfile(f, dtype=dtype)

        expected = samples_per_trace * num_traces
        if raw.size != expected:
            raw = np.pad(raw[:expected], (0, max(0, expected - raw.size)))

        self.data = raw.reshape(num_traces, samples_per_trace).T
        self.samples_per_trace, self.num_traces = self.data.shape
        self.shape = self.data.shape
        return self.data

    def _load_d00(self, file_path, *_):

        dtype = np.int16
        self.dtype = dtype

        header_size = 5140
        rh_nsamp = 514

        vec = np.fromfile(file_path, dtype=dtype)
        data = vec[header_size // 2:]

        num_traces = data.size // rh_nsamp
        reshaped = data[:num_traces * rh_nsamp].reshape(num_traces, rh_nsamp)

        self.data = reshaped[:, 2:].T
        self.samples_per_trace, self.num_traces = self.data.shape
        self.shape = self.data.shape
        self.header_size = header_size
        return self.data

    # =========================================================
    # HDR HANDLING
    # =========================================================
    def _load_hdr_00_auto(self, base_path, line_name):
        for ext in ("hdr_00", "hdr_dt", "HDR_00", "HDR_DT"):
            path = os.path.join(base_path, f"{line_name}.{ext}")
            if os.path.exists(path):
                self.hdr_info = self._load_hdr_00(path)
                return
        self.hdr_info = None

    def _load_hdr_00(self, hdr_path):
        res = {}
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f]

        i = 0
        while i < len(lines):
            if lines[i].startswith("<") and lines[i].endswith(">"):
                tag = lines[i][1:-1]
                i += 1
                vals = [self._num(v) for v in lines[i].split()]
                res[tag] = vals if len(vals) > 1 else vals[0]
            i += 1

        self._sample_interval_s = res.get("Y_TIME_CELL")
        self._velocity = res.get("PROP_VEL")
        self._x_cell_m = res.get("X_CELL")

        camp = res.get("CAMP", [None, None])
        if isinstance(camp, list) and len(camp) >= 2:
            self._num_traces_hdr = int(camp[0])
            self._samples_per_trace_hdr = int(camp[1])

        return res

    @staticmethod
    def _num(x):
        try:
            return int(x) if "." not in x and "E" not in x.upper() else float(x)
        except ValueError:
            return x
    def get_hdr_parameters(self):
        """
        Extract commonly used parameters from HDR file
        """
        if self.hdr_info is None:
            return None

        self._sample_interval_s = self.hdr_info.get("Y_TIME_CELL", self._sample_interval_s)
        self._velocity = self.hdr_info.get("PROP_VEL", self._velocity)
        self._x_cell_m = self.hdr_info.get("X_CELL", self._x_cell_m)

        camp = self.hdr_info.get("CAMP")
        if isinstance(camp, (list, tuple)) and len(camp) >= 2:
            self._num_traces_hdr = int(camp[0])
            self._samples_per_trace_hdr = int(camp[1])

        return {
            "dt": self._sample_interval_s,
            "velocity": self._velocity,
            "dx": self._x_cell_m,
            "ntraces": self._num_traces_hdr,
            "nsamples": self._samples_per_trace_hdr
        }
    def get_metadata(self):
        """
        Return metadata dictionary for GUI, plotting, export
        """
        meta = {
            "line_name": self.line_name,
            "data_type": self.data_type,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "header_size": self.header_size,
            "shape": self.shape,
            "samples_per_trace": self.samples_per_trace,
            "num_traces": self.num_traces,
            "dtype": str(self.dtype),
            "process_history": self.process_history.copy(),
        }

        # HDR-derived parameters
        meta.update({
            "dt": self._sample_interval_s,
            "velocity": self._velocity,
            "dx": self._x_cell_m,
            "hdr_ntraces": self._num_traces_hdr,
            "hdr_nsamples": self._samples_per_trace_hdr,
        })

        # GEOX / GPS
        if self.xyz is not None:
            meta["has_geox"] = True
            meta["y_coordinate"] = self.y_coordinate
        else:
            meta["has_geox"] = False

        return meta

    # =========================================================
    # GEOX / GPS
    # =========================================================
    def _load_geox_auto(self, base_path, line_name):
        path = os.path.join(base_path, f"{line_name}.geox")
        if os.path.exists(path):
            self.xyz = self._load_geox(path)
            if len(self.xyz):
                self.y_coordinate = float(self.xyz["Y"].iloc[0])

    def _load_geox(self, file_path):
        rows = []
        with open(file_path, "r", errors="ignore") as f:
            for ln in f:
                if ln.strip() and not ln.startswith("<"):
                    try:
                        vals = list(map(float, ln.split(",")[:7]))
                        rows.append(vals + [0.0])
                    except Exception:
                        continue
        return pd.DataFrame(rows, columns=["Marker","X","Y","Z","Lat","Lon","Alt","Time"])

    # =========================================================
    # DEPTH AXIS
    # =========================================================
    def get_depth_axis(self):
        if self._sample_interval_s is None or self._velocity is None:
            return None
        t = np.arange(self.data.shape[0]) * self._sample_interval_s
        self.depth = t * self._velocity / 2
        return self.depth

    # =========================================================
    # PROCESSING METHODS (TRACKED)
    # =========================================================
    
    # =========================================================
    # Background Removal
    # =========================================================
    def background_removal(self, br_type="full", filter_length=200, start_scan=None, end_scan=None):
        if self.data is None:
            return
        d = self.data.astype(float)
        if br_type == "full":
            d -= np.mean(d, axis=1, keepdims=True)
        elif br_type == "adaptive":
            for i in range(d.shape[1]):
                i1, i2 = max(0, i-filter_length//2), min(d.shape[1], i+filter_length//2)
                d[:, i] -= np.mean(d[:, i1:i2], axis=1)
        elif br_type == "scan_range":
            bg = np.mean(d[:, start_scan:end_scan], axis=1, keepdims=True)
            d[:, start_scan:end_scan] -= bg
        self.data = d
        self.add_process("BR")

    # =========================================================
    # Gain
    # =========================================================
    def range_gain(self, gain_type="automatic", n_points=6, overall_gain_db=3.0, horiz_tc=15):
        if self.data is None:
            return
        d = self.data.astype(float)
        z = np.linspace(0, 1, d.shape[0])
        if gain_type == "linear":
            g = np.interp(z, np.linspace(0,1,n_points), np.linspace(1,overall_gain_db,n_points))
            d *= g[:,None]
        elif gain_type == "exponential":
            g = np.interp(z, np.linspace(0,1,n_points), np.logspace(0,np.log10(overall_gain_db),n_points))
            d *= g[:,None]
        self.data = d
        self.add_process(f"RG-{gain_type}")
    # =========================================================
    # Peak extraction
    # =========================================================        
    def extract_peaks(
            self,
            peak_type="all",
            max_peaks=3,
            samples_per_point=3,
            start_sample=0,
            end_sample=None
        ):
            """
            Peak extraction similar to commercial GPR software.
            """

            if self.data is None:
                return None

            data = self.data.copy()
            nrows, ncols = data.shape

            if end_sample is None or end_sample > nrows:
                end_sample = nrows

            output = np.zeros_like(data)

            from scipy.signal import find_peaks

            half_w = samples_per_point // 2

            for col in range(ncols):
                trace = data[start_sample:end_sample, col]

                if peak_type == "positive":
                    peaks, _ = find_peaks(trace)
                elif peak_type == "negative":
                    peaks, _ = find_peaks(-trace)
                else:
                    p1, _ = find_peaks(trace)
                    p2, _ = find_peaks(-trace)
                    peaks = np.unique(np.concatenate([p1, p2]))

                if peaks.size == 0:
                    continue

                # Sort by amplitude
                amps = np.abs(trace[peaks])
                idx = np.argsort(amps)[::-1][:max_peaks]
                peaks = peaks[idx]

                for p in peaks:
                    r0 = start_sample + p
                    r1 = max(0, r0 - half_w)
                    r2 = min(nrows, r0 + half_w + 1)
                    output[r1:r2, col] = data[r1:r2, col]

            return output    
    # =========================================================
    # Denoise
    # =========================================================
    def pca_gradient_wavelet_denoise(
            self,
            bscan,
            pca_keep_ratio=0.45,
            wavelet="db7",
            wavelet_level=7
        ):
            import numpy as np
            import cv2
            import pywt
            from sklearn.decomposition import PCA

            # -------------------------
            # 1. PCA reconstruction
            # -------------------------
            X = bscan.copy()
            X -= np.mean(X, axis=0)

            n_comp = int(pca_keep_ratio * min(X.shape))
            pca = PCA(n_components=n_comp, svd_solver="full")
            Xp = pca.fit_transform(X.T)
            Xrec = pca.inverse_transform(Xp).T

            # -------------------------
            # 2. Gradient magnitude
            # -------------------------
            gx = cv2.Sobel(Xrec, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(Xrec, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(gx**2 + gy**2)

            # -------------------------
            # 3. Otsu threshold
            # -------------------------
            g_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12)
            g_uint8 = (g_norm * 255).astype(np.uint8)

            _, mask = cv2.threshold(
                g_uint8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            thresholded = Xrec * (mask > 0)

            # -------------------------
            # 4. Wavelet denoising (2D)
            # -------------------------
            coeffs = pywt.wavedec2(thresholded, wavelet, level=wavelet_level)

            cH, cV, cD = coeffs[-1]
            sigma = np.median(np.abs(cD)) / 0.6745
            T = sigma * np.sqrt(2 * np.log(thresholded.size))

            new_coeffs = [coeffs[0]]
            for d in coeffs[1:]:
                new_coeffs.append(tuple(
                    pywt.threshold(x, T, mode="soft") for x in d
                ))

            out = pywt.waverec2(new_coeffs, wavelet)
            return out[:bscan.shape[0], :bscan.shape[1]]
            
    def extract_interpretable_reflectors_hough(
        self,
        canny_low=60,
        canny_high=160,
        min_length_ratio=0.15,     # fraction of section width
        max_dip_deg=25,            # reflector dip limit
        amp_percentile=70,         # amplitude threshold
        continuity_tol=6,          # px gap tolerance
        cluster_dist=12            # px for merging similar lines
    ):
        """
        Human-like reflector extraction:
        - keeps only laterally continuous reflectors
        - suppresses hyperbola flanks and clutter
        """

        if self.data is None:
            return []

        nrows, ncols = self.data.shape
        min_line_length = int(min_length_ratio * ncols)

        # ---------------------------------
        # 1. Normalize + gradient emphasis
        # ---------------------------------
        img = self.data.astype(float)
        img -= img.min()
        img /= (img.max() + 1e-12)

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        grad = (grad / grad.max() * 255).astype(np.uint8)

        # ---------------------------------
        # 2. Edge detection
        # ---------------------------------
        edges = cv2.Canny(grad, canny_low, canny_high)

        # ---------------------------------
        # 3. Probabilistic Hough
        # ---------------------------------
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=120,
            minLineLength=min_line_length,
            maxLineGap=continuity_tol
        )

        if lines is None:
            return []

        # ---------------------------------
        # 4. Geophysical filtering
        # ---------------------------------
        candidates = []

        amp_thresh = np.percentile(np.abs(self.data), amp_percentile)

        for ln in lines:
            x1, y1, x2, y2 = ln[0]

            length = np.hypot(x2 - x1, y2 - y1)
            dip = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # --- dip constraint (remove hyperbola flanks) ---
            if abs(dip) > max_dip_deg:
                continue

            # --- amplitude coherence along line ---
            num = int(length)
            xs = np.linspace(x1, x2, num).astype(int)
            ys = np.linspace(y1, y2, num).astype(int)

            valid = (
                (xs >= 0) & (xs < ncols) &
                (ys >= 0) & (ys < nrows)
            )

            xs = xs[valid]
            ys = ys[valid]

            if xs.size < 0.5 * length:
                continue

            mean_amp = np.mean(np.abs(self.data[ys, xs]))

            if mean_amp < amp_thresh:
                continue

            candidates.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "length": length,
                "dip": dip,
                "mean_amp": mean_amp
            })

        if not candidates:
            return []

        # ---------------------------------
        # 5. Merge overlapping / redundant lines
        # ---------------------------------
        merged = []

        for c in sorted(candidates, key=lambda x: -x["length"]):
            keep = True
            for m in merged:
                dy = abs((c["y1"] + c["y2"]) / 2 - (m["y1"] + m["y2"]) / 2)
                if dy < cluster_dist and abs(c["dip"] - m["dip"]) < 5:
                    keep = False
                    break
            if keep:
                merged.append(c)

        # ---------------------------------
        # 6. Output
        # ---------------------------------
        reflectors = []
        for m in merged:
            reflectors.append({
                "length": m["length"],
                "dip_angle_deg": m["dip"],
                "mean_amplitude": m["mean_amp"],
                "endpoints": (m["x1"], m["y1"], m["x2"], m["y2"])
            })

        return reflectors
    
    def plot_hough_reflectors(self, lines_info):
        if self.data is None or not lines_info:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))

        vmin, vmax = np.percentile(self.data, [2, 98])
        ax.imshow(
            self.data,
            cmap="gray",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )

        for ln in lines_info:
            x1, y1, x2, y2 = ln["endpoints"]
            ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)

        ax.set_title("Long Continuous Reflectors (Hough)")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample / Depth")
        # ax.invert_yaxis()

        plt.tight_layout()
        return fig
    def pick_layers_semi_auto(
        self,
        amp_percentile=75,
        max_vertical_jump=4,
        min_trace_coverage=0.4
    ):
        """
        Semi-automatic GPR layer picking (horizon tracking)
        """

        if self.data is None:
            return []

        data = self.data.copy()

        # ---------------------------------
        # 1. Envelope (Hilbert)
        # ---------------------------------
        from scipy.signal import hilbert
        env = np.abs(hilbert(data, axis=0))

        # ---------------------------------
        # 2. Threshold strong reflectors
        # ---------------------------------
        thresh = np.percentile(env, amp_percentile)
        mask = env > thresh

        nrows, ncols = env.shape

        visited = np.zeros_like(mask, dtype=bool)
        layers = []

        # ---------------------------------
        # 3. Track laterally (region growing)
        # ---------------------------------
        for col in range(ncols):
            for row in np.where(mask[:, col])[0]:

                if visited[row, col]:
                    continue

                layer = [(row, col)]
                visited[row, col] = True
                cur_row = row

                for c in range(col + 1, ncols):
                    search = np.arange(
                        max(0, cur_row - max_vertical_jump),
                        min(nrows, cur_row + max_vertical_jump + 1)
                    )

                    candidates = search[mask[search, c]]

                    if len(candidates) == 0:
                        break

                    cur_row = candidates[np.argmax(env[candidates, c])]
                    visited[cur_row, c] = True
                    layer.append((cur_row, c))

                if len(layer) / ncols >= min_trace_coverage:
                    layers.append(layer)

        return layers
        def plot_picked_layers(self, layers):
            fig, ax = plt.subplots(figsize=(12, 7))

            vmin, vmax = np.percentile(self.data, [2, 98])
            ax.imshow(
                self.data,
                cmap="gray",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            for lyr in layers:
                ys = [p[0] for p in lyr]
                xs = [p[1] for p in lyr]
                ax.plot(xs, ys, 'r', linewidth=1.5)

            ax.set_title("Semi Automatic Layer Picking")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            # ax.invert_yaxis()

            return fig        
        def _load_geox(self, file_path: str) -> pd.DataFrame:
            try:
                data_lines = []
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and not line.isdigit():
                        parts = line.split(',')
                        if len(parts) >= 7:
                            try:
                                values = [float(p) for p in parts[:7]]
                                time = float(parts[7]) if len(parts) > 7 else 0.0
                                data_lines.append([*values, time])
                            except ValueError:
                                continue

                return pd.DataFrame(data_lines, columns=["Marker", "X", "Y", "Z", "Lat", "Lon", "Alt", "Time"])
            except Exception as e:
                print(f"GEOX read error: {e}")
                return pd.DataFrame(columns=["Marker", "X", "Y", "Z", "Lat", "Lon", "Alt", "Time"])
        
        def _load_geox_auto(self, base_path: str, line_name: str):
            geox_path = os.path.join(base_path, f"{line_name}.geox")
            if os.path.exists(geox_path):
                self.xyz = self._load_geox(geox_path)
                if self.xyz is not None and len(self.xyz) > 0:
                    self.y_coordinate = float(self.xyz['Y'].iloc[0])
    # =========================================================
    # Predictive decon
    # =========================================================                    
    def predictive_deconvolution(
        self,
        operator_length=32,
        prediction_lag=8,
        prewhitening=0.08,
        overall_gain=4.0,
        start_sample=0,
        end_sample=None,
        progress_callback=None  # ← NEW: for progress bar
    ):
        """
        RADAN-style Predictive Deconvolution
        Removes antenna ringing and multiples
        """
        if self.data is None:
            return

        data = self.data.astype(float)
        n_samples, n_traces = data.shape

        if end_sample is None or end_sample > n_samples:
            end_sample = n_samples

        out = data.copy()

        for itr in range(n_traces):
            trace = data[start_sample:end_sample, itr].copy()

            # Autocorrelation
            autocorr = np.correlate(trace, trace, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr[0] += autocorr[0] * prewhitening
            autocorr /= autocorr[0] + 1e-12

            # Toeplitz matrix
            n = operator_length
            lag = prediction_lag
            R = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    R[i, j] = autocorr[abs(i - j) + lag]

            g = autocorr[lag:lag + n]

            # Solve for filter
            try:
                f = np.linalg.solve(R, g)
            except np.linalg.LinAlgError:
                f = np.zeros(n)

            # Apply
            predicted = np.convolve(trace, f, mode='full')[:len(trace)]
            residual = trace - predicted
            residual *= overall_gain

            out[start_sample:end_sample, itr] = residual

            # ← REPORT PROGRESS
            if progress_callback is not None:
                progress_callback((itr + 1) / n_traces)

        self.data = out
        self.add_process("DECON")     
    def clear_data(self):
        for attr in ['data','xyz','hdr_info','depth','base_path','line_name',
                     'data_type','file_path','file_size','header_size',
                     'samples_per_trace','num_traces','shape','dtype',
                     '_sample_interval_s','_velocity','_x_cell_m',
                     '_num_traces_hdr','_samples_per_trace_hdr']:
            setattr(self, attr, None)
                    
                    


        
 


# ===================================================================================================================================================================================
#                             GUI Application
# ====================================================================================================================================================================================

class V00ReaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPR .V00 File Viewer | Mrinal: layek.mk@gmail.com")
        self.root.geometry("1200x900")
        
        self.loader = GPR2DLoader()
        
        # Main frames
        self.input_frame = tk.Frame(root, padx=10, pady=10)
        self.input_frame.pack(fill=tk.X)
        
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.meta_frame = tk.Frame(root, padx=10, pady=5)
        self.meta_frame.pack(fill=tk.X)
        
        
        # ================= ROW 0 : FILE & CORE =================
        tk.Label(self.input_frame, text="V00 File:", font=("Helvetica", 11)).grid(row=0, column=0, padx=(0,5), pady=5, sticky="e")

        self.file_var = tk.StringVar()
        tk.Entry(self.input_frame, textvariable=self.file_var,width=55, state="readonly").grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        tk.Button(self.input_frame, text="Browse V00", command=self.browse_file, bg="#E8F5E9").grid(row=0, column=4, padx=4)
        tk.Button(self.input_frame, text="Browse DT", command=self.browse_dt_file, bg="#E8F5E9").grid(row=0, column=5, padx=4)
        tk.Button(self.input_frame, text="Load Data", command=self.load_selected_file, bg="#4CAF50", fg="white").grid(row=0, column=6, padx=6)

        tk.Button(self.input_frame, text="FFT / DFT", command=self.fft_dialog, bg="#FFFFFF").grid(row=0, column=7, padx=4)
        tk.Button(self.input_frame, text="HHT", command=self.hht_dialog, bg="#FFFFFF").grid(row=0, column=8, padx=4)
        tk.Button(self.input_frame, text="HHT (T–F)", command=self.hht_tf_dialog, bg="#FFFFFF").grid(row=0, column=9, padx=4)

        tk.Button(self.input_frame, text="ICA MF Denoise", command=self.ica_denoise_dialog, bg="#E1F5FE").grid(row=0, column=10, padx=4)
        tk.Button(self.input_frame, text="Peaks Extraction", command=self.peaks_extraction_popup, bg="#E8F5E9").grid(row=0, column=11, padx=4)

        # ================= ROW 1 : DISPLAY & VISUALIZATION =================
        tk.Label(self.input_frame, text="Colormap:", font=("Helvetica", 11)).grid(row=1, column=0, padx=(0,5), pady=5, sticky="e")

        self.cmap_var = tk.StringVar(value="gray")
        ttk.Combobox( self.input_frame, textvariable=self.cmap_var,values=["gray","seismic","wiggle","RdBu_r","coolwarm","viridis","magma","plasma","hot"],state="readonly", width=14).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        tk.Button(self.input_frame, text="Apply", command=self.change_colormap, bg="#F44336", fg="white").grid(row=1, column=2, padx=4)
        tk.Button(self.input_frame, text="Zoom In", command=lambda: self.zoom(0.8), bg="#E3F2FD").grid(row=1, column=3, padx=4)
        tk.Button(self.input_frame, text="Zoom Out", command=lambda: self.zoom(1.25), bg="#E3F2FD").grid(row=1, column=4, padx=4)
        tk.Button(self.input_frame, text="Save Figure", command=self.save_figure, bg="#2196F3", fg="white").grid(row=1, column=5, padx=4)

        tk.Button(self.input_frame, text="A-scan", command=self.toggle_ascan, bg="#9C27B0", fg="white").grid(row=1, column=6, padx=4)
        tk.Button(self.input_frame, text="Hilbert", command=self.hilbert_dialog, bg="#FFFFFF").grid(row=1, column=7, padx=4)
        tk.Button(self.input_frame, text="Show ICA", command=self.show_ica_components, bg="#FFFDE7").grid(row=1, column=8, padx=4)

        tk.Button(self.input_frame, text="Plot GPS (NMEA)", command=self.plot_gps_nmea, bg="#E3F2FD").grid(row=1, column=9, padx=4)
        tk.Button(self.input_frame, text="Plot GEOX Path", command=self.plot_geox_path, bg="#FFE0B2").grid(row=1, column=10, padx=4)
        tk.Button(self.input_frame, text="Plot GEOX on GPS", command=self.plot_geox_on_gps, bg="#B3E5FC").grid(row=1, column=11, padx=4)

        # ================= ROW 2 : ANALYSIS & INTERPRETATION =================
        tk.Button(self.input_frame, text="Statistics", command=self.show_stats, bg="#F5F5F5").grid(row=2, column=0, padx=4, pady=6)
        tk.Button(self.input_frame, text="Plot XYZ", command=self.plot_geox, bg="#F5F5F5").grid(row=2, column=1, padx=4, pady=6)
        tk.Button(self.input_frame, text="Survey Map", command=self.plot_survey_map, bg="#3F51B5", fg="white").grid(row=2, column=2, padx=4, pady=6)
        tk.Button(self.input_frame, text="Load GEOX / GEC", command=self.load_geo_file, bg="#E8F5E9").grid(row=2, column=3, padx=4, pady=6)

        tk.Button(self.input_frame, text="FIR Lowpass", command=self.apply_fir_low, bg="#FFFFFF").grid(row=2, column=4, padx=4, pady=6)
        tk.Button(self.input_frame, text="FIR Bandpass", command=self.fir_bandpass_dialog, bg="#FFFFFF").grid(row=2, column=5, padx=4, pady=6)
        tk.Button(self.input_frame, text="Background Removal", command=self.background_removal_dialog, bg="#FFF3E0").grid(row=2, column=6, padx=4, pady=6)
        tk.Button(self.input_frame, text="Range Gain", command=self.range_gain_dialog, bg="#E3F2FD").grid(row=2, column=7, padx=4, pady=6)

        tk.Button(self.input_frame, text="Hough Reflectors", command=self.run_hough_reflectors, bg="#E0F2F1").grid(row=2, column=8, padx=4, pady=6)
        tk.Button(self.input_frame, text="Pick Layers", command=self.layer_picker_popup, bg="#FFF3E0").grid(row=2, column=9, padx=4, pady=6)
        tk.Button(self.input_frame, text="Amplitude Map", command=self.show_amplitude_map, bg="#E3F2FD").grid(row=2, column=10, padx=4, pady=6)

        tk.Button(self.input_frame, text="Reset Data", command=self.reset_data, bg="#FFB74D").grid(row=2, column=11, padx=4, pady=6)
        tk.Button(self.input_frame, text="Clear All", command=self.clear_all, bg="#D32F2F", fg="white").grid(row=2, column=12, padx=4, pady=6)
        tk.Button(self.input_frame, text="Deconvolution", command=self.deconvolution_dialog, bg="#D32F2F", fg="white").grid(row=2, column=14, padx=4, pady=6)
   

        
        # # Metadata display
        tk.Label(self.meta_frame, text="Metadata & Information:", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.meta_text = scrolledtext.ScrolledText(self.meta_frame, height=10, width=100, font=("Consolas", 10))
        self.meta_text.pack(fill=tk.X, pady=5)
        
        # Plot area
        self.figure = plt.Figure(figsize=(12, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        # ---------------- A-sCAN WINDOW ----------------
        self.ascan_win = tk.Toplevel(self.root)
        self.ascan_win.title("A-scan")
        self.ascan_win.geometry("350x600")
        self.ascan_win.withdraw()  # hidden until data loaded

        self.ascan_fig = plt.Figure(figsize=(3, 5), dpi=100)
        self.ascan_ax = self.ascan_fig.add_subplot(111)

        self.ascan_canvas = FigureCanvasTkAgg(self.ascan_fig, master=self.ascan_win)
        self.ascan_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.ascan_enabled = False
        self.ascan_cid = None
        # ---------------- A-scan window ----------------
        self.ascan_win = tk.Toplevel(self.root)
        self.ascan_win.title("A-scan")
        self.ascan_win.geometry("350x600")
        self.ascan_win.withdraw()  # hidden initially

        self.ascan_fig = plt.Figure(figsize=(3, 5), dpi=100)
        self.ascan_ax = self.ascan_fig.add_subplot(111)

        self.ascan_canvas = FigureCanvasTkAgg(self.ascan_fig, master=self.ascan_win)
        self.ascan_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



        # For colormap update
        self.current_image = None
    
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select .V00 File",
            filetypes=[("V00 files", "*.V00 *.v00"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)
    def browse_dt_file(self):
        path = filedialog.askopenfilename(
            title="Select .DT File",
            filetypes=[("DT files", "*.DT *.dt"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)
            
    def load_selected_file(self):
        path = self.file_var.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid file first.")
            return

        try:
            base = os.path.dirname(path)
            name, ext = os.path.splitext(os.path.basename(path))
            ext = ext.lower()

            if ext == ".v00":
                data_type = "v00"
            elif ext == ".dt":
                data_type = "dt"
            else:
                messagebox.showerror(
                    "Unsupported",
                    "Only .V00 and .DT files are supported."
                )
                return
            self.loader.load_data(base, name, data_type)

            # Save original (unfiltered) data safely
            self.original_data = self.loader.data.copy()

            self.show_metadata()
            self.plot_gpr()
                         

            # self.loader.load_data(base, name, data_type)

            # self.show_metadata()
            # self.plot_gpr()

        except Exception as e:
            messagebox.showerror("Load Failed", str(e))

    def fir_filter(self, data, dt, f1=None, f2=None, numtaps=101, ftype="bandpass"):
        """
        FIR filtering using windowed-sinc method.
        dt: sample interval (seconds)
        f1, f2: cutoff frequencies (Hz)
        """
        fs = 1.0 / dt
        nyq = fs / 2.0

        if ftype == "lowpass":
            fc = f2 / nyq
            h = sinc(2 * fc * (np.arange(numtaps) - (numtaps - 1) / 2))
        elif ftype == "highpass":
            fc = f1 / nyq
            h = sinc(np.arange(numtaps) - (numtaps - 1) / 2) - \
                sinc(2 * fc * (np.arange(numtaps) - (numtaps - 1) / 2))
        else:  # bandpass
            fc1 = f1 / nyq
            fc2 = f2 / nyq
            h = (
                sinc(2 * fc2 * (np.arange(numtaps) - (numtaps - 1) / 2)) -
                sinc(2 * fc1 * (np.arange(numtaps) - (numtaps - 1) / 2))
            )

        window = np.hamming(numtaps)
        h *= window
        h /= np.sum(h)

        # Filter trace by trace
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = np.convolve(data[:, i], h, mode="same")

        return filtered
        
    def apply_fir_low(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        self.loader.data = self.fir_filter(
            self.loader.data,
            dt=self.loader._sample_interval_s,
            f2=200e6,     # example cutoff (200 MHz)
            numtaps=121,
            ftype="lowpass"
        )
        self.plot_gpr()
        self.process_history.append("FIR-low")


    def apply_fir_band(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        self.loader.data = self.fir_filter(
            self.loader.data,
            dt=self.loader._sample_interval_s,
            f1=50e6,      # example band
            f2=250e6,
            numtaps=151,
            ftype="bandpass"
        )
        self.plot_gpr()
        self.process_history.append("FIR-BP")

    def fir_bandpass_dialog(self):
        if self.loader.data is None or self.loader._sample_interval_s is None:
            messagebox.showerror("Error", "Load data with valid HDR first.")
            return

        win = tk.Toplevel(self.root)
        win.title("FIR Bandpass Filter")
        win.geometry("330x250")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()   # modal dialog

        tk.Label(win, text="Low cut frequency (MHz)").pack(pady=(10, 0))
        f1_mhz = tk.DoubleVar(value=50.0)
        tk.Entry(win, textvariable=f1_mhz, width=15).pack()

        tk.Label(win, text="High cut frequency (MHz)").pack(pady=(10, 0))
        f2_mhz = tk.DoubleVar(value=250.0)
        tk.Entry(win, textvariable=f2_mhz, width=15).pack()

        tk.Label(win, text="Number of taps").pack(pady=(10, 0))
        ntap_var = tk.IntVar(value=151)
        tk.Entry(win, textvariable=ntap_var, width=15).pack()

        # ---------------- Apply logic ----------------
        def apply_and_close():
            try:
                f1 = f1_mhz.get() * 1e6   # MHz → Hz
                f2 = f2_mhz.get() * 1e6
                ntaps = ntap_var.get()

                if f1 <= 0 or f2 <= 0 or f2 <= f1:
                    raise ValueError("Require: 0 < f_low < f_high (MHz)")

                self.loader.data = self.fir_filter(
                    self.loader.data,
                    dt=self.loader._sample_interval_s,
                    f1=f1,
                    f2=f2,
                    numtaps=ntaps,
                    ftype="bandpass"
                )

                win.destroy()
                self.plot_gpr()

            except Exception as e:
                messagebox.showerror("Invalid input", str(e))

        # ---------------- Buttons ----------------
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame, text="OK",
            width=10, bg="#4CAF50", fg="white",
            command=apply_and_close
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            btn_frame, text="Cancel",
            width=10,
            command=win.destroy
        ).grid(row=0, column=1, padx=5)


    def on_mouse_move(self, event):
        if (
            self.loader.data is None or
            event.inaxes is None or
            event.xdata is None
        ):
            return

        data = self.loader.data
        n_traces = data.shape[1]

        # X-axis values
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            xvals = self.loader.xyz['X'].values
        else:
            xvals = np.arange(n_traces)

        # Find nearest trace index
        idx = np.argmin(np.abs(xvals - event.xdata))
        idx = np.clip(idx, 0, n_traces - 1)

        trace = data[:, idx]

        # Depth / time axis
        y = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(len(trace))
        )

        self.update_ascan(trace, y, idx)

    def update_ascan(self, trace, y, idx):
        self.ascan_ax.cla()

        self.ascan_ax.plot(trace, y, color="black", linewidth=1.0)
        self.ascan_ax.axvline(0, color="gray", linewidth=0.5)

        self.ascan_ax.set_ylim(y[-1], y[0])  # depth down
        self.ascan_ax.set_xlabel("Amplitude")
        self.ascan_ax.set_ylabel("Depth (m)")
        self.ascan_ax.set_title(f"A-scan | Trace {idx}")

        self.ascan_ax.grid(True, alpha=0.3)

        self.ascan_fig.tight_layout()
        self.ascan_canvas.draw_idle()

        if not self.ascan_win.winfo_viewable():
            self.ascan_win.deiconify()
    def toggle_ascan(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        if not self.ascan_enabled:
            # Enable A-scan
            self.ascan_cid = self.canvas.mpl_connect(
                "motion_notify_event", self.on_mouse_move
            )
            self.ascan_enabled = True
            self.ascan_win.deiconify()
        else:
            # Disable A-scan
            if self.ascan_cid is not None:
                self.canvas.mpl_disconnect(self.ascan_cid)
                self.ascan_cid = None

            self.ascan_enabled = False
            self.ascan_win.withdraw()
    def on_mouse_move(self, event):
        if not self.ascan_enabled:
            return

        if (
            self.loader.data is None or
            event.inaxes is None or
            event.xdata is None
        ):
            return

        data = self.loader.data
        n_traces = data.shape[1]

        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            xvals = self.loader.xyz['X'].values
        else:
            xvals = np.arange(n_traces)

        idx = np.argmin(np.abs(xvals - event.xdata))
        idx = np.clip(idx, 0, n_traces - 1)

        trace = data[:, idx]

        y = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(len(trace))
        )

        self.update_ascan(trace, y, idx)
        self.last_trace_index = idx

    def hilbert_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Hilbert Transform")
        win.resizable(False, False)

        tk.Label(win, text="Hilbert Output:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=10, sticky="e")

        mode_var = tk.StringVar(value="envelope")
        modes = [("Envelope", "envelope"), ("Phase", "phase"), ("Frequency", "frequency")]

        for i, (txt, val) in enumerate(modes):
            tk.Radiobutton(win, text=txt, variable=mode_var, value=val).grid(row=i, column=1, padx=5, pady=5, sticky="w")

        def apply_hilbert():
            self.apply_hilbert(mode_var.get())
            win.destroy()

        tk.Button(win, text="OK", command=apply_hilbert, bg="#4CAF50", fg="white", width=10).grid(row=4, column=0, columnspan=2, pady=12)
    def apply_hilbert(self, mode="envelope"):
        data = self.loader.data
        analytic = hilbert(data, axis=0)

        if mode == "envelope":
            self.loader.data = np.abs(analytic)

        elif mode == "phase":
            self.loader.data = np.unwrap(np.angle(analytic), axis=0)

        elif mode == "frequency":
            phase = np.unwrap(np.angle(analytic), axis=0)
            dt = self.loader.dt if hasattr(self.loader, "dt") and self.loader.dt is not None else 1.0
            self.loader.data = np.diff(phase, axis=0) / (2 * np.pi * dt)

        self.plot_gpr()
    def fft_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Fourier Transform (FFT)")
        win.resizable(False, False)

        tk.Label(win, text="FFT Mode:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="e")

        mode_var = tk.StringVar(value="ascan")
        tk.Radiobutton(win, text="Current A-scan (cursor)", variable=mode_var, value="ascan").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(win, text="Trace Average", variable=mode_var, value="average").grid(row=1, column=1, sticky="w")

        tk.Label(win, text="Spectrum:", font=("Helvetica", 11)).grid(row=2, column=0, padx=10, pady=8, sticky="e")

        spec_var = tk.StringVar(value="amplitude")
        tk.Radiobutton(win, text="Amplitude", variable=spec_var, value="amplitude").grid(row=2, column=1, sticky="w")
        tk.Radiobutton(win, text="Power", variable=spec_var, value="power").grid(row=3, column=1, sticky="w")

        def apply_fft():
            self.compute_fft(mode_var.get(), spec_var.get())
            win.destroy()

        tk.Button(win, text="OK", command=apply_fft, bg="#4CAF50", fg="white", width=10).grid(row=4, column=0, columnspan=2, pady=12)
    def compute_fft(self, mode="ascan", spectrum="amplitude"):
        data = self.loader.data

        if mode == "ascan":
            if not hasattr(self, "last_trace_index"):
                messagebox.showinfo("Info", "Move mouse over section to select A-scan.")
                return
            trace = data[:, self.last_trace_index]
        else:
            trace = np.mean(data, axis=1)

        n = len(trace)
        dt = self.loader.dt if hasattr(self.loader, "dt") and self.loader.dt is not None else 1.0

        freq = np.fft.rfftfreq(n, d=dt)
        # freq = np.fft.rfftfreq(n, d=dt) * 1e-6

        fftv = np.fft.rfft(trace)

        if spectrum == "power":
            spec = np.abs(fftv) ** 2
            ylabel = "Power"
        else:
            spec = np.abs(fftv)
            ylabel = "Amplitude"

        self.plot_fft(freq, spec, ylabel)
        
    def plot_fft(self, freq, spec, ylabel):
        win = tk.Toplevel(self.root)
        win.title("FFT Spectrum")

        fig = plt.Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(freq, spec, color="black", linewidth=1.2)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title("Frequency Spectrum")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw_idle()
    def hht_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Hilbert-Huang Transform (HHT)")
        win.resizable(False, False)

        tk.Label(win, text="Trace index:", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="e")
        trace_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=trace_var, width=10).grid(row=0, column=1, pady=8)

        tk.Label(win, text="IMF number:", font=("Helvetica", 11)).grid(row=1, column=0, padx=10, pady=8, sticky="e")
        imf_var = tk.IntVar(value=1)
        tk.Entry(win, textvariable=imf_var, width=10).grid(row=1, column=1, pady=8)

        def apply_hht():
            self.apply_hht(trace_var.get(), imf_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=apply_hht,
            bg="#4CAF50",
            fg="white",
            width=10
        ).grid(row=2, column=0, columnspan=2, pady=12)
      
    def apply_hht(self, trace_idx=0, imf_idx=1):
        data = self.loader.data

        if trace_idx < 0 or trace_idx >= data.shape[1]:
            messagebox.showerror("Error", "Invalid trace index.")
            return

        trace = data[:, trace_idx]

        emd = EMD(trace)
        imfs = emd.decompose()

        if imf_idx < 1 or imf_idx > imfs.shape[0]:
            messagebox.showerror("Error", "Invalid IMF number.")
            return

        imf = imfs[imf_idx - 1]

        analytic = hilbert(imf)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))

        # Time axis
        y = self.loader.depth if self.loader.depth is not None else np.arange(len(imf))

        self.figure.clf()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        ax1.plot(amplitude, y, color="black")
        ax1.set_title(f"HHT Envelope | Trace {trace_idx} | IMF {imf_idx}")
        ax1.set_ylabel("Depth (m)")
        # ax1.invert_yaxis()

        ax2.plot(phase, y, color="red")
        ax2.set_title("Instantaneous Phase")
        ax2.set_xlabel("Amplitude / Phase")
        ax2.set_ylabel("Depth (m)")
        # ax2.invert_yaxis()

        self.figure.tight_layout()
        self.canvas.draw_idle()
    def hht_tf_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("HHT Time–Frequency (H(t,f))")
        win.resizable(False, False)

        tk.Label(win, text="Trace index:", font=("Helvetica", 11)).grid(
            row=0, column=0, padx=10, pady=8, sticky="e"
        )
        trace_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=trace_var, width=10).grid(row=0, column=1, pady=8)

        def apply():
            self.plot_hht_tf(trace_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=apply,
            bg="#4CAF50",
            fg="white",
            width=10
        ).grid(row=1, column=0, columnspan=2, pady=12)
    def plot_hht_tf(self, trace_idx=0):
        data = self.loader.data

        if trace_idx < 0 or trace_idx >= data.shape[1]:
            messagebox.showerror("Error", "Invalid trace index.")
            return

        trace = data[:, trace_idx]

        # --- EMD decomposition ---
        emd = EMD(trace)
        imfs = emd.decompose()

        # --- Sampling interval ---
        dt = self.loader._sample_interval_s
        if dt is None:
            dt = 1.0  # fallback (relative units)

        n = len(trace)

        # Depth or time axis
        if self.loader.depth is not None:
            y = self.loader.depth
            ylabel = "Depth (m)"
        else:
            y = np.arange(n) * dt
            ylabel = "Time (s)"

        # --- Build HHT spectrum ---
        freq_bins = np.linspace(0, 1.0 / (2 * dt), 300)   # Hz
        freq_bins_mhz = freq_bins * 1e-6                  # MHz

        H = np.zeros((n, len(freq_bins)))

        for imf in imfs:
            analytic = hilbert(imf)
            amp = np.abs(analytic)

            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) / (2 * np.pi * dt)
            inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  # align length

            for i in range(n):
                f = inst_freq[i]
                if 0 < f < freq_bins[-1]:
                    k = np.searchsorted(freq_bins, f)
                    H[i, k] += amp[i]

        # --- Plot ---
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        pcm = ax.pcolormesh(
            freq_bins_mhz,
            y,
            H,
            shading="auto",
            cmap="jet"
        )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"HHT Time–Frequency | Trace {trace_idx}")
        # ax.invert_yaxis()

        self.figure.colorbar(pcm, ax=ax, label="Amplitude")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        
    def plot_survey_map(self):
        if (
            self.loader is None or
            not hasattr(self.loader, "lat") or
            not hasattr(self.loader, "lon") or
            self.loader.lat is None or
            self.loader.lon is None
        ):
            messagebox.showerror(
                "Error",
                "No geographic coordinates found.\n"
                "Load GEO / GEOX data first."
            )
            return

        lat = self.loader.lat
        lon = self.loader.lon

        if len(lat) < 2:
            messagebox.showerror("Error", "Not enough points to plot survey line.")
            return

        import folium
        import tempfile
        import webbrowser
        import os

        # Center map
        center_lat = lat.mean()
        center_lon = lon.mean()

        # Create map (ESRI World Imagery)
        m = folium.Map(
        location=[self.loader.lat[0], self.loader.lon[0]],
        zoom_start=18,
        tiles="Esri.WorldImagery"
        )



        # Survey line
        coords = list(zip(lat, lon))
        folium.PolyLine(
            coords,
            color="yellow",
            weight=4,
            opacity=0.9,
            tooltip="GPR Survey Line"
        ).add_to(m)

        # Start and end markers
        folium.Marker(
            coords[0],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

        folium.Marker(
            coords[-1],
            popup="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

        # Save temporary HTML
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        m.save(tmpfile.name)

        # Open in browser (acts like popup)
        webbrowser.open("file://" + os.path.realpath(tmpfile.name))
        
    def projected_to_latlon(self, x, y, epsg):
        """
        Convert projected coordinates to latitude longitude.
        epsg: int (e.g., 32645, 5179, 4326)
        """
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(x, y)
        return np.asarray(lat), np.asarray(lon)
        
    def projected_to_latlon(self, x, y, epsg):
        """
        Convert projected coordinates to latitude longitude.
        epsg: int (e.g., 32645, 5179, 4326)
        """
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(x, y)
        return np.asarray(lat), np.asarray(lon)
    def load_geo(self, filename):
        data = np.loadtxt(filename)

        self.lat = data[:, 1]
        self.lon = data[:, 2]

        self.coord_type = "geographic"
    def load_geox(self, filename, epsg):
        data = np.loadtxt(filename)

        x = data[:, 1]
        y = data[:, 2]

        self.lat, self.lon = self.projected_to_latlon(x, y, epsg)
        self.coord_type = "projected"


    def ask_epsg(self):
        win = tk.Toplevel(self.root)
        win.title("Select EPSG Code")
        win.resizable(False, False)

        tk.Label(win, text="EPSG Code:").grid(row=0, column=0, padx=10, pady=5)

        epsg_var = tk.StringVar(value="32645")

        tk.Entry(win, textvariable=epsg_var, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        result = {"epsg": None}

        def ok():
            result["epsg"] = int(epsg_var.get())
            win.destroy()

        tk.Button(
            win,
            text="OK",
            command=ok,
            bg="#4CAF50",
            fg="white"
        ).grid(row=1, column=0, columnspan=2, pady=10)

        win.grab_set()
        win.wait_window()

        return result["epsg"]


    def load_geo_file(self):
        path = filedialog.askopenfilename(
            title="Select GEOX or GEC file",
            filetypes=[
                ("GEOX files", "*.geox *.GEOX"),
                ("GEC files", "*.gec *.GEC"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".gec":
                self.loader.load_gec(path)

            elif ext == ".geox":
                messagebox.showwarning(
                    "GEOX ignored",
                    "GEOX coordinates are often local.\n"
                    "Use GEC (GPS) for correct map location."
                )
                epsg = self.ask_epsg()
                if epsg is None:
                    return
                self.loader.load_geox(path, epsg)


            else:
                messagebox.showerror("Error", "Unsupported geo file format.")
                return

            messagebox.showinfo("Success", "Geographic data loaded.")
            self.plot_survey_map()

        except Exception as e:
            messagebox.showerror("Geo load failed", str(e))
    def plot_gps_nmea(self):
        path = filedialog.askopenfilename(
            title="Select GPS NMEA file",
            filetypes=[
                ("GPS files", "*.gps *.GPS *.txt"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.loader.load_gps_nmea(path)
            self._plot_gps_contextily()

        except Exception as e:
            messagebox.showerror("GPS error", str(e))
    def _plot_gps_contextily(self):
        lat = self.loader.lat
        lon = self.loader.lon

        if lat is None or lon is None or len(lat) == 0:
            raise ValueError("No GPS data available to plot.")

        # ----------------------------------
        # Create GeoDataFrame (WGS84)
        # ----------------------------------
        geometry = [Point(xy) for xy in zip(lon, lat)]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

        # Convert to Web Mercator for basemap
        gdf_3857 = gdf.to_crs(epsg=3857)

        # ----------------------------------
        # Plot
        # ----------------------------------
        fig, ax = plt.subplots(figsize=(14, 9))

        gdf_3857.plot(
            ax=ax,
            linewidth=1.0,
            alpha=0.9,
            color="blue",
            label="Survey Path"
        )

        # Start / End points
        gdf_3857.iloc[[0]].plot(ax=ax, color="green", markersize=30, label="Start")
        gdf_3857.iloc[[-1]].plot(ax=ax, color="red", markersize=30, label="End")

        # ----------------------------------
        # Compute extent with margin
        # ----------------------------------
        minx, miny, maxx, maxy = gdf_3857.total_bounds
        dx = maxx - minx
        dy = maxy - miny
        margin = 0.3

        ax.set_xlim(minx - dx * margin, maxx + dx * margin)
        ax.set_ylim(miny - dy * margin, maxy + dy * margin)

        # ----------------------------------
        # Basemap (ESRI)
        # ----------------------------------
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            attribution=False
        )

        ax.set_aspect("equal")
        ax.set_title("GPR Survey Path (GPS)")
        ax.legend()
        ax.axis("off")

        # ----------------------------------
        # Enable scroll zoom
        # ----------------------------------
        self._enable_scroll_zoom(fig, ax)

        plt.show()
    def _enable_scroll_zoom(self, fig, ax):
        base_scale = 1.2

        def zoom(event):
            if event.inaxes != ax:
                return

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is None or ydata is None:
                return

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([
                xdata - new_width * (1 - relx),
                xdata + new_width * relx
            ])
            ax.set_ylim([
                ydata - new_height * (1 - rely),
                ydata + new_height * rely
            ])

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", zoom)

    def plot_geox_path(self):
        if not hasattr(self.loader, "geo_x") or not hasattr(self.loader, "geo_y"):
            messagebox.showerror("Error", "No GEOX data loaded.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            self.loader.geo_x,
            self.loader.geo_y,
            "-o",
            color="blue",
            markersize=3,
            linewidth=1,
            label="GEOX Path"
        )

        ax.scatter(
            self.loader.geo_x[0],
            self.loader.geo_y[0],
            color="green",
            s=50,
            label="Start"
        )
        ax.scatter(
            self.loader.geo_x[-1],
            self.loader.geo_y[-1],
            color="red",
            s=50,
            label="End"
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X (meters, relative)")
        ax.set_ylabel("Y (meters, relative)")
        ax.set_title("GEOX Survey Path (relative coordinates)")
        ax.legend()
        plt.show()
    def plot_geox_on_gps(self):
        try:
            lat_geo, lon_geo = self.loader.geox_to_wgs84()

            fig, ax = plt.subplots(figsize=(10, 8))
            # GEOX path
            ax.plot(lon_geo, lat_geo, "-o", color="blue", markersize=3, label="GEOX Path")
            # GPS start/end
            ax.scatter(self.loader.lon[0], self.loader.lat[0], color="green", s=80, label="GPS Start")
            ax.scatter(self.loader.lon[-1], self.loader.lat[-1], color="red", s=80, label="GPS End")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("GEOX path overlay on GPS map")
            ax.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Cannot plot GEOX: {e}")


    def preprocess_for_ica(self, data, agc_window=None):
        """
        VERY conservative preprocessing for ICA.
        No trace-wise AGC. No Z-score.
        Only weak global normalization.
        """

        proc = data.astype(float).copy()

        # --- remove DC bias per trace (safe) ---
        proc -= np.mean(proc, axis=0, keepdims=True)

        # --- global RMS normalization (very weak, reversible) ---
        rms = np.sqrt(np.mean(proc**2))
        if rms > 0:
            proc /= rms

        return proc, rms


    def ica_multifractal_denoise(
            self,
            block_size=8,
            reject_ratio=0.08,
            agc_window=None,
            progress_var=None,
            progress_win=None
    ):
        raw = self.loader.data
        if raw is None:
            return

        # --- conservative preprocessing ---
        data, rms = self.preprocess_for_ica(raw)

        n_samples, n_traces = data.shape
        output = np.zeros_like(data)

        from sklearn.decomposition import FastICA

        for i in range(0, n_traces, block_size):
            j = min(i + block_size, n_traces)
            block = data[:, i:j].T

            if block.shape[0] < 2:
                output[:, i:j] = block.T
                continue

            ica = FastICA(
                whiten="unit-variance",
                max_iter=2000,
                tol=1e-3,
                random_state=0
            )

            try:
                S = ica.fit_transform(block)
                A = ica.mixing_
            except Exception:
                output[:, i:j] = block.T
                continue

            if A is None:
                output[:, i:j] = block.T
                continue

            # --- multifractal discrimination ---
            widths = np.array([
                self.multifractal_spectrum_width(S[:, k])
                for k in range(S.shape[1])
            ])

            # reject only extreme outliers
            threshold = np.percentile(widths, 100 * (1 - reject_ratio))
            reject_idx = np.where(widths > threshold)[0]

            S[:, reject_idx] = 0.0
            recon = S @ A.T

            output[:, i:j] = recon.T

            # --- progress ---
            if progress_var is not None:
                progress_var.set(100.0 * j / n_traces)
                if progress_win is not None:
                    progress_win.update_idletasks()

        # --- restore physical scale ---
        output *= rms

        self.loader.data = output

        if progress_var is not None:
            progress_var.set(100.0)
            if progress_win is not None:
                progress_win.update_idletasks()
        self.loader.data = self.loader.pca_gradient_wavelet_denoise(
            self.loader.data
        )
        self.plot_gpr()

    @staticmethod
    def multifractal_spectrum_width(signal):
        signal = np.asarray(signal)
        signal = signal[np.isfinite(signal)]

        if signal.size < 256:
            return 0.0

        scales = [4, 8, 16, 32]
        q = [-1, 1]

        eps = 1e-12
        slopes = []

        for qq in q:
            vals = []
            used = []
            for s in scales:
                if s >= signal.size:
                    continue
                diff = np.abs(signal[s:] - signal[:-s]) + eps
                vals.append(np.mean(diff ** qq))
                used.append(s)

            if len(vals) > 1:
                slopes.append(
                    np.polyfit(np.log(used), np.log(vals), 1)[0]
                )

        if len(slopes) < 2:
            return 0.0

        return abs(slopes[1] - slopes[0])

    def ica_denoise_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("ICA Multifractal Denoising (Conservative)")
        win.resizable(False, False)

        # ---------------- variables ----------------
        block_var = tk.IntVar(value=8)     # SAFE default
        perc_var = tk.IntVar(value=8)      # SAFE default (%)
        progress_var = tk.DoubleVar(value=0.0)

        row = 0

        # ---------------- block size ----------------
        tk.Label(win, text="Block size (traces)").grid(
            row=row, column=0, padx=10, pady=6, sticky="w"
        )
        tk.Spinbox(
            win, from_=4, to=20, increment=1,
            textvariable=block_var, width=8
        ).grid(row=row, column=1, padx=5, pady=6)
        row += 1

        # ---------------- reject ratio ----------------
        tk.Label(win, text="Reject ratio (%)").grid(
            row=row, column=0, padx=10, pady=6, sticky="w"
        )
        tk.Spinbox(
            win, from_=2, to=20, increment=1,
            textvariable=perc_var, width=8
        ).grid(row=row, column=1, padx=5, pady=6)
        row += 1

        # ---------------- progress bar ----------------
        ttk.Label(win, text="Progress").grid(
            row=row, column=0, padx=10, pady=(10, 2), sticky="w"
        )
        ttk.Progressbar(
            win,
            variable=progress_var,
            maximum=100,
            length=180,
            mode="determinate"
        ).grid(row=row, column=1, padx=10, pady=(10, 2))
        row += 1

        # ---------------- buttons ----------------
        tk.Button(
            win,
            text="Apply",
            command=lambda: self._apply_ica_denoise(
                block_var.get(),
                perc_var.get(),
                progress_var,
                win
            )
        ).grid(row=row, column=0, padx=10, pady=12)

        tk.Button(
            win,
            text="Cancel",
            command=win.destroy
        ).grid(row=row, column=1, padx=10, pady=12)

        win.grab_set()
    def _apply_ica_denoise(self, block, perc, progress_var, win):
        self.ica_multifractal_denoise(
            block_size=block,
            reject_ratio=perc / 100.0,
            progress_var=progress_var,
            progress_win=win
        )
        
    def run_hough_reflectors(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            # --- Extract long reflectors ---
            lines = self.loader.extract_interpretable_reflectors_hough()


            if not lines:
                messagebox.showinfo(
                    "Result",
                    "No continuous reflectors detected."
                )
                return

            # --- Plot ---
            fig = self.loader.plot_hough_reflectors(lines)

            # Display in main canvas
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(self.loader.data, [2, 98])
            ax.imshow(
                self.loader.data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            for ln in lines:
                x1, y1, x2, y2 = ln["endpoints"]
                ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)

            ax.set_title("Long Continuous Reflectors (Hough Transform)")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

            # --- Optional: print summary ---
            print(f"Hough reflectors detected: {len(lines)}")

        except Exception as e:
            messagebox.showerror("Hough Error", str(e))
        
    # self.meta_label.config(text=self.meta_text)

    # def preprocess_for_ica(self, data, agc_window=40):
        # """AGC + normalization (trace wise)"""

        # proc = data.copy()

        # # --- AGC ---
        # for i in range(proc.shape[1]):
            # trace = proc[:, i]
            # env = np.abs(hilbert(trace))
            # smooth = np.convolve(env, np.ones(agc_window)/agc_window, mode="same")
            # smooth[smooth == 0] = 1.0
            # proc[:, i] = trace / smooth

        # # --- Z-score normalization ---
        # mean = np.mean(proc, axis=0)
        # std = np.std(proc, axis=0)
        # std[std == 0] = 1.0
        # proc = (proc - mean) / std

        # return proc, mean, std
    # @staticmethod
    # def multifractal_spectrum_width(signal):
        # """
        # Robust multifractal spectrum width estimator
        # Noise-dominated ICA components have larger width
        # """
        # import numpy as np

        # signal = np.asarray(signal)
        # signal = signal[np.isfinite(signal)]

        # if signal.size < 128:
            # return 0.0

        # scales = [2, 4, 8, 16, 32, 64]
        # q = [-2, -1, 1, 2]

        # eps = 1e-12
        # Fq = []

        # for qq in q:
            # vals = []
            # used_scales = []
            # for s in scales:
                # if s >= signal.size:
                    # continue
                # diff = np.abs(signal[s:] - signal[:-s]) + eps
                # vals.append(np.mean(diff ** qq))
                # used_scales.append(s)

            # if len(vals) > 1:
                # Fq.append(
                    # np.polyfit(np.log(used_scales),
                               # np.log(vals), 1)[0]
                # )

        # if len(Fq) < 2:
            # return 0.0

        # return float(max(Fq) - min(Fq))

        

    # def ica_multifractal_denoise(self, block_size=10, reject_ratio=0.12,  agc_window=50, progress_var=None, progress_win=None):
        # raw = self.loader.data
        # if raw is None:
            # return

        # # --- preprocess ---
        # data, mean, std = self.preprocess_for_ica(raw)

        # n_samples, n_traces = data.shape
        # output = np.zeros_like(data)

        # from sklearn.decomposition import FastICA

        # for i in range(0, n_traces, block_size):
            # j = min(i + block_size, n_traces)
            # block = data[:, i:j].T

            # ica = FastICA(
                # whiten="unit-variance",
                # max_iter=3000,
                # tol=1e-3,
                # random_state=0
            # )

            # try:
                # S = ica.fit_transform(block)
                # A = ica.mixing_
            # except Exception:
                # output[:, i:j] = block.T
                # continue

            # if A is None:
                # output[:, i:j] = block.T
                # continue

            # # --- multifractal discrimination ---
            # widths = np.array([
                # # multifractal_spectrum_width(S[:, k])
                # self.multifractal_spectrum_width(S[:, k])

                # for k in range(S.shape[1])
            # ])

            # n_reject = int(reject_ratio * len(widths))
            # reject_idx = np.argsort(widths)[-n_reject:]

            # S[:, reject_idx] = 0.0
            # recon = np.dot(S, A.T)

            # output[:, i:j] = recon.T

        # # --- undo normalization ---
        # output = output * std + mean
        # if progress_var is not None:
            # progress = 100.0 * min(i + block_size, n_traces) / n_traces
            # progress_var.set(progress)
            # if progress_win is not None:
                # progress_win.update_idletasks()

        # self.loader.data = output
        # if progress_var is not None:
            # progress_var.set(100.0)
            # if progress_win is not None:
                # progress_win.update_idletasks()
        
        # self.plot_gpr()
        
    # def _apply_ica_denoise(self, block, agc, perc, progress_var, win):
        # self.ica_multifractal_denoise(
            # block_size=block,
            # agc_window=agc,
            # reject_ratio=perc / 100.0,
            # progress_var=progress_var,
            # progress_win=win
        # )


    # def ica_denoise_dialog(self):
        # win = tk.Toplevel(self.root)
        # win.title("ICA Multifractal Denoising")
        # win.resizable(False, False)

        # # --- variables ---
        # block_var = tk.IntVar(value=10)
        # agc_var = tk.IntVar(value=50)
        # perc_var = tk.IntVar(value=12)
        # progress_var = tk.DoubleVar(value=0.0)


        # # --- layout ---
        # row = 0

        # tk.Label(win, text="Block size (traces)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=4, to=40, increment=1,
            # textvariable=block_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1

        # tk.Label(win, text="AGC window (samples)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=10, to=200, increment=5,
            # textvariable=agc_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1

        # tk.Label(win, text="Reject ratio (%)").grid(
            # row=row, column=0, padx=10, pady=6, sticky="w"
        # )
        # tk.Spinbox(
            # win, from_=5, to=50, increment=1,
            # textvariable=perc_var, width=8
        # ).grid(row=row, column=1, padx=5, pady=6)
        # row += 1
        # # --- progress bar ---
        # ttk.Label(win, text="Progress").grid(
            # row=row, column=0, padx=10, pady=(10, 2), sticky="w"
        # )

        # progress = ttk.Progressbar(
            # win,
            # variable=progress_var,
            # maximum=100,
            # length=180,
            # mode="determinate"
        # )
        # progress.grid(row=row, column=1, padx=10, pady=(10, 2))
        # row += 1

        # # --- buttons ---
        # # --- buttons --- 
        # tk.Button( win, text="Apply", command=lambda: self._apply_ica_denoise( block_var.get(), agc_var.get(), perc_var.get(), progress_var, win ) ) .grid(row=row, column=0, padx=10, pady=12)
        

        # tk.Button(
            # win, text="Cancel",
            # command=win.destroy
        # ).grid(row=row, column=1, padx=10, pady=12)

        # win.grab_set()


 
    def show_ica_components(self, trace_start=0, block_size=20):
        data, _, _ = self.preprocess_for_ica(self.loader.data)

        block = data[:, trace_start:trace_start + block_size].T

        from sklearn.decomposition import FastICA
        ica = FastICA(whiten="unit-variance", random_state=0)
        S = ica.fit_transform(block)

        fig, axes = plt.subplots(
            nrows=min(8, S.shape[1]),
            figsize=(6, 10),
            sharex=True
        )

        for i, ax in enumerate(axes):
            ax.plot(S[:, i], color="black")
            ax.set_title(f"ICA Component {i}")

        fig.tight_layout()
        plt.show()

    def run_layer_picking(self, amp_p, max_jump, min_cov):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        # ---- Progress bar popup ----
        prog = tk.Toplevel(self.root)
        prog.title("Picking Layers")
        prog.geometry("300x90")
        prog.transient(self.root)
        prog.grab_set()

        tk.Label(prog, text="Picking layers...").pack(pady=5)
        pb = ttk.Progressbar(prog, mode="indeterminate", length=250)
        pb.pack(pady=10)
        pb.start(10)

        try:
            self.root.update_idletasks()

            layers = self.loader.pick_layers_semi_auto(
                amp_percentile=amp_p,
                max_vertical_jump=max_jump,
                min_trace_coverage=min_cov
            )

            pb.stop()
            prog.destroy()

            if not layers:
                messagebox.showinfo("Result", "No layers detected.")
                return

            # ---- Plot ----
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(self.loader.data, [2, 98])
            ax.imshow(
                self.loader.data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            for lyr in layers:
                xs = [p[1] for p in lyr]
                ys = [p[0] for p in lyr]
                ax.plot(xs, ys, 'r', linewidth=1.5)

            ax.set_title("Semi Automatic Layer Picking")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

            print(f"Layers picked: {len(layers)}")

        except Exception as e:
            pb.stop()
            prog.destroy()
            messagebox.showerror("Layer Picking Error", str(e))

    
    
    def layer_picker_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Layer Picking Parameters")
        popup.geometry("300x220")
        popup.transient(self.root)
        popup.grab_set()

        tk.Label(popup, text="Amplitude Percentile").pack(pady=4)
        amp_var = tk.DoubleVar(value=55)
        tk.Entry(popup, textvariable=amp_var).pack()

        tk.Label(popup, text="Max Vertical Jump (samples)").pack(pady=4)
        jump_var = tk.IntVar(value=4)
        tk.Entry(popup, textvariable=jump_var).pack()

        tk.Label(popup, text="Min Trace Coverage (0–1)").pack(pady=4)
        cov_var = tk.DoubleVar(value=0.4)
        tk.Entry(popup, textvariable=cov_var).pack()

        def run():
            popup.destroy()
            self.run_layer_picking(
                amp_var.get(),
                jump_var.get(),
                cov_var.get()
            )

        tk.Button(popup, text="Run", bg="#C8E6C9", command=run).pack(pady=12)

    def show_amplitude_map(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            # ---- Compute envelope ----
            from scipy.signal import hilbert
            env = np.abs(hilbert(self.loader.data, axis=0))

            # ---- Plot on existing canvas ----
            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(env, [5, 95])
            im = ax.imshow(
                env,
                cmap="hot",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title("Instantaneous Amplitude (Envelope Map)")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            # ax.invert_yaxis()

            self.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            self.figure.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Amplitude Map Error", str(e))
    def peaks_extraction_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Peaks Extraction")
        popup.geometry("320x360")
        popup.transient(self.root)
        popup.grab_set()

        # --- Peak type ---
        tk.Label(popup, text="Select Peaks").pack(pady=4)
        peak_type = tk.StringVar(value="all")
        ttk.Combobox(
            popup,
            textvariable=peak_type,
            values=["all", "positive", "negative"],
            state="readonly"
        ).pack()

        # --- Max peaks ---
        tk.Label(popup, text="Max # of Points (per trace)").pack(pady=4)
        max_peaks = tk.IntVar(value=3)
        tk.Entry(popup, textvariable=max_peaks).pack()

        # --- Vertical width ---
        tk.Label(popup, text="Samples / Point").pack(pady=4)
        samp_width = tk.IntVar(value=3)
        tk.Entry(popup, textvariable=samp_width).pack()

        # --- Start / End samples ---
        tk.Label(popup, text="Start Sample").pack(pady=4)
        start_samp = tk.IntVar(value=0)
        tk.Entry(popup, textvariable=start_samp).pack()

        tk.Label(popup, text="End Sample").pack(pady=4)
        end_samp = tk.IntVar(value=self.loader.data.shape[0] if self.loader.data is not None else 0)
        tk.Entry(popup, textvariable=end_samp).pack()

        def run():
            popup.destroy()
            self.run_peaks_extraction(
                peak_type.get(),
                max_peaks.get(),
                samp_width.get(),
                start_samp.get(),
                end_samp.get()
            )

        tk.Button(popup, text="Apply", bg="#C8E6C9", command=run).pack(pady=12)
            
    def run_peaks_extraction(
        self,
        peak_type,
        max_peaks,
        samp_width,
        start_samp,
        end_samp
    ):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        try:
            peak_data = self.loader.extract_peaks(
                peak_type=peak_type,
                max_peaks=max_peaks,
                samples_per_point=samp_width,
                start_sample=start_samp,
                end_sample=end_samp
            )

            self.figure.clf()
            ax = self.figure.add_subplot(111)

            vmin, vmax = np.percentile(peak_data[peak_data != 0], [5, 95]) \
                         if np.any(peak_data) else (None, None)

            ax.imshow(
                peak_data,
                cmap=self.cmap_var.get(),
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(f'Peaks Extraction - {self.loader.line_name}: {self.loader.file_path}')
            # ax.set_title("Peaks Extraction")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample / Depth")
            # ax.invert_yaxis()

            self.figure.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Peaks Extraction Error", str(e))
    def background_removal_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Background Removal (Horizontal FIR)")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # ------------------ BR Type ------------------
        tk.Label(win, text="BR Type").grid(row=0, column=0, padx=10, pady=6, sticky="e")

        br_type = tk.StringVar(value="full")
        ttk.Combobox(
            win,
            textvariable=br_type,
            values=["full", "scan_range", "adaptive"],
            state="readonly",
            width=15
        ).grid(row=0, column=1, padx=5, pady=6)

        # ------------------ Filter Length ------------------
        tk.Label(win, text="Filter Length (scans)").grid(
            row=1, column=0, padx=10, pady=6, sticky="e"
        )
        flen = tk.IntVar(value=200)
        tk.Entry(win, textvariable=flen, width=10).grid(row=1, column=1)

        # ------------------ Scan Range ------------------
        tk.Label(win, text="Start Scan").grid(row=2, column=0, padx=10, pady=6, sticky="e")
        start_var = tk.IntVar(value=0)
        tk.Entry(win, textvariable=start_var, width=10).grid(row=2, column=1)

        tk.Label(win, text="End Scan").grid(row=3, column=0, padx=10, pady=6, sticky="e")
        end_var = tk.IntVar(value=self.loader.data.shape[1])
        tk.Entry(win, textvariable=end_var, width=10).grid(row=3, column=1)

        # ------------------ APPLY ------------------
        def apply():
            try:
                self.loader.background_removal(
                    br_type=br_type.get(),
                    filter_length=flen.get(),
                    start_scan=start_var.get(),
                    end_scan=end_var.get()
                )
                win.destroy()
                self.plot_gpr()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(
            win, text="Apply",
            bg="#4CAF50", fg="white",
            width=12,
            command=apply
        ).grid(row=4, column=0, columnspan=2, pady=12)
    def range_gain_dialog(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Range Gain")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # ---------------- Gain Type ----------------
        tk.Label(win, text="Gain Type").grid(row=0, column=0, padx=10, pady=6, sticky="e")
        gain_type = tk.StringVar(value="automatic")
        ttk.Combobox(
            win,
            textvariable=gain_type,
            values=["automatic", "linear", "exponential", "smart"],
            state="readonly",
            width=15
        ).grid(row=0, column=1)

        # ---------------- Number of Points ----------------
        tk.Label(win, text="# of Points").grid(row=1, column=0, padx=10, pady=6, sticky="e")
        npts = tk.IntVar(value=6)
        tk.Entry(win, textvariable=npts, width=10).grid(row=1, column=1)

        # ---------------- Overall Gain ----------------
        tk.Label(win, text="Overall Gain (dB)").grid(row=2, column=0, padx=10, pady=6, sticky="e")
        ogain = tk.DoubleVar(value=3.0)
        tk.Entry(win, textvariable=ogain, width=10).grid(row=2, column=1)

        # ---------------- Horizontal TC ----------------
        tk.Label(win, text="Horiz TC (scans)").grid(row=3, column=0, padx=10, pady=6, sticky="e")
        htc = tk.IntVar(value=15)
        tk.Entry(win, textvariable=htc, width=10).grid(row=3, column=1)

        # ---------------- Apply ----------------
        def apply():
            try:
                self.loader.range_gain(
                    gain_type=gain_type.get(),
                    n_points=npts.get(),
                    overall_gain_db=ogain.get(),
                    horiz_tc=htc.get()
                )
                win.destroy()
                self.plot_gpr()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(
            win,
            text="Apply",
            bg="#2196F3",
            fg="white",
            width=12,
            command=apply
        ).grid(row=4, column=0, columnspan=2, pady=12)
    
    # ────────────────────────────────────────────────────────────────
    #  Deconvolution with parameter dialog + progress bar
    # ────────────────────────────────────────────────────────────────

    def deconvolution_dialog(self):
        """
        Opens a dialog where user can set deconvolution parameters
        """
        if self.loader.data is None:
            messagebox.showinfo("Info", "Load GPR data first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Predictive Deconvolution Settings")
        dialog.geometry("420x380")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # ─── Variables ───────────────────────────────────────────────
        op_len_var   = tk.IntVar(value=32)
        lag_var      = tk.IntVar(value=8)
        prewhite_var = tk.DoubleVar(value=0.08)
        gain_var     = tk.DoubleVar(value=4.0)
        start_var    = tk.IntVar(value=60)
        end_var      = tk.IntVar(value=self.loader.data.shape[0])

        progress_var = tk.DoubleVar(value=0.0)

        row = 0

        # ─── Labels & Entries ──────────────────────────────────────
        tk.Label(dialog, text="Operator length (samples):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=op_len_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Prediction lag (samples):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=lag_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Prewhitening (%):").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=prewhite_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Overall gain factor:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=gain_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="Start sample:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=start_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        tk.Label(dialog, text="End sample:").grid(row=row, column=0, padx=12, pady=6, sticky="e")
        tk.Entry(dialog, textvariable=end_var, width=10).grid(row=row, column=1, padx=8, pady=6)
        row += 1

        # Progress bar
        tk.Label(dialog, text="Progress:").grid(row=row, column=0, padx=12, pady=(16,4), sticky="e")
        progress = ttk.Progressbar(dialog, orient="horizontal", length=260,
                                   mode="determinate", variable=progress_var)
        progress.grid(row=row, column=1, padx=8, pady=(16,4), sticky="w")
        row += 1

        # ─── Buttons ───────────────────────────────────────────────
        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=20)

        def run_decon():
            try:
                op_len   = op_len_var.get()
                lag      = lag_var.get()
                prewhite = prewhite_var.get()
                gain     = gain_var.get()
                start    = start_var.get()
                end      = end_var.get()

                if op_len < 4 or lag < 1 or lag >= op_len:
                    messagebox.showwarning("Invalid parameters", "Check operator length and lag values.")
                    return

                if start < 0 or end <= start or end > self.loader.data.shape[0]:
                    messagebox.showwarning("Invalid gate", "Check start/end sample values.")
                    return

                # Close dialog and run processing
                dialog.destroy()

                self._run_predictive_decon_with_progress(
                    operator_length=op_len,
                    prediction_lag=lag,
                    prewhitening=prewhite,
                    overall_gain=gain,
                    start_sample=start,
                    end_sample=end,
                    progress_var=progress_var
                )

            # except Exception as exc:
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("Deconvolution failed", str(err)))

        tk.Button(btn_frame, text="Apply Deconvolution",
                  command=run_decon,
                  bg="#4CAF50", fg="white", width=18, font=("Helvetica", 10, "bold")).pack(side="left", padx=10)

        tk.Button(btn_frame, text="Cancel",
                  command=dialog.destroy,
                  width=12).pack(side="left", padx=10)

    def _run_predictive_decon_with_progress(self,
                                           operator_length,
                                           prediction_lag,
                                           prewhitening,
                                           overall_gain,
                                           start_sample,
                                           end_sample,
                                           progress_var):
        """
        Runs deconvolution in the background thread + updates progress
        """
        def task():
            try:
                self.loader.predictive_deconvolution(
                    operator_length=operator_length,
                    prediction_lag=prediction_lag,
                    prewhitening=prewhitening,
                    overall_gain=overall_gain,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    progress_callback=lambda p: progress_var.set(p * 100)
                )
                # Refresh display on main thread
                self.root.after(0, self.plot_gpr)
                self.root.after(0, lambda: messagebox.showinfo("Done", "Predictive deconvolution completed."))

            except Exception as exc:
                self.root.after(0, lambda err=exc: messagebox.showerror("Deconvolution failed", str(err)))

            finally:
                self.root.after(0, lambda: progress_var.set(0))

        # Run in background thread so GUI stays responsive
        import threading
        thread = threading.Thread(target=task, daemon=True)
        thread.start() 

    # ----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------- 
    def show_metadata(self):
        meta = self.loader.get_metadata()
        hdr = self.loader.get_hdr_parameters()
        
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, "=== FILE & DATA INFO ===\n")
        for k, v in meta.items():
            self.meta_text.insert(tk.END, f"{k:22}: {v}\n")
        
        self.meta_text.insert(tk.END, "\n=== HDR PARAMETERS ===\n")
        for k, v in hdr.items():
            self.meta_text.insert(tk.END, f"{k:22}: {v}\n")

    # def plot_gpr(self):
        # if self.cmap_var.get() == "wiggle":
            # self.plot_wiggle()
            # return

        # self.figure.clf()
        # ax = self.figure.add_subplot(111)

        # data = self.loader.data
        # if data is None:
            # return

        # x_axis = (
            # self.loader.xyz['X'].values
            # if self.loader.xyz is not None and len(self.loader.xyz) > 0
            # else np.arange(data.shape[1])
        # )

        # y_axis = (
            # self.loader.depth
            # if self.loader.depth is not None
            # else np.arange(data.shape[0])
        # )

        # extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]

        # # ────────────────────────────────────────────────
        # # Improved contrast handling
        # # ────────────────────────────────────────────────
        # abs_data = np.abs(data)
        # mad = np.median(abs_data)                      # robust scale
        # if mad < 1e-6:                                 # almost zero data
            # vmin, vmax = -1, 1
        # else:
            # clip = 5.0 * mad                           # aggressive but GPR-friendly
            # vmin = -clip
            # vmax = clip

        # # Alternative: still use percentiles but with floor
        # # p2, p98 = np.percentile(data, [2, 98])
        # # vmin = max(p2, -3 * mad)
        # # vmax = min(p98,  3 * mad)

        # self.current_image = ax.imshow(
            # data,
            # cmap=self.cmap_var.get(),
            # aspect='auto',
            # extent=extent,
            # vmin=vmin,
            # vmax=vmax,
            # interpolation='nearest'   # optional: sharper look
        # )

        # ax.set_xlabel("Distance (m)")
        # ax.set_ylabel("Depth (m)")
        # ax.set_title(f'GPR Section - {self.loader.line_name} {self.loader.file_path}')

        # # ax.invert_yaxis()           # ← make sure this is here!
        # self.figure.tight_layout()
        # self.canvas.draw_idle()            
    def plot_gpr(self):
        if self.cmap_var.get() == "wiggle":
            self.plot_wiggle()
            return

        self.figure.clf()
        ax = self.figure.add_subplot(111)

        data = self.loader.data
        if data is None:
            return

        x_axis = (
            self.loader.xyz['X'].values
            if self.loader.xyz is not None and len(self.loader.xyz) > 0
            else np.arange(data.shape[1])
        )

        y_axis = (
            self.loader.depth
            if self.loader.depth is not None
            else np.arange(data.shape[0])
        )

        extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
        vmin, vmax = np.percentile(data, [2, 98])

        self.current_image = ax.imshow(
            data,
            cmap=self.cmap_var.get(),
            aspect='auto',
            extent=extent,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        # ax.set_title(f"GPR Section - {self.loader.line_name}")
        ax.set_title(f'GPR Section - {self.loader.line_name} {self.loader.file_path}')

        self.figure.tight_layout()
        self.canvas.draw_idle()
        
 
    def reset_data(self):
        if hasattr(self, "original_data") and self.original_data is not None:
            self.loader.data = self.original_data.copy()
            self.plot_gpr()
        else:
            messagebox.showinfo("Info", "No original data to reset.")
        
    def save_figure(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "Nothing to save yet.")
            return

        filetypes = [
            ("PNG image", "*.png"),
            ("JPEG image", "*.jpg"),
            ("PDF document", "*.pdf"),
            ("SVG vector", "*.svg"),
            ("TIFF image", "*.tiff"),
            ("All files", "*.*")
        ]

        proc_tag = "-".join(self.loader.process_history) if self.loader.process_history else "RAW"
        default_name = f"{self.loader.line_name}_{proc_tag}_{self.cmap_var.get()}"

        # default_name = f"{self.loader.line_name}_{self.cmap_var.get()}"

        filepath = filedialog.asksaveasfilename(
            title="Save GPR Figure",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=filetypes
        )

        if not filepath:
            return

        try:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )
            messagebox.showinfo("Saved", f"Figure saved successfully:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def zoom(self, scale_factor):
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_center = np.mean(xlim)
        y_center = np.mean(ylim)

        x_half = (xlim[1] - xlim[0]) * scale_factor / 2
        y_half = (ylim[1] - ylim[0]) * scale_factor / 2

        ax.set_xlim(x_center - x_half, x_center + x_half)
        ax.set_ylim(y_center - y_half, y_center + y_half)

        self.canvas.draw_idle()
    
 
    def change_colormap(self):
        if self.loader.data is None:
            messagebox.showinfo("Info", "No GPR data loaded yet.")
            return

        self.plot_gpr()
        

    def plot_wiggle(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        data = self.loader.data
        if data is None:
            return

        n_samples, n_traces = data.shape

        # X coordinates
        if self.loader.xyz is not None and len(self.loader.xyz) > 0:
            x = self.loader.xyz['X'].values
        else:
            x = np.arange(n_traces)

        # Y coordinates
        y = self.loader.depth if self.loader.depth is not None else np.arange(n_samples)

        # Normalize traces
        scale = 0.5 * np.median(np.diff(x)) if n_traces > 1 else 1.0
        norm = np.max(np.abs(data))
        if norm == 0:
            norm = 1.0

        for i in range(n_traces):
            trace = data[:, i] / norm
            xtrace = x[i] + trace * scale

            ax.plot(xtrace, y, color="black", linewidth=0.6)
            ax.fill_betweenx(
                y,
                x[i],
                xtrace,
                where=(trace > 0),
                color="black",
                alpha=0.7
            )

        ax.invert_yaxis()
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"GPR Wiggle Plot - {self.loader.line_name}")
        ax.grid(False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.current_image = None

            
    def plot_geox(self):
        if self.loader.xyz is None or len(self.loader.xyz) == 0:
            messagebox.showinfo("Info", "No GEOX data available.")
            return

        self.figure.clf()
        axes = self.figure.subplots(2, 2)
        axes = axes.flat

        xyz = self.loader.xyz

        axes[0].plot(xyz['X'], xyz['Y'])
        axes[0].set_title("Survey Path (X-Y)")

        axes[1].plot(xyz['Marker'], xyz['Z'])
        axes[1].set_title("Elevation")

        axes[2].plot(xyz['Marker'], xyz['Lat'])
        axes[2].set_title("Latitude")

        axes[3].plot(xyz['Marker'], xyz['Lon'])
        axes[3].set_title("Longitude")

        self.figure.tight_layout()
        self.canvas.draw_idle()

 
    
    def show_stats(self):
        stats = self.loader.get_statistics()
        if not stats:
            messagebox.showinfo("Info", "No data loaded yet.")
            return
        
        win = tk.Toplevel(self.root)
        win.title("Data Statistics")
        win.geometry("600x700")
        
        text = scrolledtext.ScrolledText(win, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for k, v in stats.items():
            text.insert(tk.END, f"{k:24}: {v}\n")
            
    def on_scroll(self, event):
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.canvas.draw_idle()
        
    def clear_all(self):
        self.loader.clear_data()
        self.meta_text.delete("1.0", tk.END)

        self.figure.clf()
        self.current_image = None
        self.canvas.draw_idle()

        if self.ascan_cid is not None:
            self.canvas.mpl_disconnect(self.ascan_cid)
            self.ascan_cid = None

        self.ascan_enabled = False
        self.ascan_win.withdraw()

  
if __name__ == "__main__":
    root = tk.Tk()
    app = V00ReaderGUI(root)

    root.mainloop()
