# GPR Viewer Application (v00readerV3.py)
@Mrinal Kanti Layek @layek.mk@gmail.com
## Summary of Fixes and Enhancements

This document summarizes the fixes and enhancements made to the GPR viewer application.

---

## 1. SciPy/PyHHT Compatibility Fix

### Problem
```
ImportError: cannot import name 'angle' from 'scipy'
```
The `pyhht` library (v0.1.0) attempts to import `scipy.angle`, which was removed in SciPy 1.12+ and moved to `numpy.angle`.

### Solution
Added a compatibility patch at the top of the file (before importing pyhht):

```python
import scipy
if not hasattr(scipy, 'angle'):
    scipy.angle = np.angle

from pyhht.emd import EMD
```

### Location
Lines 14-23 in `v00readerV2.py`

---

## 2. Decluttering Modules (MS, SVD, RNMF)

### Description
Added three GPR clutter removal methods based on the paper:
> Ge et al. 2024, IEEE TGRS - "Wavelet-GAN: A GPR Noise and Clutter Removal Method Based on Small Real Datasets"

### New GUI Buttons
Three new buttons added to the interface (column 15):
- **MS** - Mean Subtraction
- **SVD** - Singular Value Decomposition
- **RNMF** - Robust Non-negative Matrix Factorization

### Method Details

#### 2.1 Mean Subtraction (MS)
- **Function**: `ms_declutter_dialog()`, `_apply_mean_subtraction()`
- **Algorithm**: Subtracts the mean trace from all traces
- **Use Case**: Removes horizontal clutter (direct wave, ringing noise with strict horizontal distribution)
- **Parameters**: None (one-click operation)

#### 2.2 SVD Declutter
- **Function**: `svd_declutter_dialog()`, `_apply_svd_declutter()`
- **Algorithm**: Uses Singular Value Decomposition to separate clutter from signal
- **Use Case**: Removes first N singular value components which typically capture horizontal clutter
- **Parameters**:
  - Components to remove (default: 1, range: 1-20)

#### 2.3 RNMF Declutter
- **Function**: `rnmf_declutter_dialog()`, `_apply_rnmf_declutter()`
- **Algorithm**: Robust Non-negative Matrix Factorization
- **Use Case**: Separates low-rank clutter from sparse target responses
- **Parameters**:
  - Number of components/rank (default: 5)
  - Max iterations (default: 200)
  - Sparsity weight alpha (default: 0.1)
- **Note**: Includes progress bar during processing

### Key Features
- Original raw data preserved (all processing on copies)
- Each method adds process tag to history (MS, SVD-N, RNMF-N)
- Reset button restores original data
- Full backward compatibility with existing codebase

---

## 3. Predictive Deconvolution Fix

### Problem
The original deconvolution was removing almost all the signal:
- The prediction filter was predicting the entire signal (not just ringing)
- Subtracting prediction left near-zero residuals
- Only the top portion of data was visible; everything below was flat gray

### Solution
Complete rewrite of `predictive_deconvolution()` with proper RADAN-style implementation:

#### Key Changes

1. **Proper Prediction-Error Filter (PEF)**:
   ```python
   # PEF = [1, 0, ..., 0, -f0, -f1, ..., -fn]
   pef = np.zeros(lag + n)
   pef[0] = 1.0  # Unit spike preserves signal
   pef[lag:lag + n] = -f  # Negative filter removes predictable ringing
   ```

2. **Amplitude Preservation**:
   ```python
   # Normalize output to match input amplitude
   filtered = filtered * (trace_std / filtered_std)
   ```

3. **Better Autocorrelation**:
   - Uses unbiased autocorrelation estimate (divides by overlap count)
   - Better numerical stability

4. **Improved Regularization**:
   - Adds prewhitening to both autocorr[0] and the diagonal of R matrix
   - Prevents singular matrix issues

#### Updated Default Parameters
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| Prediction lag | 8 | 4 |
| Prewhitening | 0.08 | 0.1 |
| Overall gain | 4.0 | 1.0 |
| Start sample | 60 | 0 |

### Location
- Function: `predictive_deconvolution()` in `GPR2DLoader` class
- Dialog: `deconvolution_dialog()` in `GPR2DViewer` class

---

## 4. Wiggle Plot Enhancement (RADAN7-Style)

### Problem
Basic wiggle implementation lacked features found in commercial software like RADAN7.

### Solution
Complete rewrite with professional-grade features:

#### New Features

1. **Parameter Dialog** (`plot_wiggle()`):
   - Interactive settings before plotting
   - Real-time parameter adjustment

2. **Trace Decimation**:
   - `trace_skip` parameter (1=all, 2=every 2nd, etc.)
   - Auto-adjusts for large datasets (default: n_traces // 200)

3. **Variable Area Fill Modes**:
   - `positive` - Standard VA (fill positive peaks)
   - `negative` - Fill negative peaks
   - `both` - Fill both with different colors
   - `none` - Wiggle lines only

4. **Color Options**:
   - Positive fill color (black, blue, red, gray, darkblue)
   - Negative fill color (white, red, blue, gray, lightgray)
   - Background color (white, lightgray, beige, black)

5. **Display Controls**:
   - Amplitude scale factor
   - Clip percentage (clips outliers)
   - Line width control
   - Show/hide wiggle lines
   - Fill opacity (0-1)

6. **Professional Rendering**:
   - 99th percentile normalization (avoids outlier distortion)
   - Proper trace spacing calculation
   - Depth grid lines
   - Automatic axis scaling

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Trace skip | auto | Plot every Nth trace |
| Amplitude scale | 1.0 | Scale factor for amplitudes |
| Clip (%) | 100 | Clip at % of max amplitude |
| Fill mode | positive | Which peaks to fill |
| Positive color | black | Fill color for positive |
| Negative color | white | Fill color for negative |
| Line width | 0.5 | Wiggle line thickness |
| Show lines | True | Display wiggle traces |
| Fill opacity | 0.85 | Fill transparency |
| Background | white | Plot background color |

### Location
- Dialog: `plot_wiggle()` method
- Renderer: `_render_wiggle()` method

---

## Dependencies

Required Python packages:
```
numpy
scipy
matplotlib
tkinter
pandas
pyproj
geopandas
shapely
contextily
scikit-learn
pyhht
pywt
opencv-python (cv2)
scikit-image
folium (optional)
```

### Installation
```bash
pip install numpy scipy matplotlib pandas pyproj geopandas shapely contextily scikit-learn pyhht pywt opencv-python scikit-image folium
```

---

## Usage

Run the application:
```bash
python v00readerV2.py
```

### Workflow
1. **Load Data**: Use file browser to load .v00, .d00, or .dt files
2. **Apply Processing**: Use toolbar buttons for various processing options
3. **Declutter**: Use MS, SVD, or RNMF buttons for clutter removal
4. **Deconvolution**: Click "Deconvolution" button to remove ringing
5. **Visualization**: Switch between image, wiggle, and other plot modes
6. **Reset**: Click "Reset" to restore original data

---

## File Structure

```
v00readerV2.py
├── GPR2DLoader class
│   ├── Data loading (v00, d00, dt, HDR, GEOX)
│   ├── predictive_deconvolution()  [FIXED]
│   ├── Processing methods
│   └── Data management
│
└── GPR2DViewer class
    ├── GUI setup
    ├── Plot functions
    │   ├── plot_gpr() - Image display
    │   ├── plot_wiggle() - Wiggle dialog [ENHANCED]
    │   └── _render_wiggle() - Wiggle renderer [NEW]
    ├── Decluttering modules [NEW]
    │   ├── ms_declutter_dialog()
    │   ├── _apply_mean_subtraction()
    │   ├── svd_declutter_dialog()
    │   ├── _apply_svd_declutter()
    │   ├── rnmf_declutter_dialog()
    │   └── _apply_rnmf_declutter()
    └── deconvolution_dialog() [FIXED defaults]
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-29 | Added MS, SVD, RNMF decluttering modules |
| 2026-01-29 | Fixed pyhht/scipy compatibility issue |
| 2026-01-29 | Fixed predictive deconvolution (amplitude preservation) |
| 2026-01-29 | Enhanced wiggle plot (RADAN7-style features) |

---

## References

1. Ge et al. 2024, IEEE TGRS - "Wavelet-GAN: A GPR Noise and Clutter Removal Method Based on Small Real Datasets"
2. GSSI RADAN 7 User Manual
3. Yilmaz, O. (2001). Seismic Data Analysis (for deconvolution theory)

