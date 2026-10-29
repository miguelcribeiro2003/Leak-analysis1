"""
Resample a water-hammer transient-pressure .txt record to a new sampling frequency.

INPUT  : one TAB-separated .txt with
           row 0 : column names   ->  Time   Channel A   Channel B  [...]  Channel D
           row 1 : units          ->  (s)    (m)         (m)        [...]  (l/h)
           row 2 : blank line
           row 3+: numeric data
         Columns are typed by their UNIT (not by position), so any number of
         transducers is supported:
           (s)    -> time axis
           (m)    -> pressure transducer  (1, 2, 3, ... of them)
           (l/h)  -> flow rate

WHAT IT DOES
  1. Reads the record and reports, before resampling:
       - input sampling frequency  (from Dt between consecutive instants)
       - record start, end (t_end) and total duration
       - maneuver onset t0  (reused from the analysis notebook: robust |dH|
         threshold = median + k*MAD over the pre-maneuver baseline)
       - the interval (t_end - t0)
  2. Asks the user for the NEW frequency (Hz).
  3. Resamples the WHOLE recorded interval onto a uniform grid at Dt_new = 1/f_new
     by LINEAR interpolation:
       - f_new > f_in  -> more points (interpolated between acquired samples)
       - f_new < f_in  -> fewer points (re-gridded by interpolation); where the
         new grid lands exactly on acquired instants the original values are kept
  4. Writes a new .txt in the SAME format (names + units + blank line preserved),
     named  <original>_<f_new>Hz.txt   (e.g. ..._40Hz.txt).

Author note: code in English by request; comments kept minimal.
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Onset detection  (reused verbatim from Analysis_average_signals_V4.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
def _dh_threshold(dh, t, base_end_t, multiplier):
    """Robust threshold from the pre-maneuver baseline (median + k*MAD)."""
    base_mask = t[:-1] <= base_end_t           # dh has len(signal)-1
    baseline  = dh[base_mask]
    median    = np.median(baseline)
    mad       = np.median(np.abs(baseline - median))
    return median + multiplier * mad


def detect_onset(signal, t, base_end_t=-0.01, multiplier=10, consecutive=3):
    """First index where |dH| stays above the robust threshold for
    `consecutive` samples. Returns None if no onset is found."""
    dh    = np.abs(np.diff(signal))
    thr   = _dh_threshold(dh, t, base_end_t, multiplier)
    above = dh > thr
    for i in range(len(above) - consecutive):
        if np.all(above[i:i + consecutive]):
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_record(path):
    """Return (names, units, dataframe). Columns are kept in file order."""
    with open(path, 'r') as fh:
        names = fh.readline().rstrip('\r\n').split('\t')
        units = fh.readline().rstrip('\r\n').split('\t')
    df = pd.read_csv(path, sep='\t', header=0, skiprows=[1, 2])
    df = df.dropna(how='all')                                  # drop blank rows
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='any')
    n = len(df.columns)                                        # align headers
    names = (names + [''] * n)[:n]
    units = (units + [''] * n)[:n]
    return names, units, df.reset_index(drop=True)


def _freq_tag(hz):
    """Filename-friendly frequency tag: 40Hz, 8000Hz, 33.3Hz, ..."""
    if abs(hz - round(hz)) < 1e-9:
        return f'{int(round(hz))}Hz'
    return f'{hz:g}Hz'


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────
def resample_txt(path, new_hz, out_dir=None, verbose=True):
    """Resample one .txt record to `new_hz` and write a new file. Returns the
    output path."""
    names, units, df = _read_record(path)
    units_norm = [u.strip().lower() for u in units]

    # type columns by their unit
    time_pos  = next((i for i, u in enumerate(units_norm) if u == '(s)'), None)
    if time_pos is None:
        raise ValueError("No time column with unit '(s)' was found.")
    press_pos = [i for i, u in enumerate(units_norm) if u == '(m)']
    flow_pos  = [i for i, u in enumerate(units_norm) if u in ('(l/h)', '(l/hr)')]

    cols      = list(df.columns)
    time_col  = cols[time_pos]
    data_cols = [c for i, c in enumerate(cols) if i != time_pos]
    t         = df[time_col].to_numpy(float)

    # acquired sampling (robust to tiny jitter)
    dt_in = float(np.median(np.diff(t)))
    f_in  = 1.0 / dt_in

    # maneuver onset on the FIRST pressure channel (for reporting only)
    sig0   = df[cols[press_pos[0]]].to_numpy(float) if press_pos \
             else df[data_cols[0]].to_numpy(float)
    on_idx = detect_onset(sig0, t)
    t0     = t[on_idx] if on_idx is not None else None
    tstart, tend = t[0], t[-1]

    if verbose:
        print('=' * 62)
        print(f'  INPUT FILE : {os.path.basename(path)}')
        print('=' * 62)
        print(f'  Channels         : {len(press_pos)} transducer(s) (m), '
              f'{len(flow_pos)} flow (l/h)')
        print(f'  Samples          : {len(t)}')
        print(f'  Dt acquired      : {dt_in:.6g} s   ->   f_in = {f_in:.6g} Hz')
        print(f'  Record start     : {tstart:+.5f} s')
        print(f'  Record end t_end : {tend:+.5f} s')
        if t0 is not None:
            print(f'  Maneuver onset t0: {t0:+.5f} s')
            print(f'  Interval t_end-t0: {tend - t0:.5f} s')
        else:
            print('  Maneuver onset t0: not detected')
        print(f'  Record duration  : {tend - tstart:.5f} s')
        print('-' * 62)

    # new uniform grid over the WHOLE recorded interval [tstart, tend]
    dt_new = 1.0 / float(new_hz)
    n_new  = int(np.floor((tend - tstart) / dt_new + 1e-9)) + 1
    t_new  = tstart + np.arange(n_new) * dt_new

    if   dt_new < dt_in: mode = 'UPSAMPLING   (linear interpolation -> more points)'
    elif dt_new > dt_in: mode = 'DOWNSAMPLING (re-gridded by linear interpolation)'
    else:                mode = 'SAME Dt      (no resampling needed)'

    if verbose:
        print(f'  REQUESTED f_new  : {new_hz:g} Hz   ->   Dt_new = {dt_new:.6g} s')
        print(f'  Mode             : {mode}')
        print(f'  Output samples   : {n_new}')

    # interpolate every data column (pressures + flow) onto the new grid
    out = pd.DataFrame({time_col: t_new})
    for c in data_cols:
        out[c] = np.interp(t_new, t, df[c].to_numpy(float))

    # write preserving the names + units + blank-line header and CRLF endings
    base, ext = os.path.splitext(os.path.basename(path))
    out_name  = f'{base}_{_freq_tag(new_hz)}{ext or ".txt"}'
    out_dir   = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, out_name)

    with open(out_path, 'w', newline='') as fh:
        fh.write('\t'.join(names) + '\r\n')
        fh.write('\t'.join(units) + '\r\n')
        fh.write('\r\n')
        for row in out.itertuples(index=False):
            fh.write('\t'.join(f'{v:.8f}' for v in row) + '\r\n')

    if verbose:
        print('-' * 62)
        print(f'  OUTPUT FILE: {out_path}')
        print('=' * 62)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Interactive driver (runs when executed directly or pasted into a notebook cell)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    in_path = input('Path to the input .txt: ').strip().strip('"').strip("'")

    # peek at the input sampling frequency BEFORE asking for the new one
    _, _, _df = _read_record(in_path)
    _t   = _df.iloc[:, 0].to_numpy(float)
    _fin = 1.0 / float(np.median(np.diff(_t)))
    print(f'\nInput sampling frequency: {_fin:.6g} Hz '
          f'(Dt = {1.0/_fin:.6g} s)\n')

    new_hz = float(input('New frequency to resample to (Hz): ').strip())
    print()
    resample_txt(in_path, new_hz)
