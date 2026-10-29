# -*- coding: utf-8 -*-
"""Batch runner for the SIMULATED signals (Numerical_data_CS/*.dat).

Same role as batch_analysis.py, and the same output schema: one row per leak
case, written to Numerical_data_CS/pair_single.csv, so that the report_*.py
scripts treat the numerical dataset exactly like CS and CC.

    python batch_simulated.py

What it computes, with the functions of analysis.py:
  - location: an.locate for every celerity, the ordering criterion
    (an.order_constrained_errors) and the blind zone of Section 5.4;
  - initial pressure drop: an.damping_NL (ratio_H_self, dH_m, dHJ_NL/L);
  - residual energy: an.residual_energy (E_R/E0);
  - damping over time: an.decay_rate (zeta_total, zeta_NL, zeta_leak);
  - ratio_Q: the TRUE value prescribed in the model, from the orifice equation
    with the steady head at the leak, summed over the leaks, not a flow-meter
    reading.

The leak diameters of the model are NOT the nominal ones. The model was set up
with the effective diameters measured on the rig (Table 4.x, D_exp): 0.60, 1.25,
1.62 and 2.29 mm for the nominal 0.5, 1.0, 1.5 and 2.0 mm orifices, with
C_d = 0.6. Using the nominal diameter here understates QL by up to a factor of
two and biases every calibration slope; the steady state of the .dat files
confirms D_exp to within 1-2 % (leak flow from the friction balance between the
reservoir, the leak node and the valve).

The pre-processing is the same as for the experimental pairs (same functions):
smoothing, onset detection, vertical alignment of the leak test onto the no-leak
one and horizontal alignment by cross-correlation. The model has neither noise
nor jitter, but the leak lowers the steady head by up to 1.6 m here, and that
offset must be removed before measuring a drop in metres.

Pipe config CS_sim: the model records the head at the VALVE (PT1 = 0 m), the
leak nodes are on the MOC grid (2.000, 7.625 and 10.250 m) and the celerity is
the calibrated one, 1250 m/s.

Leak positions are used only to compute errors, never to pick an echo.
"""
import contextlib
import glob
import io
import os

import numpy as np
import pandas as pd

import analysis as an

ROOT = 'Numerical_data_CS'
OUT_CSV = os.path.join(ROOT, 'pair_single.csv')
PIPE_CONFIG = 'CS_sim'
# Steady state of the model, read from rede_rp/rp_*.dat (the solver's own output:
# node heads, leak area Af, leak discharge Qf, and the flow Q0 of every pipe).
# Q0 is the no-leak flow at the valve; QL is the model's own leak discharge, not a
# recomputed orifice equation. The .dat pressure records of Numerical_data_CS are
# the SAME runs: their valve head loss reproduces these flows to 0.1 L/h.
STEADY_DIR = 'rede_rp'
Q0_NOMINAL = 246.22 / 3.6e6       # m3/s, no-leak flow at the valve (rp_NL)


def wide(path, label):
    """The simulated record as the wide frame the experimental pipeline expects.

    The model writes a single pressure series, so PT2 repeats it and Q carries a
    constant: neither is used here (the celerity from the transducers and the one
    from Joukowsky need two sections and a real flow reading, and CS_sim marks the
    flow channel as invalid), but they keep the frame in the shape that
    vertical_align and horizontal_align_single_nl expect.
    """
    _, t, h = an.load_sim(path)
    return pd.DataFrame({f'{label}_PT1': h, f'{label}_PT2': h,
                         f'{label}_Q': np.full(len(h), Q0_NOMINAL)},
                        index=pd.Index(np.asarray(t, float), name='Time (s)'))


# Leak discharge of the model (L/h), read from rede_rp/rp_<LOC>_L<size>.dat.
# Falls back to the orifice equation only if a case has no steady-state file.
QL_MODEL = {
    ('LOC1', '0.5'):  18.0, ('LOC1', '1.0'):  78.8,
    ('LOC1', '1.5'): 132.5, ('LOC1', '2.0'): 223.2,
    ('LOC2', '0.5'):  18.4, ('LOC2', '1.0'):  78.8,
    ('LOC2', '1.5'): 133.2, ('LOC2', '2.0'): 224.6,
    ('LOC3', '0.5'):  18.0, ('LOC3', '1.0'):  78.8,
    ('LOC3', '1.5'): 131.0, ('LOC3', '2.0'): 219.6,
}

# effective leak diameter of the model (mm), by nominal label (mm): the model was
# run with the diameters measured on the rig, not with the nominal ones. Only used
# for the cases QL_MODEL does not cover (the two-leak runs).
D_EXP = {'0.5': 0.60, '1.0': 1.25, '1.5': 1.62, '2.0': 2.11}


def ratio_Q_true(leaks, head_at_leak, loc=None):
    """QL/Q0, with QL taken from the model's own steady state where available.

    For the single-leak cases QL is the Qf the solver prints (rede_rp/rp_*.dat),
    which needs no assumption about C_d or about the hole diameter. For the
    two-leak cases, which have no steady-state file here, it falls back to the
    orifice equation with the effective diameters the model was run with.
    """
    QL = 0.0
    for name, size in leaks:
        label = f'{float(size):.1f}'
        key = (name, label)
        if key in QL_MODEL:
            QL += QL_MODEL[key] / 3.6e6
            continue
        if label not in D_EXP:
            raise KeyError(f'no effective diameter for the {label} mm leak; '
                           f'add it to D_EXP')
        area = np.pi * (D_EXP[label] / 1000.0) ** 2 / 4.0
        QL += an.CD * area * np.sqrt(2 * an.G * head_at_leak)
    return QL / Q0_NOMINAL


def process_case(path, nl_path, meta):
    leaks = meta['leaks']
    lab_L, lab_NL = 'L', 'NL'
    with contextlib.redirect_stdout(io.StringIO()):
        # EXACTLY the pipeline of batch_analysis: smoothing, onset, vertical and
        # horizontal alignment, then the indicators. The model has no jitter and
        # no noise, but the leak does lower the steady head (up to 1.6 m here),
        # and that offset has to be removed the same way, or it goes straight
        # into dH_m, which is measured in metres.
        raw, raw_nl = wide(path, lab_L), wide(nl_path, lab_NL)
        events = an.detect_events(raw, raw_nl)
        va, va_nl, va_info = an.vertical_align(raw, raw_nl, events)
        # correlate=False: the model has no jitter, both runs start at the same
        # instant, and the correlation would answer the deformation the leak
        # itself causes in the front with a shift of up to 0.5 ms
        aligned, aligned_nl, _, _, align_info = an.horizontal_align_single_nl(
            va, va_nl, events, lab_NL, correlate=False)
        result = an.make_treated(aligned, aligned_nl)
        t = result.index.to_numpy(float)
        ath = an.a_theoretical()
        cels = {'Theoretical': ath,
                'M1_period': an.a_period(result['NL_PT1_mean'].to_numpy(float), t, ath)}
        T = an.period_time(cels)
        an.nondimensionalize(result, T)
        L = result['L_PT1_smooth'].to_numpy(float)
        N = result['NL_PT1_smooth'].to_numpy(float)
        L_nd = result['L_PT1_nondim'].to_numpy(float)
        N_nd = result['NL_PT1_nondim'].to_numpy(float)

        loc = an.locate(L, t, T, cels, n_leaks=len(leaks), NL_sig=N)
        x_reals = [an.LOC_POSITIONS[l] for l, _ in leaks]
        blind = an.location_errors_all(loc, x_reals) if loc else [None] * len(leaks)
        ordered = (an.order_constrained_errors(loc, x_reals, cels, t) if loc
                   else [None] * len(leaks))

        fsd = an.damping_NL(L, N, t, T)
        er = an.residual_energy(N_nd, L_nd, N, t, T)
        _Ls = result['L_PT1_nondim_self'].to_numpy(float)
        _Ns = result['NL_PT1_nondim_self'].to_numpy(float)
        er_c = an.residual_energy(_Ns, _Ls, N, t, T)
        t_on = fsd['t_on'] if fsd else None
        dc_L = an.decay_rate(L_nd, t, T, t_on=t_on)
        dc_N = an.decay_rate(N_nd, t, T, t_on=t_on)
        o = an.detect_onset(N, t, base_end_t=an.ONSET_BASE_END)
        k = an.maneuver_end(N, t, T)

    t_c = (t[k] - t[o]) if (k is not None and o is not None) else np.nan
    t1 = (loc['t1'] - loc['t_on']) if loc else np.nan
    a_ref = cels['M1_period'] if np.isfinite(cels['M1_period']) else cels['Theoretical']
    x_min = an.PT1_OFFSET + a_ref * (t_c - t1) / 2        # blind zone, Eq. 5.8

    base = dict(facility=PIPE_CONFIG, maneuver=meta['maneuver'], flow=meta['flow'],
                loc='+'.join(l for l, _ in leaks),
                leak='+'.join(s for _, s in leaks),
                n_leaks=len(leaks), NL_test=os.path.basename(nl_path),
                leak_test=os.path.basename(path),
                a_theoretical=cels['Theoretical'], a_period=cels['M1_period'],
                T_period_ms=T * 1e3, t1_ms=t1 * 1e3, t_fim_manobra_ms=t_c * 1e3,
                t_lim_ms=((fsd['t_lim'] - fsd['t_on']) * 1e3) if fsd else np.nan,
                x_min_domain=x_min,
                align_shift_ms=align_info['lag_ms'], peak_corr=align_info['peak'],
                leak_head_shift_m=va_info['leak_head_shift'])
    if fsd:
        base.update(ratio_H_self=fsd['ratio_H_self'],
                    H_star_max=fsd['H_star_max'], H_star_min=fsd['H_star_min'],
                    dH_over_dHJ=fsd['dH_over_dHJ'],
                    dH_over_HNL=fsd['dH_over_HNL'],
                    H0_m=fsd['H0_L'], H_max_m=fsd['H_max_m'],
                    H_min_m=fsd['H_min_m'], dH_m=fsd['dH_m'],
                    dt_drop_ms=fsd['dt_drop'] * 1e3,
                    dt_drop_dh_ms=fsd['dt_drop_dh'] * 1e3,
                    dHJ_NL=fsd['dHJ_NL'], dHJ_L=fsd['dHJ_L'])
        base['ratio_Q'] = ratio_Q_true(leaks, fsd['H0_L'])
    if er:
        base['E_R_over_E0'] = er['E_R_over_E0']
    if er_c:
        base['E_R_over_E0_self'] = er_c['E_R_over_E0']
    if dc_L and dc_N:
        base.update(zeta_total=dc_L['zeta'], zeta_NL=dc_N['zeta'],
                    zeta_leak=dc_L['zeta'] - dc_N['zeta'],
                    zeta_leak_u=float(np.hypot(dc_L['zeta_u'], dc_N['zeta_u'])),
                    r2_lnA_L=dc_L['r2'], n_cycles_L=dc_L['n_cycles'])

    rows = []
    for j, (lname, size) in enumerate(leaks):
        # leak 1 keeps the plain column names, the others take a _leakK suffix,
        # exactly as batch_analysis does
        sfx = '' if j == 0 else f'_leak{j + 1}'
        row = dict(base)
        row[f'leak_loc{sfx}'] = lname
        row[f'leak_size{sfx}'] = size
        row[f'x_real{sfx}'] = x_reals[j]
        row[f'in_domain{sfx}'] = bool(x_reals[j] > x_min)
        for name in cels:
            for tag, errs in (('', blind[j]), ('_ord', ordered[j])):
                e = (errs or {}).get(name) or {}
                x = e.get('x_leak', np.nan)
                row[f'x_{name}{tag}{sfx}'] = x
                row[f'x_err_{name}_m{tag}{sfx}'] = (x - x_reals[j]
                                                    if np.isfinite(x) else np.nan)
        row[f'echo_src_ord{sfx}'] = (ordered[j] or {}).get('_source')
        # delay of the echo assigned to this leak, blind and under the ordering
        # rule, so the reports can show t2 = t1 + delta_t
        row[f'delta_t_s{sfx}'] = ((blind[j] or {}).get('_echo') or {}).get('delta_t', np.nan)
        row[f'delta_t_s_ord{sfx}'] = ((ordered[j] or {}).get('_echo') or {}).get('delta_t', np.nan)
        rows.append(row)
    # one row per case, the extra leaks merged into it by suffix
    merged = {}
    for r in rows:
        merged.update(r)
    return merged


def _keep_newest(meta):
    """Uma corrida por caso: a mais recente, pela data no nome do ficheiro.

    As corridas re-feitas ficam na pasta ao lado das antigas (o calibre de
    2.0 mm foi refeito a 27/09 com D_exp = 2.10 mm em vez de 2.29 mm). Sem isto
    o mesmo caso entrava duas vezes na calibracao.
    """
    best = {}
    for path, m in meta.items():
        if not m:
            continue
        key = (m['maneuver'], m['flow'], m.get('is_nl'),
               tuple(m.get('leaks') or ()))
        prev = best.get(key)
        if prev is None or os.path.basename(path) > os.path.basename(prev):
            best[key] = path
    kept = {p: meta[p] for p in best.values()}
    for path in meta:
        if path not in kept:
            print(f'[skip] {os.path.basename(path)}: substituido por corrida '
                  f'mais recente')
    return kept


def main():
    an.configure_pipe(PIPE_CONFIG)
    files = sorted(glob.glob(os.path.join(ROOT, 'SIM_*.dat')))
    meta = {f: an.parse_filename(os.path.basename(f)) for f in files}
    meta = _keep_newest(meta)
    nl = {(m['maneuver'], m['flow']): f for f, m in meta.items() if m and m['is_nl']}
    rows = []
    for f, m in meta.items():
        if not m or m['is_nl']:
            continue
        nl_path = nl.get((m['maneuver'], m['flow']))
        if nl_path is None:
            print(f'[skip] {os.path.basename(f)}: no NL for {m["maneuver"]} {m["flow"]}')
            continue
        rows.append(process_case(f, nl_path, m))
    df = pd.DataFrame(rows).sort_values(['maneuver', 'flow', 'loc', 'leak'])
    df.to_csv(OUT_CSV, index=False)
    print(f'Written: {OUT_CSV}   ({len(df)} cases, {df.n_leaks.max()} leaks max)')
    cols = [c for c in ['maneuver', 'flow', 'loc', 'leak', 'x_real', 'x_Theoretical',
                        'x_err_Theoretical_m', 'in_domain', 'ratio_H_self',
                        'E_R_over_E0', 'zeta_leak', 'ratio_Q'] if c in df.columns]
    with pd.option_context('display.width', 200):
        print(df[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
