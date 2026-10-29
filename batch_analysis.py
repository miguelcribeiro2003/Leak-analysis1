# -*- coding: utf-8 -*-
"""Batch runner for the water-hammer leak analysis.

Processes EVERY leak test in a facility's flat dataset folder with the same
pipeline as experimental_analysis.ipynb (no plots): each leak test is
paired with the matching NL reference (same maneuver + flow), analysed, and
exported to <root>/outputs/. A full run REBUILDS pair_single.csv from scratch.

Data files are discovered/selected by parsing their names (analysis.parse_
filename): KIND_FACILITY_MANEUVER_FLOW_(NL | LOCi_L<size> ...)_[test]_DATE.

Usage:
    python batch_analysis.py                # CS, all maneuvers/flows
    python batch_analysis.py CC             # CC
    python batch_analysis.py CS 250 350     # only these flows; CSV left untouched
"""
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import analysis as an

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

# ── Batch configuration ───────────────────────────────────────────────────────
FACILITY_DIRS = {
    'CS': 'Experimental_data_CS',      # copper straight
    'CC': 'Experimental_data_CC',      # copper coil
}
DEFAULT_FACILITY = 'CS'

NL_TEST_DEFAULT = 1          # NL reference test (1-based) used for every case...
NL_TEST_BY_FLOW = {          # ...except these per-facility/flow overrides
    'CS': {250: 3},
    'CC': {},
}

WRITE_EXPORTS = True          # write the 4 per-test files for every pair
VERBOSE = False               # True -> stream the full pipeline log of every pair

# Pipeline notes worth surfacing even for pairs that completed successfully.
WARN_PREFIXES = ('[flow]', '[period_time]', '[warn]', '[decay_rate]')


def discover(root, only_flows=None):
    """Typologies with leak files: [(maneuver, flow, leaks, has_nl), ...]."""
    leak_typ, nl_set = set(), set()
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(('.txt', '.dat')):
            continue
        a = an.parse_filename(fn)
        if a is None:                       # not a data file (e.g. a report)
            continue
        if only_flows and a['flow'] not in only_flows:
            continue
        if a['is_nl']:
            nl_set.add((a['maneuver'], a['flow']))
        elif a['leaks']:
            leak_typ.add((a['maneuver'], a['flow'], tuple(a['leaks'])))
    plan = []
    for man, flow, leaks in sorted(leak_typ):
        plan.append((man, flow, list(leaks), (man, flow) in nl_set))
    return plan


def process_pair(root, facility, maneuver, flow, leaks, nl_test, leak_test):
    """Same pipeline as the notebook, without plots. Returns the results row."""
    loc_name = '+'.join(l for l, _ in leaks)
    leak     = '+'.join(s for _, s in leaks)
    lab_NL, NL_raw0 = an.load_one(root, maneuver, flow, nl_test, is_nl=True)
    lab_L,  L_raw0  = an.load_one(root, maneuver, flow, leak_test, leaks=leaks)
    if not NL_raw0.index.equals(L_raw0.index):
        NL_raw0 = an.interp_to_index(NL_raw0, L_raw0.index)

    ath     = an.a_theoretical()
    x_reals = [an.LOC_POSITIONS.get(l, np.nan) for l, _ in leaks]
    x_real  = x_reals[0]
    raw_data, raw_data_nl = an.to_wide(L_raw0, lab_L), an.to_wide(NL_raw0, lab_NL)

    events = an.detect_events(raw_data, raw_data_nl)
    va_data, va_data_nl, va_info = an.vertical_align(raw_data, raw_data_nl, events)
    (aligned_data, aligned_data_nl, va_data_clock, va_data_nl_clock,
     align_info) = an.horizontal_align_single_nl(va_data, va_data_nl, events, lab_NL)

    result = an.make_treated(aligned_data, aligned_data_nl)
    t = result.index.to_numpy(float)

    cel = dict(
        a_theoretical=ath,
        a_period=an.a_period(result['NL_PT1_mean'].to_numpy(float), t, ath,
                             maneuver=maneuver),
        a_transducer=an.a_transducer(result['NL_PT1_smooth'].to_numpy(float),
                                     result['NL_PT2_smooth'].to_numpy(float), t, ath),
        a_joukowsky=an.a_joukowsky(result['NL_PT1_smooth'].to_numpy(float),
                                   result['NL_Q_smooth'].to_numpy(float), t, ath),
    )
    cels = dict(Theoretical=cel['a_theoretical'], M1_period=cel['a_period'],
                M2_transducer=cel['a_transducer'], M3_Joukowsky=cel['a_joukowsky'])
    T_period = an.period_time(cels)      # which celerity: [processing] period_from
    # H* again with this T_period, so the end of the maneuver it is read at is
    # the same instant damping_NL uses below
    an.nondimensionalize(result, T_period)

    NL_pt1 = result['NL_PT1_smooth'].to_numpy(float)
    L_pt1  = result['L_PT1_smooth'].to_numpy(float)
    NL_nd  = result['NL_PT1_nondim'].to_numpy(float)
    L_nd   = result['L_PT1_nondim'].to_numpy(float)
    loc = an.locate(L_pt1, t, T_period, cels, n_leaks=len(leaks), NL_sig=NL_pt1)
    leak_errors_list = an.location_errors_all(loc, x_reals)
    leak_errors = leak_errors_list[0] if leak_errors_list else {}
    # validation only: no echo before the nearer location (reported apart, _ord)
    ordered_errors_list = an.order_constrained_errors(loc, x_reals, cels, t)

    # damping: queda da primeira sobrepressao medida contra o ensaio sem fuga
    fsd = an.damping_NL(L_pt1, NL_pt1, t, T_period, x_leak=x_reals, cels=cels)
    hl = (an.head_losses(fsd, va_info['pre_medians_loc'], lab_L,
                         [s for _, s in leaks], x_reals) if fsd else None)

    er = an.residual_energy(NL_nd, L_nd, NL_pt1, t, T_period)
    # alternative: the same energy with each curve on its OWN rise. NL_nd/L_nd
    # already carry the common normalisation, which is the one of eq. 5.11.
    _Ls = result['L_PT1_nondim_self'].to_numpy(float)
    _Ns = result['NL_PT1_nondim_self'].to_numpy(float)
    er_c = an.residual_energy(_Ns, _Ls, NL_pt1, t, T_period)

    t_on_common = fsd['t_on'] if fsd else None
    # On the dimensionless head H*: dividing every amplitude by a constant
    # shifts ln A but not its slope, so zeta is unchanged, while the decay of
    # the leak and of the reference become directly comparable on one plot.
    dc_L  = an.decay_rate(L_nd,  t, T_period, t_on=t_on_common, who=lab_L)
    dc_NL = an.decay_rate(NL_nd, t, T_period, t_on=t_on_common, who=lab_NL)
    ldc = None
    if dc_L and dc_NL:
        ldc = dict(zeta_total=dc_L['zeta'], zeta_NL=dc_NL['zeta'],
                   zeta_leak=dc_L['zeta'] - dc_NL['zeta'],
                   zeta_total_u=dc_L['zeta_u'], zeta_NL_u=dc_NL['zeta_u'],
                   zeta_leak_u=float(np.hypot(dc_L['zeta_u'], dc_NL['zeta_u'])),
                   r2_lnA_L=dc_L['r2'], r2_lnA_NL=dc_NL['r2'],
                   n_cycles_L=dc_L['n_cycles'], n_cycles_NL=dc_NL['n_cycles'],
                   # periods spanned vs peaks found: equal unless an
                   # oscillation escaped detection (see amplitude_cycles).
                   n_span_L=dc_L['n_span'], n_span_NL=dc_NL['n_span'])

    row = an.build_row(facility=facility, maneuver=maneuver, flow=flow,
                       loc_name=loc_name, leak=leak, lab_NL=lab_NL, lab_L=lab_L,
                       x_real=x_real, al=align_info, cel=cel, T_period=T_period,
                       va_info=va_info, loc=loc, leak_errors=leak_errors, er=er,
                       er_common=er_c,
                       hl=hl, ldc=ldc, leaks=leaks, x_reals=x_reals,
                       leak_errors_list=leak_errors_list,
                       ordered_errors_list=ordered_errors_list, fsd=fsd)

    if WRITE_EXPORTS:
        an.write_pair_exports(leak_dir=root, lab_L=lab_L, lab_NL=lab_NL,
                              va_data_clock=va_data_clock,
                              va_data_nl_clock=va_data_nl_clock, result=result,
                              loc_name=loc_name, leak=leak, x_real=x_real,
                              T_period=T_period, al=align_info, cel=cel, loc=loc,
                              leak_errors=leak_errors, hl=hl, er=er, ldc=ldc,
                              fsd=fsd,
                              leaks=leaks, x_reals=x_reals,
                              leak_errors_list=leak_errors_list)
    return row


def main():
    args = sys.argv[1:]
    fac = [a for a in args if not a.isdigit()]
    facility = fac[0] if fac else DEFAULT_FACILITY
    if facility not in FACILITY_DIRS:
        print(f'Unknown facility {facility!r}; options: {sorted(FACILITY_DIRS)}')
        return
    root = FACILITY_DIRS[facility]
    out_csv = os.path.join(root, 'pair_single.csv')
    an.configure_pipe(facility)
    only = {int(a) for a in args if a.isdigit()} or None
    print(f'Facility: {facility}   root: {root}   L={an.L_pipe:g} m, '
          f'a_teo={an.a_theoretical():.0f} m/s')

    plan = discover(root, only)
    if not plan:
        print(f'No leak files found under {root} (filter: {sorted(only) if only else "none"}).')
        return

    rows, failures, total, warned = [], [], 0, 0
    for man, flow, leaks, has_nl in plan:
        loc_name = '+'.join(l for l, _ in leaks)
        leak     = '+'.join(s for _, s in leaks)
        if not has_nl:
            print(f'[skip] {man}/{flow}/{loc_name}/L{leak}: no NL reference for this maneuver/flow')
            continue
        nl_test = NL_TEST_BY_FLOW.get(facility, {}).get(flow, NL_TEST_DEFAULT)
        for i, fname in enumerate(an.list_tests(root, man, flow, leaks=leaks), start=1):
            total += 1
            tag = f'{man}/{flow}/{loc_name}/L{leak} {os.path.splitext(fname)[0]}'
            buf = io.StringIO()
            try:
                if VERBOSE:
                    print(f'\n===== {tag} =====')
                    row = process_pair(root, facility, man, flow, leaks, nl_test, i)
                else:
                    with redirect_stdout(buf):
                        row = process_pair(root, facility, man, flow, leaks, nl_test, i)
                rows.append(row)
                e1 = row.get('x_err_M1_period_m')
                e2 = row.get('x_err_M1_period_m_leak2')
                p1 = f'{e1:.3f} m' if e1 is not None and np.isfinite(e1) else 'n/d'
                if len(leaks) > 1:
                    p2 = f'{e2:.3f} m' if e2 is not None and np.isfinite(e2) else 'n/d'
                    msg = f'x_err(M1) leak1 = {p1}   leak2 = {p2}'
                else:
                    msg = (f'x_err(M1) = {p1}' if p1 != 'n/d' else 'location not resolved')
                print(f'[ok]   {tag:<58s} {msg}')
                # A pair can succeed and still be telling us something is wrong
                # (a disconnected flow meter, a celerity that did not resolve).
                # Without this, those notes die inside the suppressed log and
                # the bad numbers reach the CSV unannounced.
                if not VERBOSE:
                    # lstrip: the pipeline indents some of its notes.
                    notes = [l.strip() for l in buf.getvalue().splitlines()
                             if l.lstrip().startswith(WARN_PREFIXES)
                             or 'NOT RESOLVED' in l]
                    warned += len(notes)
                    for l in dict.fromkeys(notes):
                        print(f'       ! {l}')
            except Exception as exc:
                failures.append((tag, exc))
                print(f'[FAIL] {tag:<58s} {type(exc).__name__}: {exc}')
                if VERBOSE:
                    traceback.print_exc()
                else:
                    log = buf.getvalue().rstrip()
                    if log:
                        print('       last pipeline log lines:')
                        print('\n'.join(f'       | {l}' for l in log.splitlines()[-10:]))

    print('\n' + '=' * 74)
    print(f'BATCH DONE: {len(rows)}/{total} pair(s) ok, {len(failures)} failed'
          + (f', {warned} warning(s) raised on pairs that otherwise succeeded.'
             if warned else '.'))
    for tag, exc in failures:
        print(f'  FAILED  {tag}: {type(exc).__name__}: {exc}')

    if only:
        print('\n[filtered run] result CSV left untouched - run without a flow '
              'filter to regenerate it.')
    elif rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'\nWrote {out_csv}  ({len(rows)} pairs, rebuilt from scratch)')

    if rows:
        df = pd.DataFrame(rows)
        err_cols = [c for c in df.columns
                    if c.startswith('x_err_') and c.endswith('_m') and c != 'x_err_period_m']
        show = (['maneuver', 'flow', 'loc', 'leak', 'leak_test'] + err_cols
                + [c for c in ('E_R_over_E0', 'ratio_Q', 'zeta_leak') if c in df.columns])
        show = [c for c in show if c in df.columns]
        print('\nLOCALISATION ERRORS AND QUANTIFICATION METRICS PER PAIR:')
        with pd.option_context('display.width', 240, 'display.float_format', '{:.4f}'.format):
            print(df[show].sort_values([c for c in ('maneuver', 'flow', 'loc', 'leak', 'leak_test')
                                        if c in df.columns]).to_string(index=False))
        if err_cols:
            print('\nMEAN ABSOLUTE LOCALISATION ERROR PER METHOD (m):')
            print(df[err_cols].mean().to_string(float_format='{:.3f}'.format))
        ord_cols = [c for c in df.columns if c.startswith('x_err_') and c.endswith('_m_ord')]
        if ord_cols:
            print('\nVALIDATION ONLY - no echo before the nearer location, '
                  'mean absolute error (m):')
            print(df[ord_cols].mean().to_string(float_format='{:.3f}'.format))
            if 'echo_src_ord' in df.columns:
                print('echo source (leak 1):', df['echo_src_ord'].value_counts().to_dict())


if __name__ == '__main__':
    main()
