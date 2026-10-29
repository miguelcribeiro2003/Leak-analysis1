# -*- coding: utf-8 -*-
"""Leak-sizing indicators: values per test and per combination.

    python report_indicators.py CS     # or CC, or SIM

There is no true value to compare against here, unlike the location, so the
report gives the values themselves and their scatter:

    1. by test         one row per test: the three indicators side by side
    2. by combination  mean +/- sd over the tests of each combination

Indicators, in the order of Chapter 5:

    dH*        initial pressure drop (Section 5.5): each curve normalised by
               its own head rise at the end of the manoeuvre, then the largest
               separation between them over the no-leak Joukowsky rise
               (dH_over_dHJ, the main indicator). Every indicator is built
               on ONE normalisation, applied up front in an.nondimensionalize:
               each signal referred to its own H0, both divided by the head
               rise of the NO-LEAK test (eq. 5.11). ratio_H_self and
               E_R_over_E0_self are the alternatives, with each curve on its
               own rise. ratio_H_self, the same
               separation with a single normalisation, comes alongside as a
               diagnostic.
    E_R/E0     residual signal energy (Section 5.6)
    zeta_leak  pressure damping over time (Section 5.7)

QL/Q0 is reported where the flow channel is usable (it is not on the coil) and,
for the numerical model, it is the true value from the orifice equation.

Two flags mark what should not be pooled blindly: 'cavitation' for the flows
with column separation, and 'dH_valid' for the validity domain of dH*, which
requires the leak reflection to return after the end of the manoeuvre,
x - x_PT1 > a t_c / 2 (Section 5.5). Nothing is dropped: the flags are columns.

Writes indicators_by_test.csv and indicators_by_combination.csv.
"""
import sys

import numpy as np
import pandas as pd

import analysis as an
import report_common as rc

# indicator -> label used in the printed tables
INDICATORS = {'dH_over_dHJ': 'dH_d', 'ratio_H_self': 'dH*', 'E_R_over_E0': 'E_R/E0',
              'zeta_leak': 'zeta_leak', 'ratio_Q': 'QL/Q0'}


def _dH_valid(df):
    """Validity domain of dH* (Section 5.5): x - x_PT1 > a t_c / 2.

    Unlike the blind zone of the location, t1 is NOT subtracted here: no part
    of the reflection may arrive before the end of the manoeuvre, since the
    head rise up to that instant is what normalises both signals.
    """
    cel_col = {'period': 'a_period', 'transducer': 'a_transducer'}.get(
        an.PERIOD_FROM, 'a_period')
    a = df[cel_col] if cel_col in df.columns else df['a_theoretical']
    a = a.where(np.isfinite(a), df['a_theoretical'])
    limit = an.PT1_OFFSET + a * df['t_fim_manobra_ms'] / 2e3
    # the nearest leak of the case is the one that has to clear the limit
    nearest = df.apply(
        lambda r: min(r.get(f'x_real{"" if j == 0 else f"_leak{j + 1}"}', np.nan)
                      for j in range(int(r['n_leaks']))), axis=1)
    return nearest > limit, limit


def by_test(cases):
    ok, limit = _dH_valid(cases)
    out = cases.copy()
    out['dH_limit_m'] = limit
    out['dH_valid'] = ok
    if 'dH_over_dHJ' not in out.columns:          # older tables
        out['dH_over_dHJ'] = out['dH_m'] / out['dHJ_NL']
    # instants behind the indicators, all in ms after the onset: the end of the
    # manoeuvre (t_c, where H_e is read) and the instant of the largest
    # separation between the curves, which is where dH is measured
    if 'dt_drop_ms' in out.columns:
        out['t_drop_ms'] = out['t_fim_manobra_ms'] + out['dt_drop_ms']
    cols = rc.present(out, ['maneuver', 'flow', 'loc', 'leak', 'test', 'n_leaks',
                            'cavitation', 'dH_valid', 't_fim_manobra_ms',
                            't_drop_ms', 'T_period_ms', 'dH_over_dHJ',
                            'ratio_H_self', 'dH_m', 'dHJ_NL', 'E_R_over_E0',
                            'E_R_over_E0_self', 'zeta_leak',
                            'zeta_leak_u', 'n_cycles_L', 'r2_lnA_L', 'ratio_Q'])
    return out[cols].sort_values(['maneuver', 'flow', 'loc', 'leak', 'test'])


def by_combination(tests):
    quantities = [c for c in ['dH_over_dHJ', 'ratio_H_self', 'dH_m', 'E_R_over_E0',
                              'E_R_over_E0_self',
                              'zeta_leak', 'ratio_Q', 't_fim_manobra_ms',
                              't_drop_ms', 'n_cycles_L'] if c in tests.columns]
    out = []
    for keys, g in tests.groupby(rc.COMBO_COLS, dropna=False):
        row = dict(zip(rc.COMBO_COLS, keys))
        row.update(n_tests=len(g), n_leaks=int(g['n_leaks'].iloc[0]),
                   cavitation=bool(g['cavitation'].iloc[0]),
                   dH_valid=bool(g['dH_valid'].all()))
        for q in quantities:
            v = g[q]
            row[q] = v.mean()
            row[f'{q}_sd'] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[f'{q}_cv_pct'] = (row[f'{q}_sd'] / abs(row[q]) * 100
                                  if row[q] else np.nan)
        out.append(row)
    return pd.DataFrame(out).sort_values(rc.COMBO_COLS)


def main():
    key, _ = rc.pick(sys.argv[1] if len(sys.argv) > 1 else 'CS')
    rc.header(key)
    cases = rc.load_cases(key)

    t1 = by_test(cases)
    rc.print_table(t1.round(4), '1. BY TEST', floatfmt='{:.4f}')
    rc.save(t1, key, 'indicators_by_test.csv')

    t2 = by_combination(t1)
    show = rc.present(t2, rc.COMBO_COLS + ['n_tests', 'n_leaks', 'cavitation',
                                           'dH_valid']
                      + [c for q in INDICATORS for c in (q, f'{q}_sd', f'{q}_cv_pct')])
    rc.print_table(t2[show].round(4),
                   '2. BY COMBINATION   (mean, sd between tests, CV in %)',
                   floatfmt='{:.4f}')
    rc.save(t2, key, 'indicators_by_combination.csv')

    bad = t2[~t2.dH_valid]
    if len(bad):
        print(f'\n  dH* outside its validity domain in {len(bad)} combination(s): '
              + ', '.join(f'{r.loc}/{r.leak}' for r in bad.itertuples()))
    cav = t2[t2.cavitation]
    if len(cav):
        print(f'  cavitation flows present in {len(cav)} combination(s): '
              'do not pool them with the rest.')


if __name__ == '__main__':
    main()
