# -*- coding: utf-8 -*-
"""Test-to-test repeatability: how far the individual tests sit from the mean.

    python report_repeatability.py CS      # or CC

For every combination of test conditions (manoeuvre, flow, location, leak size)
and every quantity, the scatter of the repetitions about their own mean:

    mean, sd        sample mean and standard deviation (n - 1)
    MAE, RMSE       mean |x_i - mean| and its quadratic counterpart
    CV              sd / |mean|, in %, the dimensionless measure of scatter

Quantities covered: the leak position estimated with each celerity, the two
instants it is read from (t1 and t2), the three sizing indicators (dH*,
E_R/E0 and zeta_leak), the end of the manoeuvre and the instant of the largest
separation, plus QL/Q0 where the flow channel is usable.

    1. by quantity and combination   the detail
    2. median across combinations    one number per quantity, for the text

Combinations with a single test carry no information about repeatability and
are reported with n = 1 and empty scatter. The numerical dataset (SIM) has one
run per combination by construction, so the script says so and stops.

Writes repeatability_by_combination.csv and repeatability_summary.csv.
"""
import sys

import numpy as np
import pandas as pd

import report_common as rc


def _scatter(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    mean = float(v.mean())
    if len(v) == 1:
        return dict(n=1, mean=mean, sd=np.nan, MAE=np.nan, RMSE=np.nan,
                    CV_pct=np.nan)
    dev = v - mean
    return dict(n=len(v), mean=mean, sd=float(v.std(ddof=1)),
                MAE=float(np.abs(dev).mean()),
                RMSE=float(np.sqrt((dev ** 2).mean())),
                CV_pct=float(v.std(ddof=1) / abs(mean) * 100) if mean else np.nan)


def location_rows(key, cels):
    """One row per combination, leak and celerity, plus the two instants."""
    leaks = rc.load_leaks(key)
    rows = []
    for keys, g in leaks.groupby(rc.COMBO_COLS + ['leak_no', 'leak_loc'],
                                 dropna=False):
        base = dict(zip(rc.COMBO_COLS + ['leak_no', 'leak_loc'], keys))
        for cel in cels:
            s = _scatter(g[f'x_{cel}'])
            if s:
                rows.append(dict(quantity=f'x_{cel} (m)', **base, **s))
        # the instants the location is read from: how repeatable they are is
        # what makes the position repeatable
        for q, lab in (('t1_ms', 't1 (ms)'), ('t2_ms', 't2 (ms)'),
                       ('delta_t_ms', 't2-t1 (ms)')):
            if q in g.columns:
                s = _scatter(g[q])
                if s:
                    rows.append(dict(quantity=lab, **base, **s))
    return rows


def indicator_rows(key):
    cases = rc.load_cases(key)
    if 'dt_drop_ms' in cases.columns:
        cases['t_drop_ms'] = cases['t_fim_manobra_ms'] + cases['dt_drop_ms']
    quantities = [c for c in ['dH_over_dHJ', 'ratio_H_self', 'dH_m', 'E_R_over_E0',
                              'E_R_over_E0_self',
                              'zeta_leak', 'ratio_Q', 't_fim_manobra_ms',
                              't_drop_ms'] if c in cases.columns]
    rows = []
    for keys, g in cases.groupby(rc.COMBO_COLS, dropna=False):
        base = dict(zip(rc.COMBO_COLS, keys))
        base.update(leak_no=np.nan, leak_loc='')
        for q in quantities:
            s = _scatter(g[q])
            if s:
                rows.append(dict(quantity=q, **base, **s))
    return rows


def main():
    key, _ = rc.pick(sys.argv[1] if len(sys.argv) > 1 else 'CS')
    rc.header(key)
    if key == 'SIM':
        raise SystemExit('  The numerical model has one run per combination, so '
                         'there is no test-to-test scatter to report.\n'
                         '  Use report_indicators.py SIM for the values themselves.')

    leaks = rc.load_leaks(key)
    cels = rc.celerities(leaks)
    df = pd.DataFrame(location_rows(key, cels) + indicator_rows(key))
    cols = ['quantity'] + rc.COMBO_COLS + ['leak_no', 'leak_loc', 'n', 'mean',
                                           'sd', 'MAE', 'RMSE', 'CV_pct']
    df = df[rc.present(df, cols)].sort_values(['quantity'] + rc.COMBO_COLS)
    rc.print_table(df.round(4), '1. BY QUANTITY AND COMBINATION',
                   floatfmt='{:.4f}')
    rc.save(df, key, 'repeatability_by_combination.csv')

    rep = df[df['n'] > 1]
    summary = (rep.groupby('quantity')
               .agg(combinations=('n', 'size'), tests_per_combination=('n', 'median'),
                    median_sd=('sd', 'median'), median_MAE=('MAE', 'median'),
                    median_CV_pct=('CV_pct', 'median'),
                    worst_CV_pct=('CV_pct', 'max'))
               .reset_index())
    rc.print_table(summary.round(4),
                   '2. MEDIAN ACROSS COMBINATIONS   (only those with more than one test)',
                   floatfmt='{:.4f}')
    rc.save(summary, key, 'repeatability_summary.csv')

    single = df[df['n'] == 1]['quantity'].count()
    if single:
        print(f'\n  {single} row(s) come from combinations with a single test: '
              'no scatter can be measured there.')


if __name__ == '__main__':
    main()
