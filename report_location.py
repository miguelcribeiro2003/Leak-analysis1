# -*- coding: utf-8 -*-
"""Leak-location report: estimates, errors and how they average per combination.

    python report_location.py CS      # or CC, or SIM

Reads pair_single.csv (batch_analysis.py for CS/CC, batch_simulated.py for SIM)
and reports, for every celerity estimate:

    1. by test         one row per test and per leak, x_est, error (m) and (%L)
    2. by combination  mean over the tests of a combination, with the scatter
    3. summary         bias, MAE, RMSE, MAPE and NRMSE over the combinations,
                       split into inside / outside the blind zone of Section 5.4

The ordering criterion (Section 5.4) uses the known leak positions, so it is
NOT blind detection: its columns carry the _ord suffix and are reported apart.

Writes location_by_test.csv, location_by_combination.csv and
location_summary.csv next to the data.
"""
import sys

import numpy as np
import pandas as pd

import analysis as an
import report_common as rc


def by_test(leaks, cels):
    cols = (['maneuver', 'flow', 'loc', 'leak', 'test', 'leak_no', 'leak_loc',
             'leak_size', 'x_real', 'in_domain', 't1_ms', 't2_ms', 'delta_t_ms']
            + [f'{p}_{c}{s}' for c in cels
               for p, s in (('x', ''), ('err', '_m'), ('err', '_pctL'))]
            + ['x_Theoretical_ord', 'err_Theoretical_ord_m', 'echo_src_ord'])
    return leaks[rc.present(leaks, cols)].sort_values(
        ['maneuver', 'flow', 'leak_loc', 'leak_size', 'test'])


def by_combination(leaks, cels):
    out = []
    for keys, g in leaks.groupby(rc.COMBO_COLS + ['leak_no', 'leak_loc',
                                                  'leak_size'], dropna=False):
        row = dict(zip(rc.COMBO_COLS + ['leak_no', 'leak_loc', 'leak_size'], keys))
        row.update(n_tests=len(g), x_real=g['x_real'].iloc[0],
                   in_domain=bool(g['in_domain'].all()),
                   t1_ms=g['t1_ms'].mean(), t2_ms=g['t2_ms'].mean(),
                   delta_t_ms=g['delta_t_ms'].mean())
        for cel in cels:
            x = g[f'x_{cel}']
            row[f'x_{cel}'] = x.mean()
            # scatter between the tests of the combination (sample sd)
            row[f'x_{cel}_sd'] = x.std(ddof=1) if len(x) > 1 else np.nan
            row[f'err_{cel}_m'] = x.mean() - row['x_real']
            row[f'err_{cel}_pctL'] = (x.mean() - row['x_real']) / an.L_pipe * 100
            # o mesmo erro em valor absoluto, a convencao das tabelas da tese
            # (eps = |x_est - x_real| / L * 100); as colunas com sinal ficam ao
            # lado, porque e nelas que se ve o enviesamento de cada localizacao
            row[f'err_{cel}_abs_pctL'] = abs(row[f'err_{cel}_pctL'])
            xo = g[f'x_{cel}_ord']
            row[f'x_{cel}_ord'] = xo.mean()
            row[f'err_{cel}_ord_m'] = xo.mean() - row['x_real']
            row[f'err_{cel}_ord_pctL'] = (xo.mean() - row['x_real']) / an.L_pipe * 100
            row[f'err_{cel}_ord_abs_pctL'] = abs(row[f'err_{cel}_ord_pctL'])
        out.append(row)
    return pd.DataFrame(out).sort_values(rc.COMBO_COLS + ['leak_no'])


def _metrics(err, x_real):
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if not len(e):
        return None
    xr = np.asarray(x_real, float)[:len(e)]
    return dict(n=len(e), bias_m=e.mean(), MAE_m=np.abs(e).mean(),
                RMSE_m=float(np.sqrt((e ** 2).mean())),
                MAPE_pct=float(np.mean(np.abs(e) / np.abs(xr)) * 100),
                MAE_pctL=float(np.abs(e).mean() / an.L_pipe * 100))


def summary(combos, cels):
    rows = []
    subsets = [('all', combos), ('in domain', combos[combos.in_domain]),
               ('outside', combos[~combos.in_domain])]
    for cel in cels:
        for tag, sub in subsets:
            for rule, col in (('blind', f'err_{cel}_m'),
                              ('ordering', f'err_{cel}_ord_m')):
                if not len(sub) or col not in sub:
                    continue
                m = _metrics(sub[col], sub['x_real'])
                if m:
                    rows.append(dict(celerity=cel, subset=tag, rule=rule, **m))
    return pd.DataFrame(rows)


def main():
    key, _ = rc.pick(sys.argv[1] if len(sys.argv) > 1 else 'CS')
    rc.header(key)
    leaks = rc.load_leaks(key)
    cels = rc.celerities(leaks)
    print(f'  celerity estimates available: {", ".join(cels)}')

    t1 = by_test(leaks, cels)
    rc.print_table(t1.round(3), '1. BY TEST   (x in m; error in m and in % of L)')
    rc.save(t1, key, 'location_by_test.csv')

    t2 = by_combination(leaks, cels)
    show = rc.present(t2, rc.COMBO_COLS + ['leak_loc', 'leak_size', 'n_tests',
                                           'x_real', 'in_domain']
                      + [f'x_{c}' for c in cels]
                      + ['x_Theoretical_sd', 'err_Theoretical_m',
                         'err_Theoretical_pctL', 'err_Theoretical_abs_pctL',
                         'err_Theoretical_ord_m'])
    rc.print_table(t2[show].round(3),
                   '2. BY COMBINATION   (mean over the tests; sd = scatter between them)')
    rc.save(t2, key, 'location_by_combination.csv')

    t3 = summary(t2, cels)
    rc.print_table(t3.round(3),
                   '3. SUMMARY over the combinations   (blind vs ordering criterion)')
    rc.save(t3, key, 'location_summary.csv')

    n_out = int((~t2.in_domain).sum())
    if n_out:
        print(f'\n  {n_out} combination(s) outside the blind zone limit '
              f'x_PT1 + a (t_c - t1) / 2; listed as "outside" above.')


if __name__ == '__main__':
    main()
