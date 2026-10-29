# -*- coding: utf-8 -*-
"""Shared plumbing for the report_*.py scripts.

The three reports (location, indicators, repeatability) read the same tables and
speak the same language, so everything they have in common lives here:

    DATASETS      CS / CC / SIM -> folder holding pair_single.csv
    load_cases    one row per analysed pair, as written by the batch
    load_leaks    one row per LEAK: the two-leak cases become two rows, with the
                  location columns of that leak under plain names
    print_table   aligned table on screen
    save          csv next to the data, with a one-line confirmation

Nothing is recomputed here: batch_analysis.py (CS, CC) and batch_simulated.py
(SIM) are the only places where signals are processed.
"""
import os

import numpy as np
import pandas as pd

import analysis as an

DATASETS = {
    'CS':  dict(folder='Experimental_data_CS', pipe='CS',
                label='copper straight, experimental'),
    'CC':  dict(folder='Experimental_data_CC', pipe='CC',
                label='copper coil, experimental'),
    'SIM': dict(folder='Numerical_data_CS', pipe='CS_sim',
                label='copper straight, numerical model'),
}

# celerity estimates, in the order they are reported
CELERITIES = ['Theoretical', 'M1_period', 'M2_transducer', 'M3_Joukowsky']

# what identifies a test and what identifies a combination of test conditions
ID_COLS = ['maneuver', 'flow', 'loc', 'leak', 'leak_test']
COMBO_COLS = ['maneuver', 'flow', 'loc', 'leak']

CAVITATION_FLOWS = [850]


def pick(name):
    """Validate the dataset argument and configure the pipe it belongs to."""
    key = (name or 'CS').upper()
    if key not in DATASETS:
        raise SystemExit(f'Unknown dataset {name!r}. Use one of: '
                         + ', '.join(DATASETS))
    cfg = DATASETS[key]
    an.configure_pipe(cfg['pipe'])
    return key, cfg


def load_cases(key):
    """pair_single.csv of that dataset, one row per analysed pair."""
    cfg = DATASETS[key]
    path = os.path.join(cfg['folder'], 'pair_single.csv')
    if not os.path.exists(path):
        raise SystemExit(f'{path} does not exist: run '
                         + ('batch_simulated.py' if key == 'SIM'
                            else f'batch_analysis.py {key}') + ' first.')
    df = pd.read_csv(path, dtype={'loc': str, 'leak': str})
    df['n_leaks'] = df['loc'].astype(str).str.count(r'\+') + 1
    df['cavitation'] = df['flow'].isin(CAVITATION_FLOWS)
    if 'leak_test' not in df.columns:
        df['leak_test'] = ''
    df['test'] = (df['leak_test'].astype(str).str.extract(r'test0*(\d+)')[0]
                  .fillna('1'))
    return df


def _blind_limit(df):
    """x_PT1 + a (t_c - t1) / 2, the blind zone of Section 5.4 (m)."""
    if 'x_min_domain' in df.columns:
        return df['x_min_domain']
    cel_col = {'period': 'a_period', 'transducer': 'a_transducer'}.get(
        an.PERIOD_FROM, 'a_period')
    a = df[cel_col] if cel_col in df.columns else df['a_theoretical']
    a = a.where(np.isfinite(a), df['a_theoretical'])
    return an.PT1_OFFSET + a * (df['t_fim_manobra_ms'] - df['t1_ms']) / 2e3


def load_leaks(key):
    """One row per LEAK: a two-leak case yields two rows.

    The location columns of each leak lose the _leakK suffix, so every report
    can use the same names whatever the number of leaks in the test.
    """
    df = load_cases(key)
    if 'x_min_domain' not in df.columns:      # SIM already carries it
        df = pd.concat([df, _blind_limit(df).rename('x_min_domain')], axis=1)
    rows = []
    for _, r in df.iterrows():
        locs = str(r['loc']).split('+')
        sizes = str(r['leak']).split('+')
        for j, (lname, size) in enumerate(zip(locs, sizes)):
            sfx = '' if j == 0 else f'_leak{j + 1}'
            row = {c: r[c] for c in ID_COLS + ['facility', 'test', 'n_leaks',
                                               'cavitation', 'x_min_domain']
                   if c in r.index}
            row.update(leak_no=j + 1, leak_loc=lname, leak_size=size,
                       x_real=r.get(f'x_real{sfx}', np.nan))
            row['in_domain'] = bool(row['x_real'] > row['x_min_domain'])
            for cel in CELERITIES:
                x = r.get(f'x_{cel}{sfx}', np.nan)
                xo = r.get(f'x_{cel}_ord{sfx}', np.nan)
                row[f'x_{cel}'] = x
                row[f'err_{cel}_m'] = x - row['x_real']
                row[f'err_{cel}_pctL'] = (x - row['x_real']) / an.L_pipe * 100
                row[f'x_{cel}_ord'] = xo
                row[f'err_{cel}_ord_m'] = xo - row['x_real']
                row[f'err_{cel}_ord_pctL'] = (xo - row['x_real']) / an.L_pipe * 100
            row['echo_src_ord'] = r.get(f'echo_src_ord{sfx}')
            # timing of the two instants the location is read from, both counted
            # from the manoeuvre onset: t1 is the steepest point of the front
            # (one per test) and t2 the drop produced by this leak's reflection
            t1 = r.get('t1_ms', np.nan)
            dt = r.get(f'delta_t_s{sfx}', np.nan) * 1e3
            dt_ord = r.get(f'delta_t_s_ord{sfx}', np.nan) * 1e3
            row['t1_ms'] = t1
            row['delta_t_ms'] = dt
            row['t2_ms'] = t1 + dt
            row['delta_t_ord_ms'] = dt_ord
            row['t2_ord_ms'] = t1 + dt_ord
            rows.append(row)
    return pd.DataFrame(rows)


def celerities(df):
    """The celerity estimates this dataset actually resolved (M2 is n/a on the
    straight pipe, M3 wherever the flow meter is unreliable)."""
    out = []
    for cel in CELERITIES:
        col = f'x_{cel}'
        if col in df.columns and np.isfinite(df[col]).any():
            out.append(cel)
    return out


def present(df, cols):
    """The requested columns that this dataset actually has."""
    return [c for c in cols if c in df.columns]


def fmt_pm(mean, sd, dec=3):
    """'7.869 +/- 0.305', or just the mean when there is no scatter."""
    if mean is None or not np.isfinite(mean):
        return 'n/a'
    if sd is None or not np.isfinite(sd) or sd == 0:
        return f'{mean:.{dec}f}'
    return f'{mean:.{dec}f} ± {sd:.{dec}f}'


def cross(df, value, index='leak_size', columns='leak_loc', dec=3, sd=None):
    """Cross table: one variable in the rows, another in the columns.

    `value` is the column to show and `sd` an optional column with its scatter,
    which is then written in the same cell as 'mean +/- sd'.
    """
    piv = df.pivot_table(index=index, columns=columns, values=value,
                         aggfunc='mean')
    if sd is None:
        return piv.round(dec)
    err = df.pivot_table(index=index, columns=columns, values=sd, aggfunc='mean')
    out = piv.copy().astype(object)
    for i in piv.index:
        for c in piv.columns:
            out.loc[i, c] = fmt_pm(piv.loc[i, c], err.loc[i, c], dec)
    return out


def print_table(df, title=None, floatfmt='{:.3f}', width=250):
    if title:
        print(f'\n{title}')
        print('-' * min(width, max(40, len(title))))
    if not len(df):
        print('  (no rows)')
        return
    with pd.option_context('display.width', width, 'display.max_columns', 100,
                           'display.float_format', floatfmt.format):
        print(df.to_string(index=False))


def save(df, key, name):
    path = os.path.join(DATASETS[key]['folder'], name)
    df.to_csv(path, index=False)
    print(f'  -> {path}  ({len(df)} rows)')


def header(key):
    cfg = DATASETS[key]
    print('=' * 78)
    print(f'  {key}: {cfg["label"]}   (L = {an.L_pipe:g} m, '
          f'x_PT1 = {an.PT1_OFFSET:g} m, a_theo = {an.a_theoretical():.1f} m/s)')
    print('=' * 78)
