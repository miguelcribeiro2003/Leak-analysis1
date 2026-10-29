# -*- coding: utf-8 -*-
"""Calibration of the sizing indicators: QL/Q0 = a * indicator.

The calibration is fitted on the SIMULATED dataset (Numerical_data_CS) and then
applied to the EXPERIMENTAL one (Experimental_data_CS), which is the workflow of
Section 5.8: the numerical model provides the relation between each indicator and
the leak flow, and the rig provides the indicator.

    python calibration_coeffs.py

Two families of coefficients are produced for each indicator:

  - per LOC  : one slope for each leak position. Uses the position, which the
               location step of Chapter 5 already gives, so it is available in
               practice.
  - global   : a single slope for the whole facility, for when the position is
               not known. Reported twice, with and without LOC1, because LOC1 is
               inside the blind zone and its slope is not comparable with the
               others (see BLIND_ZONE_NOTE).

The fit is through the origin (no leak means no signal), by least squares:
a = sum(x*y) / sum(x*x), and the quality of the fit is reported as
R2 = 1 - SSE/SST, with SSE = sum((y - a*x)^2) of the fitted line and
SST = sum((y - ybar)^2). It is the variance the LINE explains, so it is directly
comparable with the other regression forms tried in regressoes_sizing.py.

Flows
-----
Simulated: Q0 and QL come from the model's own steady state (rede_rp/rp_*.dat),
read into batch_simulated.py; Q0 = 246.22 L/h at the valve with no leak.
Experimental: QL was measured by weighing and Q0 = 240.76 L/h is the flow meter
reading of the no-leak test (the nominal set point was 250 L/h, but the meter
reads 240.76 with no leak, and the leak does not change it: 240.5 L/h on average
over the twelve leak tests).
"""
import os

import numpy as np
import pandas as pd

SIM_CSV = os.path.join('Numerical_data_CS', 'pair_single.csv')
EXP_CSV = os.path.join('Experimental_data_CS', 'pair_single.csv')
OUT_COEFFS = 'calibration_coefficients.csv'
OUT_VALID = 'calibration_validation.csv'

# As tres metodologias de sizing do Capitulo 5, na forma escolhida em
# regressoes_sizing.py (R2 = 1 - SSE/SST nos simulados, erro na bancada ao lado):
#
#   dH_over_dHJ  linear. E a queda inicial dividida pela subida de Joukowsky do
#                ensaio SEM FUGA (o dH_d/dH_j da Pauseiro).
#   E_R_over_E0  RAIZ de expoente fixo 0.5. A forma linear fica na lista ao lado
#                da raiz, nao como metodologia a mais mas para a tabela poder
#                mostrar o que a raiz corrige (R2 global 0.11 -> 0.66).
#   zeta_leak    linear.
#
# ratio_H_self -- a mesma queda normalizada pela subida de CADA curva -- saiu:
# analysis.py continua a medi-lo e os relatorios continuam a mostra-lo, mas nao
# e uma das metodologias de sizing e nao se calibra aqui.
INDICATORS = ['dH_over_dHJ', 'E_R_over_E0', 'E_R_over_E0_sqrt', 'zeta_leak']
LOCS = ['LOC1', 'LOC2', 'LOC3']

# Indicadores derivados: coluna nova = transformacao de uma coluna existente.
# E_R/E0 e um quociente de integrais QUADRATICOS no residuo R(t) = H*_NL - H*_L.
# Se o efeito da fuga escala linearmente com QL, entao R e proporcional a QL e
# E_R a QL^2, pelo que QL/Q0 e proporcional a RAIZ de E_R/E0 -- expoente 0.5
# previsto pela definicao, nao ajustado. Ajustar o expoente livremente da 0.69 a
# 0.79 e um R2 melhor nos simulados, mas 17.8% de erro na bancada contra os 8.9%
# da raiz fixa: com n = 4 pontos, estimar o expoente e ajustar ruido.
# A forma linear fica na mesma, lado a lado, para as duas poderem ser comparadas.
DERIVED = {'E_R_over_E0_sqrt': ('E_R_over_E0', np.sqrt)}

# how each indicator is written in the thesis, for the equation column
SYMBOL = {'dH_over_dHJ': 'dH_d', 'E_R_over_E0': 'E_R/E0',
          'E_R_over_E0_sqrt': 'sqrt(E_R/E0)', 'zeta_leak': 'zeta_leak'}

# leak position in the model, distance to the measuring section PT1 (m)
X_LEAK = {'LOC1': 2.000, 'LOC2': 7.625, 'LOC3': 10.250}

CAVITATION_FLOWS = [850]   # column separation invalidates every indicator
FLOW = 250                 # the only flow with a complete matrix of cases
MANEUVER = 'AUT'

# Caudal de fuga medido por pesagem (L/h), lido da folha de caracterizacao dos
# orificios (coluna QL), e o caudal sem fuga da bancada. Orificios 0.5B, 1.0A,
# 1.5A e 2.0B, os que foram usados nos ensaios de deteccao.
Q0_EXP = 240.76
QL_EXP = {
    'LOC1': {'0.5': 18.10, '1.0': 78.45, '1.5': 131.45, '2.0': 222.30},
    'LOC2': {'0.5': 19.31, '1.0': 76.01, '1.5': 132.56, '2.0': 223.01},
    'LOC3': {'0.5': 18.44, '1.0': 79.61, '1.5': 130.24, '2.0': 209.62},
}
# Os outros orificios da mesma folha, nao usados nos ensaios mas uteis para
# perceber a dispersao entre exemplares do mesmo calibre nominal:
#   0.5A   8.96 /  9.73 / 12.35      1.0B  74.87 / 80.58 / 71.02
#   1.5B 146.39 / 156.99 / 151.40    2.0A 260.68 / 263.50 / 265.46

BLIND_ZONE_NOTE = """LOC1 sits inside the blind zone: x_leak - x_PT1 = 1.815 m is
smaller than a*t_c/2, so the leak reflection arrives before the manoeuvre ends.
Its ratio_H_self collapses (0.0020 to 0.0150 in the model) while the leak flow is
the same as at the other positions, which is why its slope comes out an order of
magnitude larger. It is reported but must not be used, and it is what makes the
global fit over all three positions meaningless."""


def pearson_r2(x, y):
    """Squared Pearson correlation coefficient, eq. (5.3) of the thesis.

        R2 = [ sum((xi - xbar)(yi - ybar)) /
               (sqrt(sum((xi - xbar)^2)) * sqrt(sum((yi - ybar)^2))) ]^2

    It measures the linear association between the two variables and does not
    depend on the fitted line, so it is the same whether or not the regression
    is forced through the origin. Kept as r2_pearson, for the record; the R2 in
    the tables is the 1 - SSE/SST of fit_origin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def fit_origin(x, y):
    """Least-squares slope through the origin, plus R2 against the mean of y.

    The R2 returned here is the fraction of variance the FITTED LINE explains,
    1 - sum((y - a*x)^2) / sum((y - ybar)^2). It is the R2 reported in the
    tables, and is lower than pearson_r2 because the line is constrained
    through the origin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or not np.any(x):
        return np.nan, np.nan, len(x)
    a = float((x @ y) / (x @ x))
    resid = y - a * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return a, r2, len(x)


def load_single(path):
    """Single-leak cases of one dataset, at the calibration flow."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist: run the batch first')
    df = pd.read_csv(path, dtype={'loc': str, 'leak': str})
    df = df[~df['loc'].str.contains(r'\+')]
    df = df[~df['flow'].isin(CAVITATION_FLOWS)]
    df = df[df['flow'] == FLOW]
    if 'maneuver' in df.columns and (df['maneuver'] == MANEUVER).any():
        df = df[df['maneuver'] == MANEUVER]
    df = df.reset_index(drop=True)
    novas = {novo: fn(df[base].clip(lower=0))
             for novo, (base, fn) in DERIVED.items() if base in df.columns}
    if novas:   # de uma vez, senao o pandas avisa que o frame fica fragmentado
        df = pd.concat([df, pd.DataFrame(novas, index=df.index)], axis=1)
    return df


def calibrate(sim=None):
    """Fit QL/Q0 = a * indicator on the simulated data, per LOC and global."""
    sim = load_single(SIM_CSV) if sim is None else sim
    rows = []
    for ind in INDICATORS:
        if ind not in sim.columns:
            continue
        for loc in LOCS:
            g = sim[sim['loc'] == loc]
            a, r2, n = fit_origin(g[ind], g['ratio_Q'])
            rows.append(_row(ind, loc, a, r2, n,
                             pearson_r2(g[ind], g['ratio_Q'])))
        a, r2, n = fit_origin(sim[ind], sim['ratio_Q'])
        rows.append(_row(ind, 'GLOBAL', a, r2, n,
                         pearson_r2(sim[ind], sim['ratio_Q'])))
        v = sim[sim['loc'] != 'LOC1']
        a, r2, n = fit_origin(v[ind], v['ratio_Q'])
        rows.append(_row(ind, 'GLOBAL_no_LOC1', a, r2, n,
                         pearson_r2(v[ind], v['ratio_Q'])))
    return pd.DataFrame(rows)


def _row(indicator, group, a, r2, n, r2p=np.nan):
    """One coefficient row, with the fitted equation written out.

    r2 is 1 - SSE/SST of the through-origin line, which is what the tables
    report; r2_pearson is the squared Pearson coefficient of eq. (5.3), kept
    alongside for the record.
    """
    return dict(indicator=indicator, group=group, slope=a, r2=r2,
                r2_pearson=r2p, n=n,
                equation=f'QL/Q0 = {a:.4f} * {SYMBOL.get(indicator, indicator)}',
                equation_r2=f'QL/Q0 = {a:.4f} * {SYMBOL.get(indicator, indicator)}'
                            f'   (R2 = {r2:.4f}, n = {n})')


def slope(coeffs, indicator, group):
    s = coeffs[(coeffs['indicator'] == indicator) & (coeffs['group'] == group)]
    return float(s['slope'].iloc[0]) if len(s) else np.nan


def validate(coeffs, exp=None):
    """Apply the simulated calibration to the experimental indicators."""
    exp = load_single(EXP_CSV) if exp is None else exp
    rows = []
    for ind in INDICATORS:
        if ind not in exp.columns:
            continue
        a_glob = slope(coeffs, ind, 'GLOBAL')
        a_glob_v = slope(coeffs, ind, 'GLOBAL_no_LOC1')
        for loc in LOCS:
            for leak, ql in QL_EXP[loc].items():
                g = exp[(exp['loc'] == loc) & (exp['leak'] == leak)]
                if g.empty:
                    continue
                x = float(g[ind].mean())
                real = ql / Q0_EXP
                a_loc = slope(coeffs, ind, loc)
                rows.append(dict(
                    indicator=ind, loc=loc, leak=leak, n_tests=len(g),
                    x=x, x_std=float(g[ind].std(ddof=1)) if len(g) > 1 else np.nan,
                    QL_real=ql, ratio_Q_real=real,
                    est_loc=a_loc * x, err_loc=100 * (a_loc * x - real) / real,
                    est_global=a_glob * x,
                    err_global=100 * (a_glob * x - real) / real,
                    est_global_no_LOC1=a_glob_v * x,
                    err_global_no_LOC1=100 * (a_glob_v * x - real) / real,
                ))
    return pd.DataFrame(rows)


def summary(valid):
    """Mean absolute error over the usual subsets."""
    subsets = {
        'all (12)': lambda d: d,
        'without 0.5 mm': lambda d: d[d['leak'] != '0.5'],
        'without LOC1': lambda d: d[d['loc'] != 'LOC1'],
        'without LOC1 and 0.5 mm': lambda d: d[(d['loc'] != 'LOC1') & (d['leak'] != '0.5')],
    }
    rows = []
    for ind, d in valid.groupby('indicator', sort=False):
        for name, f in subsets.items():
            s = f(d)
            rows.append(dict(indicator=ind, subset=name, n=len(s),
                             err_loc=s['err_loc'].abs().mean(),
                             err_global=s['err_global'].abs().mean(),
                             err_global_no_LOC1=s['err_global_no_LOC1'].abs().mean()))
    return pd.DataFrame(rows)


def main():
    coeffs = calibrate()
    valid = validate(coeffs)
    summ = summary(valid)

    print('CALIBRATION  QL/Q0 = a * indicator   (fitted on the simulated data, '
          'through the origin)\n')
    for ind, d in coeffs.groupby('indicator', sort=False):
        print(f'  {ind}')
        for r in d.itertuples():
            print(f'     {r.group:16s} a = {r.slope:9.4f}   R2 = {r.r2:7.4f}   '
                  f'(pearson {r.r2_pearson:6.4f})   n = {r.n:2d}    {r.equation}')
        print()

    print('VALIDATION  (simulated calibration applied to the experimental '
          f'indicators, Q0 = {Q0_EXP} L/h)\n')
    for ind, d in valid.groupby('indicator', sort=False):
        print(f'  {ind}')
        cols = ['loc', 'leak', 'x', 'ratio_Q_real', 'est_loc', 'err_loc',
                'est_global', 'err_global']
        print(d[cols].to_string(index=False,
                                float_format=lambda v: f'{v:8.3f}'))
        print()

    print('MEAN ABSOLUTE ERROR (%)\n')
    print(summ.to_string(index=False, float_format=lambda v: f'{v:7.1f}'))

    coeffs.to_csv(OUT_COEFFS, index=False)
    valid.to_csv(OUT_VALID, index=False)
    print('\nEQUATIONS  (para copiar para as tabelas da tese)\n')
    for ind, d in coeffs.groupby('indicator', sort=False):
        for r in d.itertuples():
            print(f'  {ind:18s} {r.group:16s} {r.equation_r2}')
        print()

    print(f'Written: {OUT_COEFFS} and {OUT_VALID}')
    print(f'\nNote on LOC1:\n{BLIND_ZONE_NOTE}')


if __name__ == '__main__':
    main()
