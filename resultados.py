# -*- coding: utf-8 -*-
"""Todos os resultados de cada ensaio, uma tabela por metodologia.

Le o pair_single.csv ja escrito pelos batches e imprime, para cada metodologia
do capitulo 5, uma tabela com uma linha por ensaio e a media +- desvio-padrao
de cada combinacao. E so leitura: nao recalcula nada, para nao poder divergir
do resto.

    python resultados.py                     CS, AUT, 250 l/h, as tres LOC
    python resultados.py CS LOC2             so a LOC2
    python resultados.py CC MAN              outra instalacao e manobra
    python resultados.py SIM                 o modelo numerico
    python resultados.py CS LOC2 --csv       escreve tambem os csv

Metodologias, pela ordem do capitulo:
    localizacao      x_est e o erro, com os instantes t1 e t2 de onde saem
    queda inicial    H0, Hmax, Hmin, dH_d em metros e dH_d adimensional
    energia residual E_R/E0
    decaimento       zeta do ensaio com fuga, do sem fuga, e a diferenca
"""
import os
import sys

import numpy as np
import pandas as pd

DATASETS = {
    'CS': ('Experimental_data_CS/pair_single.csv', 'CS, experimental'),
    'CC': ('Experimental_data_CC/pair_single.csv', 'CC, experimental'),
    'SIM': ('Numerical_data_CS/pair_single.csv', 'CS, modelo numerico'),
}

# (titulo, [(coluna, cabecalho, casas decimais)])
METODOS = [
    ('LOCALIZACAO  (Seccao 5.4)', [
        ('x_real', 'x_real (m)', 3),
        ('x_M1_period', 'x_est (m)', 3),
        ('x_err_M1_period_m', 'erro (m)', 3),
        ('t1_ms', 't1 (ms)', 2),
        ('t2_ms', 't2 (ms)', 2),
        ('delta_t_ms', 't2-t1 (ms)', 2),
    ]),
    ('QUEDA INICIAL  (Seccao 5.5)', [
        ('H0_m', 'H0 (m)', 2),
        ('H_max_m', 'Hmax (m)', 2),
        ('H_min_m', 'Hmin (m)', 2),
        ('dH_m', 'dH_d (m)', 3),
        ('dHJ_NL', 'dHJ (m)', 2),
        ('dH_over_dHJ', 'dH_d', 4),
    ]),
    ('ENERGIA RESIDUAL  (Seccao 5.6)', [
        ('E_R_over_E0', 'E_R/E0', 4),
        ('E_R_over_E0_self', 'E_R/E0 alt', 4),
    ]),
    ('DECAIMENTO  (Seccao 5.7)', [
        ('zeta_total', 'zeta_L', 5),
        ('zeta_NL', 'zeta_NL', 5),
        ('zeta_leak', 'zeta_leak', 5),
        ('n_cycles_L', 'ciclos', 0),
        ('r2_lnA_L', 'R2 lnA', 4),
    ]),
]


def carregar(chave, loc=None, maneuver=None, flow=None):
    path, titulo = DATASETS[chave]
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} nao existe: corre primeiro o batch')
    df = pd.read_csv(path, dtype={'loc': str, 'leak': str})
    df = df[~df['loc'].str.contains(r'\+')]                 # so fuga simples
    if flow is not None:
        df = df[df['flow'] == flow]
    if maneuver is not None and 'maneuver' in df.columns:
        df = df[df['maneuver'] == maneuver]
    if loc:
        df = df[df['loc'] == loc]
    if df.empty:
        raise SystemExit('nenhum ensaio com esse filtro')
    col = 'leak_test' if 'leak_test' in df.columns else None
    df = df.copy()
    df['test'] = (df[col].astype(str).str.extract(r'test0*(\d+)')[0]
                  if col else '1')
    df['test'] = pd.to_numeric(df['test'], errors='coerce').fillna(1).astype(int)
    df['leak_num'] = pd.to_numeric(df['leak'], errors='coerce')
    return df.sort_values(['loc', 'leak_num', 'test']), titulo


def _fmt(v, dec):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    return f'{v:.{dec}f}' if dec else f'{v:.0f}'


def tabela(df, campos):
    """Uma linha por ensaio, mais a media +- desvio de cada combinacao."""
    campos = [(c, h, d) for c, h, d in campos if c in df.columns]
    if not campos:
        return None
    linhas = []
    for (loc, leak), g in df.groupby(['loc', 'leak'], sort=False):
        for r in g.itertuples():
            linhas.append({'LOC': loc, 'fuga': leak, 'test': str(r.test),
                           **{h: _fmt(getattr(r, c, np.nan), d)
                              for c, h, d in campos}})
        media = {'LOC': '', 'fuga': '', 'test': 'media'}
        for c, h, d in campos:
            v = pd.to_numeric(g[c], errors='coerce')
            if v.notna().sum() == 0:
                media[h] = 'n/a'
            elif len(g) > 1 and v.notna().sum() > 1:
                media[h] = f'{v.mean():.{d}f} +- {v.std(ddof=1):.{d}f}'
            else:
                media[h] = _fmt(v.mean(), d)
        linhas.append(media)
    return pd.DataFrame(linhas)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    escrever = '--csv' in argv
    if escrever:
        argv.remove('--csv')
    chave = 'CS'
    if argv and argv[0].upper() in DATASETS:
        chave = argv.pop(0).upper()
    loc = maneuver = None
    for a in argv:
        if a.upper().startswith('LOC'):
            loc = a.upper()
        elif a.upper() in ('AUT', 'MAN'):
            maneuver = a.upper()
    if maneuver is None:
        maneuver = 'AUT'

    df, titulo = carregar(chave, loc=loc, maneuver=maneuver, flow=250)
    cab = f'  {titulo}   |   {maneuver}, 250 L/h' + (f'   |   {loc}' if loc else '')
    print('=' * 78)
    print(cab)
    print('=' * 78)
    print(f'{len(df)} ensaios, {df.groupby(["loc", "leak"]).ngroups} combinacoes\n')

    for nome, campos in METODOS:
        t = tabela(df, campos)
        if t is None or t.shape[1] <= 3:
            continue
        print(nome)
        print('-' * len(nome))
        print(t.to_string(index=False))
        print()
        if escrever:
            out = os.path.join(os.path.dirname(DATASETS[chave][0]),
                               'resultados_' + nome.split()[0].lower() + '.csv')
            t.to_csv(out, index=False)
            print(f'  escrito: {out}\n')


if __name__ == '__main__':
    main()
