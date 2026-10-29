# -*- coding: utf-8 -*-
"""Tabela de um indicador por combinacao, com a dispersao entre repeticoes.

Uma linha por (localizacao, calibre): a MEDIA dos ensaios repetidos e o
coeficiente de variacao, CV = desvio-padrao / media. E a tabela que acompanha
as figuras de cada metodologia de sizing no Capitulo 5.

    python tabela_indicadores.py                  -> dH_over_dHJ (bancada)
    python tabela_indicadores.py zeta_leak
    python tabela_indicadores.py E_R_over_E0_sqrt SIM

Os casos sao os mesmos da calibracao -- fuga unica, 250 L/h, manobra AUT, sem
os caudais com cavitacao --, lidos por calibration_coeffs.load_single, pelo que
a tabela e o ajuste nunca podem discordar.

O CV usa o desvio-padrao amostral (ddof = 1) e e relativo, o que o torna
comparavel entre indicadores e entre calibres: os tres indicadores tem unidades
e ordens de grandeza diferentes, e um desvio absoluto nao diz nada sobre se a
medicao e boa ou ma sem se saber o valor a que se refere.

Nao ha RMSE nesta tabela, e nao e esquecimento: o RMSE mede o desvio a um valor
de referencia, e aqui nao ha nenhum -- sao repeticoes da mesma medicao, cujo
'erro' e a propria dispersao. Um RMSE calculado contra a media das repeticoes
seria, a menos do factor sqrt((n-1)/n), o desvio-padrao outra vez. O RMSE tem
lugar na VALIDACAO (calibration_coeffs.validate), onde a estimativa de QL/Q0 e
comparada com o caudal pesado.
"""
import os
import sys

import numpy as np
import pandas as pd

import calibration_coeffs as cc

LOCS = ['LOC1', 'LOC2', 'LOC3']
SIZES = ['0.5', '1.0', '1.5', '2.0']
# como cada indicador se escreve no cabecalho da tabela
# Como cada indicador se escreve na tese. O zeta_leak e o Delta zeta da eq.
# (5.18), a diferenca entre o amortecimento com fuga e sem fuga; a coluna do
# codigo mantem o nome antigo.
TITULO = {'dH_over_dHJ': 'dH_d (-)', 'E_R_over_E0': 'E_R/E0 (-)',
          'E_R_over_E0_sqrt': 'sqrt(E_R/E0) (-)',
          'zeta_leak': 'delta_zeta (-)'}
DECIMAIS = {'zeta_leak': 4}
DEC_DEF = 4


def tabela(indicador='dH_over_dHJ', dataset='EXP'):
    """Media e dispersao de um indicador, por localizacao e calibre."""
    csv = cc.EXP_CSV if dataset == 'EXP' else cc.SIM_CSV
    df = cc.load_single(csv)
    if indicador not in df.columns:
        raise SystemExit(f'{indicador} nao existe em {csv}')
    linhas = []
    for loc in LOCS:
        for size in SIZES:
            g = df[(df['loc'] == loc) & (df['leak'] == size)][indicador]
            g = g[np.isfinite(g)]
            if g.empty:
                continue
            sd = float(g.std(ddof=1)) if len(g) > 1 else np.nan
            media = float(g.mean())
            linhas.append({
                'Location': loc,
                'Identification': f'L{size}',
                'mean': media,
                'CV (%)': 100 * sd / media if media else np.nan,
                'n': int(len(g)),
            })
    return pd.DataFrame(linhas)


def imprime(tab, indicador):
    """A tabela no formato em que vai para a tese, uma linha por combinacao."""
    dec = DECIMAIS.get(indicador, DEC_DEF)
    cab = TITULO.get(indicador, indicador)
    larg = max(len(cab), 16)
    print(f'{"Location":<10}{"Identification":<16}{cab:>{larg}}'
          f'{"CV (%)":>9}{"n":>4}')
    print('-' * (10 + 16 + larg + 9 + 4))
    for loc, g in tab.groupby('Location', sort=False):
        for k, r in enumerate(g.itertuples()):
            print(f'{loc if k == 0 else "":<10}{r.Identification:<16}'
                  f'{r.mean:>{larg}.{dec}f}{getattr(r, "_4"):>9.2f}{r.n:>4d}')
    print()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    indicador = 'dH_over_dHJ'
    dataset = 'EXP'
    for a in argv:
        if a.upper() in ('EXP', 'SIM'):
            dataset = a.upper()
        else:
            indicador = a
    tab = tabela(indicador, dataset)
    print(f'\n{indicador}   ({"bancada" if dataset == "EXP" else "modelo"}, '
          f'{cc.FLOW} L/h, manobra {cc.MANEUVER})\n')
    imprime(tab, indicador)
    out = os.path.join(os.path.dirname(cc.EXP_CSV if dataset == 'EXP'
                                       else cc.SIM_CSV),
                       f'table_{indicador}_repeatability.csv')
    tab.to_csv(out, index=False)
    print(f'Escrito: {out}')


if __name__ == '__main__':
    main()
