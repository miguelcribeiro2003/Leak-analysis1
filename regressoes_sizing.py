# -*- coding: utf-8 -*-
"""Comparacao de formas de regressao para as tres metodologias de sizing.

Os coeficientes sao sempre ajustados nos dados SIMULADOS (Numerical_data_CS) e
depois aplicados aos indicadores da bancada, como em calibration_coeffs.py. O
que muda aqui e a FORMA do ajuste: para cada indicador e cada grupo testam-se

    y = a * x          (a recta pela origem da calibracao actual)
    y = a * sqrt(x)    (expoente 0.5 fixo, previsto pela definicao de E_R/E0)
    y = a * x^b        (expoente livre, estimado com os proprios pontos)

com y = QL/Q0. Todas passam pela origem, portanto a comparacao e justa.

R2 = 1 - SSE/SST, com SSE = sum((y - yhat)^2) do modelo em causa e
SST = sum((y - ybar)^2), SEMPRE no espaco de y. E o R2 do modelo ajustado, nao
uma correlacao, e e por isso que se pode por os tres lado a lado: o unico que
muda entre eles e o SSE. Sem termo constante o R2 pode dar negativo, o que
apenas significa que o modelo e pior do que a media dos pontos.

O expoente livre e estimado minimizando o SSE no espaco de y (nao em log), que
e exactamente o que o R2 mede, de modo que o y = a*x^b nunca pode sair com R2
inferior aos outros dois: por construcao e o melhor dos tres no ajuste. A
coluna err_exp esta aqui por causa disso -- mostra o que cada forma faz na
bancada, onde um expoente ajustado a quatro pontos pode piorar a previsao.

    python regressoes_sizing.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import calibration_coeffs as cc

OUT_CSV = 'regressoes_sizing.csv'

# as tres metodologias de sizing, na ordem em que aparecem na tese
INDICADORES = ['dH_over_dHJ', 'E_R_over_E0', 'zeta_leak']

# grupo -> filtro sobre o dataframe
GRUPOS = {
    'LOC1': lambda d: d[d['loc'] == 'LOC1'],
    'LOC2': lambda d: d[d['loc'] == 'LOC2'],
    'LOC3': lambda d: d[d['loc'] == 'LOC3'],
    'GLOBAL': lambda d: d,
    'GLOBAL_no_LOC1': lambda d: d[d['loc'] != 'LOC1'],
}


def r2_sse(y, yhat):
    """1 - SSE/SST, no espaco de y."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    if not sst:
        return np.nan
    return 1.0 - float(np.sum((y - yhat) ** 2)) / sst


def fit_power_fixo(x, y, p):
    """y = a * x^p com p fixo: minimo de SSE em a, em forma fechada."""
    u = x ** p
    den = float(u @ u)
    return float((u @ y) / den) if den else np.nan


def fit_power_livre(x, y):
    """y = a * x^b, os dois parametros ajustados por minimos quadrados em y.

    O arranque vem da regressao linear em log-log, que e um ajuste em erro
    relativo; a partir dai o curve_fit desce no SSE em y, que e o que entra
    no R2.
    """
    ok = (x > 0) & (y > 0)
    if ok.sum() < 3:          # dois parametros exigem mais de dois pontos
        return np.nan, np.nan
    b0, lna0 = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    try:
        (a, b), _ = curve_fit(lambda xx, a, b: a * xx ** b, x, y,
                              p0=(float(np.exp(lna0)), float(b0)), maxfev=20000)
    except RuntimeError:      # nao convergiu: fica o ajuste em log-log
        return float(np.exp(lna0)), float(b0)
    return float(a), float(b)


def ajustes(x, y):
    """Os tres modelos para um conjunto de pontos: (nome, a, b, yhat)."""
    out = []
    for nome, p in (('y = a*x', 1.0), ('y = a*x^0.5', 0.5)):
        a = fit_power_fixo(x, y, p)
        out.append((nome, a, p, a * x ** p))
    a, b = fit_power_livre(x, y)
    out.append(('y = a*x^b', a, b, a * x ** b if np.isfinite(a) else np.full_like(x, np.nan)))
    return out


def carrega():
    """Simulado (x e y do modelo) e experimental (x da bancada, y pesado)."""
    sim = cc.load_single(cc.SIM_CSV).copy()
    sim['y'] = sim['ratio_Q']
    exp = cc.load_single(cc.EXP_CSV).copy()
    exp['y'] = [cc.QL_EXP[l][k] / cc.Q0_EXP for l, k in zip(exp['loc'], exp['leak'])]
    return sim, exp


def tabela(sim, exp):
    linhas = []
    for ind in INDICADORES:
        for grupo, filtro in GRUPOS.items():
            g = filtro(sim)
            x = g[ind].to_numpy(float)
            y = g['y'].to_numpy(float)
            # mesmo subconjunto na bancada, com as repeticoes de cada calibre
            # reduzidas a media, que e o x que a validacao usa
            e = filtro(exp).groupby(['loc', 'leak'], as_index=False).agg(
                x=(ind, 'mean'), y=('y', 'mean'))
            for nome, a, b, yhat in ajustes(x, y):
                est = a * e['x'].to_numpy(float) ** b
                err = 100 * (est - e['y'].to_numpy(float)) / e['y'].to_numpy(float)
                linhas.append(dict(
                    indicador=ind, grupo=grupo, modelo=nome, a=a, b=b,
                    r2=r2_sse(y, yhat), n=len(y),
                    err_exp=float(np.abs(err).mean()), n_exp=len(e)))
    return pd.DataFrame(linhas)


def main():
    sim, exp = carrega()
    tab = tabela(sim, exp)

    print('REGRESSOES DE SIZING   y = QL/Q0,  ajuste nos dados simulados\n'
          'R2 = 1 - SSE/SST no espaco de y;  err_exp = erro absoluto medio (%) '
          'do mesmo modelo aplicado a bancada\n')
    for ind, d in tab.groupby('indicador', sort=False):
        print(f'  {ind}')
        for grupo, g in d.groupby('grupo', sort=False):
            melhor = g['r2'].idxmax()
            for r in g.itertuples():
                marca = ' <-- melhor R2' if r.Index == melhor else ''
                print(f'     {grupo:16s} {r.modelo:12s} a = {r.a:9.4f}  '
                      f'b = {r.b:6.3f}   R2 = {r.r2:8.4f}   '
                      f'err_exp = {r.err_exp:6.1f}%{marca}')
            print()

    print('MELHOR R2 POR INDICADOR E GRUPO\n')
    melhores = tab.loc[tab.groupby(['indicador', 'grupo'], sort=False)['r2'].idxmax()]
    print(melhores[['indicador', 'grupo', 'modelo', 'a', 'b', 'r2', 'err_exp']]
          .to_string(index=False, float_format=lambda v: f'{v:9.4f}'))

    tab.to_csv(OUT_CSV, index=False)
    print(f'\nEscrito: {OUT_CSV}')


if __name__ == '__main__':
    main()
