# -*- coding: utf-8 -*-
"""O regime permanente de cada ensaio de uma pasta: H0, perda de carga e Q0.

O regime permanente e a janela que o resto da analise usa para o H0 -- do
inicio do registo ate PRE_ONSET_MARGIN (5 ms) antes do arranque da manobra, com
o arranque detectado em cada ensaio pelo detect_onset. O valor e a MEDIANA
nessa janela, como no _h0_pre do analysis: o escoamento permanente oscila com a
bomba, e a mediana nao se deixa levar por essa oscilacao.

Por ensaio imprime, no PT1 e no PT2:

    H0      mediana na janela (m)
    sd      desvio-padrao na janela (m) -- a oscilacao, nao incerteza do H0
    deriva  declive de uma recta na janela (m/s): diz se o permanente ainda
            estava a assentar quando a manobra arrancou

e ainda a diferenca PT1-PT2 (perda de carga mais desnivel entre os dois
transdutores) e o Q0 mediano do canal de caudal, que em algumas campanhas nao
esta ligado (flow_valid = false na CC).

No fim compara os ensaios entre si: media, amplitude e CV do H0, e o desvio de
cada um face ao ensaio sem fuga.

    python h0_ensaios.py CC_20261007
    python h0_ensaios.py fenomeno_350_CC_L0.5 --pipe CC
"""
import argparse
import glob
import os

import numpy as np

import analysis as an


def janela(t, y):
    """A janela do permanente e o instante do arranque, pela regra do analysis."""
    i_on = an.detect_onset(y, t, base_end_t=-0.01)
    t_on = float(t[i_on]) if i_on is not None else None
    m = an._base_mask(t, t_on)
    if m.sum() < 5:                      # registo curto: a pre-manobra toda
        m = t < (t_on if t_on is not None else an.PRE_MANEUVER_MARGIN)
    return m, t_on


def resumo(t, y, m):
    """Mediana, desvio-padrao e declive de y na janela m."""
    yy, tt = y[m], t[m]
    declive = np.polyfit(tt, yy, 1)[0] if m.sum() >= 5 else np.nan
    return float(np.median(yy)), float(np.std(yy)), float(declive)


def pipe_da_pasta(pasta, dado):
    if dado:
        return dado
    nome = os.path.basename(os.path.normpath(pasta)).upper()
    for p in ('CC', 'CS'):
        if p in nome.replace('_', ' ').split() or f'_{p}_' in f'_{nome}_':
            return p
    return 'CC'


def etiqueta(path):
    """Um nome curto para a coluna da esquerda."""
    base = os.path.splitext(os.path.basename(path))[0]
    for pedaco in ('EXP_', 'CC_', 'CS_'):
        if base.startswith(pedaco):
            base = base[len(pedaco):]
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pasta')
    ap.add_argument('--pipe', default=None, help='CC ou CS (por omissao, pelo nome da pasta)')
    args = ap.parse_args()

    pipe = pipe_da_pasta(args.pasta, args.pipe)
    an.configure_pipe(pipe)
    fich = sorted(glob.glob(os.path.join(args.pasta, '*.txt'))
                  + glob.glob(os.path.join(args.pasta, '*.dat')))
    if not fich:
        raise SystemExit(f'Sem ficheiros em {args.pasta}')

    print(f'Regime permanente, {args.pasta}   (perfil {pipe}, '
          f'colunas {an.USECOLS}, caudal em {"kl/h" if an.Q_UNIT_TO_M3S == 1/3600 else "?"})')
    print(f'Janela: do inicio do registo ate {1000*an.PRE_ONSET_MARGIN:.0f} ms '
          f'antes do arranque de cada ensaio.\n')
    cab = (f'{"ensaio":<34}{"t_on":>8}{"n":>7}'
           f'{"H0 PT1":>9}{"sd":>7}{"deriva":>9}'
           f'{"H0 PT2":>9}{"sd":>7}{"PT1-PT2":>9}{"Q0":>9}')
    print(cab)
    print(f'{"":<34}{"(ms)":>8}{"":>7}{"(m)":>9}{"(m)":>7}{"(m/s)":>9}'
          f'{"(m)":>9}{"(m)":>7}{"(m)":>9}{"(kl/h)":>9}')
    linhas = []
    for path in fich:
        lab, df = an.load_file(path)
        t = df.index.to_numpy(float)
        pt1 = df['PT1'].to_numpy(float)
        m, t_on = janela(t, pt1)
        h1, sd1, dr1 = resumo(t, pt1, m)
        h2, sd2, _ = resumo(t, df['PT2'].to_numpy(float), m)
        q0 = float(np.median(df['Q'].to_numpy(float)[m]))
        nome = etiqueta(path)
        t_on_txt = f'{1000*t_on:8.2f}' if t_on is not None else f'{"n/a":>8}'
        print(f'{nome:<34}{t_on_txt}{m.sum():7d}'
              f'{h1:9.3f}{sd1:7.3f}{dr1:9.3f}'
              f'{h2:9.3f}{sd2:7.3f}{h1 - h2:9.3f}{q0:9.4f}')
        linhas.append((nome, h1, '_NL_' in nome or nome.startswith('NL')))

    com = [h for _n, h, nl in linhas if not nl]
    nl = [h for _n, h, is_nl in linhas if is_nl]
    if len(com) > 1:
        v = np.array(com)
        print(f'\n{len(v)} ensaios com fuga: media {v.mean():.3f} m, '
              f'amplitude {v.max() - v.min():.3f} m, '
              f'CV {100 * v.std(ddof=1) / v.mean():.2f} %')
    if nl:
        print(f'ensaio(s) sem fuga: {", ".join(f"{h:.3f}" for h in nl)} m')
        if com:
            d = np.array(com) - nl[0]
            print(f'desvio de cada ensaio com fuga face ao sem fuga: '
                  f'{"  ".join(f"{x:+.3f}" for x in d)} m')


if __name__ == '__main__':
    main()
