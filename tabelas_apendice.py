# -*- coding: utf-8 -*-
"""Tabelas dos apendices, em CSV e num .docx formatado.

So le resultados ja calculados -- o pair_single.csv de cada instalacao e os
coeficientes de calibracao do modelo. Nao recalcula indicador nenhum nem toca
em nenhum metodo: o que aqui se faz e escolher colunas, aplicar as regressoes
que ja existem e formatar.

    C.2   CS, 250 l/h, fuga unica, cada repeticao: os tres indicadores
    D.1   CS, LOC2, cada ensaio, dos 150 aos 850 l/h: localizacao,
          indicadores e Q_L/Q0 estimado por cada um
    D.2   CS, 250 l/h, duas fugas, cada repeticao: as duas localizacoes,
          indicadores e Q_L/Q0 estimado contra o caudal total

    python tabelas_apendice.py
"""
import os

import numpy as np
import pandas as pd

import calibration_coeffs as C

OUTDIR = os.path.join('results', 'appendix_tables')
EXP_CSV = os.path.join('Experimental_data_CS', 'pair_single.csv')
FAC, MANEUVER, L_PIPE = 'CS', 'AUT', 15.0
LOCS = ['LOC1', 'LOC2', 'LOC3']
SIZES = ['0.5', '1.0', '1.5', '2.0']
FLOWS = [150, 250, 350, 850]
SIZES_FLUXO = ['0.5', '1.5']    # os unicos calibres ensaiados nos 4 caudais
VAZIO = '--'               # valor que nao existe no ensaio

# Q_L/Q0 medido na bancada ao caudal de calibracao. E o valor que as tabelas
# do capitulo 7 usam, e NAO o ratio_Q gravado no pair_single.csv (0.0531,
# 0.2151, 0.4750, 0.8448), que no calibre de 1.0 mm difere 46 %.
RATIO_Q_EXP = {'0.5': 0.0724, '1.0': 0.3138, '1.5': 0.5258, '2.0': 0.8892}
FLOW_EXP = 250             # caudal a que esses valores foram medidos

CC_CSV = os.path.join('Experimental_data_CC', 'pair_single.csv')
# A fuga da CC esta a x/L = 0.629, entre a LOC2 e a LOC3 da CS: os declives
# sao os interpolados entre as duas, como no Validacao_sizing.xlsx.
CC_A = {'dH_over_dHJ': 3.824, 'E_R_over_E0_sqrt': 4.035, 'zeta_leak': 18.874}

# As tres metodologias, na forma em que sao calibradas: a energia residual
# entra pela raiz, que e a forma com que a regressao foi ajustada.
INDS = [('dH_over_dHJ', 'dH_over_dHJ', 'dH_d'),
        ('E_R_over_E0', 'E_R_over_E0_sqrt', 'E_R/E_NL'),
        ('zeta_leak', 'zeta_leak', 'd_zeta')]

CASAS = {'t': 2, 'x': 3, 'ind': 4, 'err': 2}


def _fmt(v, casas):
    return VAZIO if v is None or not np.isfinite(v) else f'{v:.{casas}f}'


def _ler(path, maneuver=None):
    """Um pair_single.csv com as colunas derivadas que as tabelas usam."""
    d = pd.read_csv(path, dtype={'loc': str, 'leak': str})
    if maneuver is not None:
        d = d[d['maneuver'] == maneuver]
    d = d.copy()
    d['test'] = d['leak_test'].str.extract(r'test(\d+)')[0].astype(int)
    d['E_R_over_E0_sqrt'] = np.sqrt(d['E_R_over_E0'].clip(lower=0))
    return d


def carregar():
    """Ensaios das duas instalacoes e os declives das regressoes do modelo."""
    d = _ler(EXP_CSV, MANEUVER)
    cc = _ler(CC_CSV)
    sim = C.load_single(C.SIM_CSV)
    sim = sim[sim['n_leaks'] == 1]
    co = C.calibrate(sim)
    a = {loc: {col: C.slope(co, col, loc) for _, col, _ in INDS}
         for loc in LOCS}
    return d, cc, a


def _agrupa(linhas, chaves):
    """Deixa em branco as celulas repetidas, como na Tabela C.1."""
    ant = [None] * len(chaves)
    for r in linhas:
        for i, k in enumerate(chaves):
            if r[k] == ant[i]:
                r[k] = ''
            else:
                ant[i] = r[k]
                ant[i + 1:] = [None] * (len(chaves) - i - 1)
    return linhas


def tabela_c2(d):
    """CS, 250 l/h, fuga unica: os tres indicadores em cada repeticao."""
    g = d[(d['flow'] == 250) & (~d['loc'].str.contains(r'\+'))]
    linhas = []
    for loc in LOCS:
        for size in SIZES:
            h = g[(g['loc'] == loc) & (g['leak'] == size)].sort_values('test')
            for _, r in h.iterrows():
                linhas.append({
                    'Location': loc,
                    'Leak size': f'L{size}',
                    'Test': f'{r["test"]:d}',
                    'dH_d (-)': _fmt(r['dH_over_dHJ'], CASAS['ind']),
                    'E_R/E_NL (-)': _fmt(r['E_R_over_E0'], CASAS['ind']),
                    'd_zeta (-)': _fmt(r['zeta_leak'], CASAS['ind']),
                })
    return pd.DataFrame(_agrupa(linhas, ['Location', 'Leak size']))


def _real_exp(size, flow, do_csv):
    """Q_L/Q0 medido da bancada: o do capitulo 7 ao caudal de calibracao.

    Nos outros caudais esse valor nao existe -- o caudal de fuga mantem-se mas
    o Q0 muda --, por isso fica o do ficheiro de resultados.
    """
    if flow == FLOW_EXP and size in RATIO_Q_EXP:
        return RATIO_Q_EXP[size]
    return float(do_csv)


def tabela_d1(d, a):
    """CS, LOC2, cada ensaio: localizacao, indicadores e Q_L/Q0 estimado."""
    g = d[(d['loc'] == 'LOC2')]
    linhas = []
    # so os calibres que existem nos quatro caudais, que e o que permite ler
    # a dependencia do caudal inicial sem comparar combinacoes diferentes
    for flow in FLOWS:
        for size in SIZES_FLUXO:
            h = g[(g['flow'] == flow) & (g['leak'] == size)].sort_values('test')
            for _, r in h.iterrows():
                x = r['x_Theoretical_ord']
                err_x = 100 * abs(x - r['x_real']) / L_PIPE
                real = _real_exp(size, flow, r['ratio_Q'])
                lin = {
                    'Initial flow rate (l/h)': f'{flow:d}',
                    'Leak size': f'L{size}',
                    'Test': f'{r["test"]:d}',
                    'X_est (m)': _fmt(x, CASAS['x']),
                    'eps_X (%)': _fmt(err_x, CASAS['err']),
                    'dH_d (-)': _fmt(r['dH_over_dHJ'], CASAS['ind']),
                    'E_R/E_NL (-)': _fmt(r['E_R_over_E0'], CASAS['ind']),
                    'd_zeta (-)': _fmt(r['zeta_leak'], CASAS['ind']),
                }
                for nome, col, rot in INDS:
                    est = a[col] * r[col]
                    lin[f'(Q_L/Q0)_est {rot} (-)'] = _fmt(est, CASAS['ind'])
                lin['(Q_L/Q0)_real (-)'] = _fmt(real, CASAS['ind'])
                for nome, col, rot in INDS:
                    est = a[col] * r[col]
                    e = (100 * (est / real - 1)
                         if np.isfinite(real) and real else np.nan)
                    lin[f'eps_Q {rot} (%)'] = _fmt(e, CASAS['err'])
                linhas.append(lin)
    return pd.DataFrame(_agrupa(linhas, ['Initial flow rate (l/h)',
                                         'Leak size']))


def tabela_d2(d, a):
    """CS, 250 l/h, duas fugas: as duas posicoes e o caudal total."""
    g = d[(d['flow'] == 250) & (d['loc'].str.contains(r'\+'))]
    linhas = []
    for loc, h0 in g.groupby('loc', sort=True):
        for leak, h in h0.groupby('leak', sort=True):
            for _, r in h.sort_values('test').iterrows():
                # por ORDEM DE CHEGADA do eco: o primeiro eco e o da fuga mais
                # proxima, por isso as estimativas ordenadas emparelham com as
                # posicoes reais ordenadas
                xr = sorted([r['x_real'], r['x_real_leak2']])
                xe = sorted([r['x_Theoretical_ord'],
                             r['x_Theoretical_ord_leak2']])
                lin = {
                    'Leak locations': loc,
                    'Leak sizes': leak,
                    'Test': f'{r["test"]:d}',
                }
                for i in (0, 1):
                    e = (100 * abs(xe[i] - xr[i]) / L_PIPE
                         if np.isfinite(xe[i]) else np.nan)
                    lin[f'X_est,{i + 1} (m)'] = _fmt(xe[i], CASAS['x'])
                    lin[f'eps_X,{i + 1} (%)'] = _fmt(e, CASAS['err'])
                lin['dH_d (-)'] = _fmt(r['dH_over_dHJ'], CASAS['ind'])
                lin['E_R/E_NL (-)'] = _fmt(r['E_R_over_E0'], CASAS['ind'])
                lin['d_zeta (-)'] = _fmt(r['zeta_leak'], CASAS['ind'])
                # o total das duas fugas, cada uma pelo seu calibre
                real = sum(RATIO_Q_EXP[k] for k in leak.split('+')
                           if k in RATIO_Q_EXP)
                for nome, col, rot in INDS:
                    lin[f'(Q_L/Q0)_est {rot} (-)'] = _fmt(a[col] * r[col],
                                                          CASAS['ind'])
                lin['(Q_L/Q0)_real (-)'] = _fmt(real, CASAS['ind'])
                for nome, col, rot in INDS:
                    est = a[col] * r[col]
                    e = (100 * (est / real - 1)
                         if np.isfinite(real) and real else np.nan)
                    lin[f'eps_Q {rot} (%)'] = _fmt(e, CASAS['err'])
                linhas.append(lin)
    return pd.DataFrame(_agrupa(linhas, ['Leak locations', 'Leak sizes']))


def _linha_sizing(r, a, ident, real=None):
    """Uma linha de tabela de dimensionamento: indicadores, estimativas e erro.

    `a` sao os declives a usar nessa linha (os da localizacao, ou os da CC).
    `real` so vem preenchido onde o caudal de fuga foi MEDIDO; onde nao foi,
    a linha fica sem o valor e sem os erros.
    """
    lin = dict(ident)
    lin['dH_d (-)'] = _fmt(r['dH_over_dHJ'], CASAS['ind'])
    lin['E_R/E_NL (-)'] = _fmt(r['E_R_over_E0'], CASAS['ind'])
    lin['d_zeta (-)'] = _fmt(r['zeta_leak'], CASAS['ind'])
    for _nome, col, rot in INDS:
        lin[f'(Q_L/Q0)_est {rot} (-)'] = _fmt(a[col] * r[col], CASAS['ind'])
    lin['(Q_L/Q0)_real (-)'] = _fmt(real, CASAS['ind']) if real else VAZIO
    for _nome, col, rot in INDS:
        e = (100 * (a[col] * r[col] / real - 1)
             if real and np.isfinite(real) else np.nan)
        lin[f'eps_Q {rot} (%)'] = _fmt(e, CASAS['err'])
    return lin


def sizing_cs_250(d, a):
    """CS, 250 l/h, fuga unica: as tres localizacoes, ensaio a ensaio."""
    g = d[(d['flow'] == 250) & (~d['loc'].str.contains(r'\+'))]
    linhas = []
    for loc in LOCS:
        for size in SIZES:
            h = g[(g['loc'] == loc) & (g['leak'] == size)].sort_values('test')
            for _, r in h.iterrows():
                linhas.append(_linha_sizing(
                    r, a[loc],
                    [('Location', loc), ('Leak size', f'L{size}'),
                     ('Test', f'{r["test"]:d}')],
                    RATIO_Q_EXP.get(size)))
    return pd.DataFrame(_agrupa(linhas, ['Location', 'Leak size']))


def sizing_cs_caudais(d, a, caudais, loc='LOC2'):
    """CS, LOC2, fora do caudal de calibracao: ensaio a ensaio.

    Sem coluna de Q_L/Q0 real: o caudal de fuga so foi medido a 250 l/h.
    """
    linhas = []
    for flow in caudais:
        for size in SIZES:
            h = d[(d['flow'] == flow) & (d['loc'] == loc)
                  & (d['leak'] == size)].sort_values('test')
            for _, r in h.iterrows():
                linhas.append(_linha_sizing(
                    r, a[loc],
                    [('Initial flow rate (l/h)', f'{flow:d}'),
                     ('Leak size', f'L{size}'), ('Test', f'{r["test"]:d}')]))
    return pd.DataFrame(_agrupa(linhas, ['Initial flow rate (l/h)',
                                         'Leak size']))


def sizing_duas_fugas(d, a):
    """CS, 250 l/h, duas fugas: ensaio a ensaio, com os declives da LOC2.

    Com duas fugas nao ha "a" localizacao: usam-se os declives da LOC2, a
    posicao central. O caudal total nao foi medido, por isso nao ha erro.
    """
    g = d[(d['flow'] == 250) & (d['loc'].str.contains(r'\+'))]
    linhas = []
    for loc, h0 in g.groupby('loc', sort=True):
        for leak, h in h0.groupby('leak', sort=True):
            for _, r in h.sort_values('test').iterrows():
                linhas.append(_linha_sizing(
                    r, a['LOC2'],
                    [('Leak locations', loc), ('Leak sizes', leak),
                     ('Test', f'{r["test"]:d}')]))
    return pd.DataFrame(_agrupa(linhas, ['Leak locations', 'Leak sizes']))


def sizing_cc(cc):
    """CC: as duas manobras e os tres caudais, um ensaio por caso."""
    linhas = []
    for man, rotulo in (('AUT', 'Automatic'), ('MAN', 'Manual')):
        for flow in sorted(cc['flow'].unique()):
            for size in SIZES:
                h = cc[(cc['maneuver'] == man) & (cc['flow'] == flow)
                       & (cc['leak'] == size)]
                for _, r in h.iterrows():
                    linhas.append(_linha_sizing(
                        r, CC_A,
                        [('Manoeuvre', rotulo),
                         ('Initial flow rate (l/h)', f'{int(flow):d}'),
                         ('Leak size', f'L{size}')]))
    return pd.DataFrame(_agrupa(linhas, ['Manoeuvre',
                                         'Initial flow rate (l/h)']))


def escrever_docx(tabelas, path):
    """Uma tabela por pagina, com a legenda por cima."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, m, Cm(1.5))
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(8)

    for i, (legenda, df) in enumerate(tabelas):
        p = doc.add_paragraph(legenda)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.runs[0].bold = True
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = 'Table Grid'
        for j, c in enumerate(df.columns):
            cel = t.rows[0].cells[j]
            cel.text = str(c)
            cel.paragraphs[0].runs[0].bold = True
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for j, c in enumerate(df.columns):
                cells[j].text = str(r[c])
                cells[j].paragraphs[0].alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT if j < 3
                    else WD_ALIGN_PARAGRAPH.RIGHT)
        if i < len(tabelas) - 1:
            doc.add_page_break()
    doc.save(path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d, cc, a = carregar()
    for loc in LOCS:
        print(f'declives da {loc} (modelo numerico): '
              + '   '.join(f'{rot} {a[loc][col]:.3f}' for _, col, rot in INDS))
    print('declives da CC (interpolados, x/L = 0.629): '
          + '   '.join(f'{rot} {CC_A[col]:.3f}' for _, col, rot in INDS))

    tabelas = [
        ('Table C.2 - CS, 250 l/h, single leak, individual repetitions. '
         'Sizing indicators.',
         'table_C2_CS_250_single_leak_indicators', tabela_c2(d)),
        ('Table D.1 - CS, LOC2, individual tests at 150, 250, 350 and '
         '850 l/h (the last with cavitation). Location, sizing indicators '
         'and leak flow estimated with the LOC2 regressions.',
         'table_D1_CS_LOC2_flow_dependence', tabela_d1(d, a['LOC2'])),
        ('Table D.2 - CS, 250 l/h, two simultaneous leaks, individual '
         'repetitions. Locations assigned by echo arrival order; leak flow '
         'estimated with the LOC2 regressions and compared with the total.',
         'table_D2_CS_250_two_leaks', tabela_d2(d, a['LOC2'])),
        ('Table E.1 - CS, 250 l/h, single leak, individual repetitions. '
         'Sizing indicators.',
         'table_E1_CS_250_sizing', sizing_cs_250(d, a)),
        ('Table E.2 - CS, LOC2, individual repetitions at 150 and 350 l/h. '
         'Sizing indicators.',
         'table_E2_CS_150_350_sizing', sizing_cs_caudais(d, a, [150, 350])),
        ('Table E.3 - CS, LOC2, 850 l/h (with cavitation), individual '
         'repetitions. Sizing indicators.',
         'table_E3_CS_850_sizing', sizing_cs_caudais(d, a, [850])),
        ('Table E.4 - CS, 250 l/h, two simultaneous leaks, individual '
         'repetitions. Sizing indicators.',
         'table_E4_CS_250_two_leaks_sizing', sizing_duas_fugas(d, a)),
        ('Table E.5 - CC, both manoeuvres and the three initial flow rates, '
         'one test per case. Sizing indicators.',
         'table_E5_CC_sizing', sizing_cc(cc)),
    ]
    # Fora da C.2 e da D.1/D.2, as tabelas levam so os indicadores medidos: as
    # estimativas de Q_L/Q0 e os erros ficam para o texto, que usa as medias.
    def _so_medido(nome, df):
        if not nome.startswith('table_E'):
            return df
        return df[[c for c in df.columns
                   if not c.startswith(('(Q_L/Q0)', 'eps_Q'))]]

    tabelas = [(leg, nome, _so_medido(nome, df).loc[
        :, ~(_so_medido(nome, df) == VAZIO).all()])
        for leg, nome, df in tabelas]
    for legenda, nome, df in tabelas:
        path = os.path.join(OUTDIR, nome + '.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'   {nome}.csv   {len(df)} linhas x {len(df.columns)} colunas')
        n = sum((df[c] == VAZIO).sum() for c in df.columns)
        if n:
            print(f'      {n} celulas sem valor, assinaladas com "{VAZIO}"')
    escrever_docx([(leg, df) for leg, _, df in tabelas],
                  os.path.join(OUTDIR, 'appendix_tables.docx'))
    print(f'\nGuardado em {OUTDIR}')


if __name__ == '__main__':
    main()
