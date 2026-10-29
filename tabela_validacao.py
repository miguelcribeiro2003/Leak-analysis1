# -*- coding: utf-8 -*-
"""O Excel de validacao das equacoes de sizing, uma folha por metodologia.

As equacoes sao as ajustadas nos dados SIMULADOS (Q_L/Q0 = a * indicador, por
localizacao) e a validacao aplica-as aos indicadores MEDIDOS na bancada. Cada
linha e uma combinacao localizacao x calibre:

    X_sim        o indicador no ensaio simulado dessa combinacao
    X_exp        o indicador medido, media dos tres ensaios da bancada
    Q_L/Q0_est   o caudal de fuga do MODELO: o Qf que o solver escreve no no da
                 fuga (rede_rp/rp_*.dat), sobre o Q0 do rp_NL (246.22 l/h)
    Q_L/Q0_real  o caudal de fuga pesado no laboratorio, sobre Q0 (240.76 l/h)
    Erro (%)     = (est - real) / real * 100                     (FORMULA)

As tres ultimas colunas repetem o exercicio com a EQUACAO de sizing em vez do
modelo: a da localizacao, a estimativa a * X_exp e o erro respectivo. Todas as
colunas de erro sao formulas, para o ficheiro continuar vivo.

    python tabela_validacao.py
"""
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import batch_simulated as B
import calibration_coeffs as C

SAIDA = 'Validacao_sizing.xlsx'
# (indicador usado na equacao, nome da folha, rotulo do indicador)
METODOS = [
    ('dH_over_dHJ', 'dH_d', 'ΔH_d'),
    ('E_R_over_E0_sqrt', 'raiz_ER_E0', '√(E_R/E0)'),
    ('zeta_leak', 'delta_zeta', 'Δζ'),
]
CAB = ['Location', 'Identification', 'X_sim', 'X_exp (média)', 'n',
       'Q_L/Q0_est', 'Q_L/Q0_real', 'Error (%)',
       'a (eq.)', 'Q_L/Q0 pela eq.', 'Error eq. (%)']

NEGRITO = Font(bold=True)
# as celulas a preencher a mao ficam a amarelo: e so nelas que se escreve
ENTRADA = PatternFill('solid', fgColor='FFF2CC')
FINA = Side(style='thin')
CENTRO = Alignment(horizontal='center')


def _formata(ws, n_linhas, inicio):
    for j, larg in enumerate([11, 13, 12, 14, 6, 12, 13, 11, 10, 15, 13],
                             start=1):
        ws.column_dimensions[get_column_letter(j)].width = larg
    for c in ws[inicio]:
        c.font = NEGRITO
        c.alignment = CENTRO
        c.border = Border(bottom=FINA, top=FINA)
    for linha in ws.iter_rows(min_row=inicio + 1, max_row=inicio + n_linhas,
                              min_col=3, max_col=11):
        for c in linha:
            c.number_format = '0.00' if c.column in (8, 11) else '0.0000'
            c.alignment = CENTRO


def folha(wb, ind, nome, rotulo, coeffs, sim, exp):
    ws = wb.create_sheet(nome)
    ws['A1'] = f'Validação de {rotulo}: equação ajustada nos dados simulados'
    ws['A1'].font = Font(bold=True, size=12)
    ws['A2'] = ('Q_L/Q0_est é o caudal de fuga do modelo (Qf dos rede_rp/'
                'rp_*.dat) sobre o Q0 do rp_NL; Q_L/Q0_real é o pesado na bancada')
    ws['A3'] = ('Erro (%) = (est − real)/real × 100.   As três últimas colunas '
                'fazem o mesmo com a equação de sizing: Q_L/Q0 = a × X_exp')
    ws.append([])
    ws.append(CAB)
    cab_linha = ws.max_row

    n = 0
    for loc in C.LOCS:
        a_loc = C.slope(coeffs, ind, loc)
        for leak, ql in C.QL_EXP[loc].items():
            g = exp[(exp['loc'] == loc) & (exp['leak'] == leak)]
            s = sim[(sim['loc'] == loc) & (sim['leak'] == leak)]
            if g.empty:
                continue
            r = ws.max_row + 1
            ql_mod = B.QL_MODEL.get((loc, f'{float(leak):.1f}'))
            ws.append([loc, f'L{leak}',
                       float(s[ind].iloc[0]) if not s.empty else None,
                       float(g[ind].mean()), len(g),
                       (ql_mod / 3.6e6) / B.Q0_NOMINAL if ql_mod else None,
                       ql / C.Q0_EXP,
                       f'=100*(F{r}-G{r})/G{r}',
                       a_loc, f'=I{r}*D{r}', f'=100*(J{r}-G{r})/G{r}'])
            n += 1
    for r in range(cab_linha + 1, cab_linha + n + 1):
        ws.cell(r, 7).fill = ENTRADA        # Q_L/Q0_real: preencher a mao
    _formata(ws, n, cab_linha)
    fim = cab_linha + n
    ws.append([])
    i0, i1 = cab_linha + 1, fim
    cal = f'B{i0}:B{i1}'
    for rotulo, cond in (('Erro absoluto médio (%)', None),
                         ('Erro absoluto médio sem L0.5 (%)', True)):
        linha = ['', rotulo, '', '', '', '', '', None, '', '', None]
        for col, letra in ((8, 'H'), (11, 'K')):
            rng = f'{letra}{i0}:{letra}{i1}'
            linha[col - 1] = (
                f'=SUMPRODUCT(ABS({rng}))/COUNT({rng})' if cond is None else
                f'=SUMPRODUCT(({cal}<>"L0.5")*ABS({rng}))/'
                f'SUMPRODUCT(--({cal}<>"L0.5"))')
        ws.append(linha)
        ws.cell(ws.max_row, 2).font = NEGRITO
        for col in (8, 11):
            ws.cell(ws.max_row, col).number_format = '0.00'
    return ws


def folha_coeficientes(wb, coeffs):
    ws = wb.create_sheet('Coeficientes', 0)
    ws['A1'] = 'Coeficientes ajustados nos dados simulados (Q_L/Q0 = a × X)'
    ws['A1'].font = Font(bold=True, size=12)
    ws.append([])
    ws.append(['Indicador', 'Grupo', 'a', 'R²', 'n', 'Equação'])
    for c in ws[3]:
        c.font = NEGRITO
    for ind, _nome, rotulo in METODOS:
        sub = coeffs[coeffs['indicator'] == ind]
        for _, r in sub.iterrows():
            ws.append([rotulo, r['group'], float(r['slope']), float(r['r2']),
                       int(r['n']), r['equation']])
    for j, larg in enumerate([14, 16, 10, 9, 6, 34], start=1):
        ws.column_dimensions[get_column_letter(j)].width = larg
    for linha in ws.iter_rows(min_row=4, min_col=3, max_col=4):
        for c in linha:
            c.number_format = '0.0000'
    return ws


# A fuga da CC esta a x/L = 0.629, entre a LOC2 e a LOC3 da CS; o coeficiente
# vem da interpolacao de a(x/L) nos tres pontos simulados.
CC_CSV = os.path.join('Experimental_data_CC', 'pair_single.csv')
CC_A = {'dH_over_dHJ': 3.824, 'E_R_over_E0_sqrt': 4.035, 'zeta_leak': 18.874}
CC_CAB = ['Flow (l/h)', 'Identification', 'X_exp', 'a (eq.)',
          'Q_L/Q0 pela eq.', 'Q_L/Q0_real', 'Error (%)']


def folha_cc(wb, ind, nome, rotulo):
    """A mesma validacao na CC, com a coluna do Q_L/Q0 real por preencher."""
    cc = pd.read_csv(CC_CSV, dtype={'loc': str, 'leak': str})
    cc = cc[(cc['maneuver'] == 'AUT') & (~cc['loc'].str.contains(r'\+'))]
    if ind == 'E_R_over_E0_sqrt':
        cc = cc.assign(**{ind: np.sqrt(cc['E_R_over_E0'].clip(lower=0))})
    ws = wb.create_sheet(f'CC_{nome}')
    ws['A1'] = f'CC: {rotulo} com a equação da CS (x/L = 0.629)'
    ws['A1'].font = Font(bold=True, size=12)
    ws['A2'] = ('Preencher só a coluna Q_L/Q0_real (a amarelo); o erro é '
                'calculado automaticamente')
    ws['A3'] = f'a = {CC_A[ind]:.3f}, interpolado entre a LOC2 e a LOC3 da CS'
    ws.append([])
    ws.append(CC_CAB)
    cab_linha = ws.max_row
    n = 0
    for _, r in cc.sort_values(['flow', 'leak']).iterrows():
        rr = ws.max_row + 1
        ws.append([int(r['flow']), f"L{r['leak']}", float(r[ind]), CC_A[ind],
                   f'=D{rr}*C{rr}', None,
                   f'=IF(F{rr}="","",100*(E{rr}-F{rr})/F{rr})'])
        ws.cell(rr, 6).fill = ENTRADA
        n += 1
    for j, larg in enumerate([11, 13, 12, 10, 15, 13, 12], start=1):
        ws.column_dimensions[get_column_letter(j)].width = larg
    for c in ws[cab_linha]:
        c.font = NEGRITO
        c.alignment = CENTRO
        c.border = Border(bottom=FINA, top=FINA)
    for linha in ws.iter_rows(min_row=cab_linha + 1, max_row=cab_linha + n,
                              min_col=3, max_col=7):
        for c in linha:
            c.number_format = '0.00' if c.column == 7 else '0.0000'
            c.alignment = CENTRO
    i0, i1 = cab_linha + 1, cab_linha + n
    ws.append([])
    ws.append(['', 'Erro absoluto médio (%)', '', '', '', '',
               f'=SUMPRODUCT(ABS(N(G{i0}:G{i1})))/MAX(1,COUNT(G{i0}:G{i1}))'])
    ws.cell(ws.max_row, 2).font = NEGRITO
    ws.cell(ws.max_row, 7).number_format = '0.00'
    return ws


def main():
    coeffs = C.calibrate()
    sim = C.load_single(C.SIM_CSV)
    exp = C.load_single(C.EXP_CSV)
    wb = Workbook()
    wb.remove(wb.active)
    folha_coeficientes(wb, coeffs)
    for ind, nome, rotulo in METODOS:
        folha(wb, ind, nome, rotulo, coeffs, sim, exp)
        folha_cc(wb, ind, nome, rotulo)
        v = C.validate(coeffs, exp)
        v = v[v['indicator'] == ind]
        print(f'{rotulo:12} erro absoluto medio por LOC: '
              f'{v["err_loc"].abs().mean():6.2f} %   '
              f'(global {v["err_global"].abs().mean():6.2f} %)')
    wb.save(SAIDA)
    print(f'\nEscrito: {os.path.abspath(SAIDA)}')


if __name__ == '__main__':
    main()
