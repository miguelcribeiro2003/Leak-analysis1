# -*- coding: utf-8 -*-
"""As seis verificacoes de uma paleta, portadas do validate_palette.js.

Delta E e a distancia euclidiana em OKLab x100; a simulacao de daltonismo e a
de Machado-Oliveira-Fernandes (2009) com severidade 1.0, que e o modelo a que
os limiares estao calibrados.
"""
import itertools
import math

M_PROT = ((0.152286, 1.052583, -0.204868),
          (0.114503, 0.786281, 0.099216),
          (-0.003882, -0.048116, 1.051998))
M_DEUT = ((0.367322, 0.860646, -0.227968),
          (0.280085, 0.672501, 0.047413),
          (-0.011820, 0.042940, 0.968881))


def _srgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _oklab(rgb):
    r, g, b = (_lin(c) for c in rgb)
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l, m, s = (math.copysign(abs(v) ** (1/3), v) for v in (l, m, s))
    return (0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s)


def oklch(hexa):
    L, a, b = _oklab(_srgb(hexa))
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _cvd(rgb, M):
    r, g, b = (_lin(c) for c in rgb)
    out = []
    for row in M:
        v = row[0]*r + row[1]*g + row[2]*b
        v = max(0.0, min(1.0, v))
        out.append(1.055 * v ** (1/2.4) - 0.055 if v > 0.0031308 else 12.92 * v)
    return tuple(out)


def dE(h1, h2, M=None):
    a, b = _srgb(h1), _srgb(h2)
    if M:
        a, b = _cvd(a, M), _cvd(b, M)
    p, q = _oklab(a), _oklab(b)
    return 100 * math.dist(p, q)


def _lum(hexa):
    r, g, b = (_lin(c) for c in _srgb(hexa))
    return 0.2126*r + 0.7152*g + 0.0722*b


def contraste(hexa, fundo='#ffffff'):
    a, b = sorted((_lum(hexa), _lum(fundo)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def valida(cores, nome='', pares='adjacentes', fundo='#ffffff'):
    print(f'\n=== {nome} ===')
    print(f'{"cor":10}{"L":>7}{"C":>7}{"contraste":>11}')
    for c in cores:
        L, C, _h = oklch(c)
        aviso = ''
        if not 0.43 <= L <= 0.77:
            aviso += '  L fora da banda 0.43-0.77'
        if C < 0.10:
            aviso += '  C abaixo de 0.10'
        if contraste(c, fundo) < 3.0:
            aviso += '  contraste < 3:1'
        print(f'{c:10}{L:7.3f}{C:7.3f}{contraste(c, fundo):11.2f}{aviso}')
    if pares == 'todos':
        ps = list(itertools.combinations(range(len(cores)), 2))
    else:
        ps = [(i, i+1) for i in range(len(cores)-1)]
    piores = []
    for i, j in ps:
        piores.append((min(dE(cores[i], cores[j], M_PROT),
                           dE(cores[i], cores[j], M_DEUT)),
                       dE(cores[i], cores[j]), cores[i], cores[j]))
    piores.sort()
    pior_cvd, pior_norm = piores[0][0], min(p[1] for p in piores)
    print(f'  pior par CVD ({pares}): dE {pior_cvd:5.1f}  '
          f'[{piores[0][2]} vs {piores[0][3]}]   alvo >= 8, minimo 6')
    print(f'  pior par visao normal:  dE {pior_norm:5.1f}                          '
          f'      minimo 15')
    veredicto = 'PASSA' if (pior_cvd >= 8 and pior_norm >= 15) else (
        'LIMITE (6-8, exige segunda codificacao)' if pior_cvd >= 6 and pior_norm >= 15
        else 'FALHA')
    print(f'  -> {veredicto}')
    return pior_cvd, pior_norm
