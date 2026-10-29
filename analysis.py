"""Core analysis for the water-hammer leak detection notebooks (V7).

All computation lives here: pair discovery/loading, maneuver detection,
vertical/horizontal alignment, wave celerities, leak location, residual
energy, damping, and the per-test exports. The notebook keeps only
configuration, the run calls, and the plots.

Configuration: module-level knobs below are defaults. The notebook overrides
them once per session via `configure(...)`, which also recomputes the derived
quantities (ONSET_BASE_END, S_AREA, A_CEILING).
"""
import configparser
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════════════
# Configuration (override from the notebook via configure())
# ══════════════════════════════════════════════════════════════════════════

TEST_RE = None            # optional test-name filter (compiled regex) for list_tests

# ── Signal treatment ──
MA_WINDOW = 9

# ── Alignment knobs ──
PRE_ONSET_MARGIN    = 0.005   # vertical-baseline window ends this long before earliest onset (s)
PAD_S               = 0.001   # horizontal correlation window padding around onset/offset (s)
SEARCH_S            = 0.008   # +/- refine margin around the onset-anchored position (s)
PRE_BASE_S          = 0.010   # flat pre-onset baseline included in the correlation template (s)
PRE_MANEUVER_MARGIN = 0.010   # after clock shift, earliest onset is placed here (s)
REF_CHANNEL         = 'PT1'
ONSET_FREE_MARGIN   = 0.005   # the onset-free (zero-offset) candidate must beat the
#                               onset-anchored one by this much. Everything else in
#                               the alignment hangs off the detected onset, so when
#                               that detection is wrong the search never visits the
#                               right place; this gives it one candidate that does
#                               not depend on the onset at all. Safe because these
#                               acquisitions are pre-triggered with the maneuver
#                               near t=0, so a zero offset is a sound prior.
PERIOD_TEST_MARGIN  = 0.005   # +/- period candidate must beat the onset-anchored
#                               candidate's correlation by this much to be accepted
#                               (leak echoes can make the wrong period win marginally)
TEMPLATE_RISE_S     = None    # optionally cap how much of the rise enters the
#                               correlation template (s). None -> full incident front.
#                               (Short caps make the template mostly flat baseline and
#                               the correlation ambiguous - keep None unless testing.)

# ── Fluid / pipe properties ──
K, rho   = 2.19e9, 1000
D, e, E  = 0.02, 0.001, 1.20e11
ALPHA    = 0.976   # Wylie & Streeter, pipe fixed throughout, D/e < 25, mu'=0.34 (copper)
L_pipe   = 15.0
DX_AB    = 7.625
G        = 9.81
Q_UNIT_TO_M3S = 1 / 3.6e6        # txt Q column unit -> m3/s (facility 1: L/h)
USECOLS = (0, 1, 2, 3)           # txt columns to read: time, PT1, PT2, Q
A_THEO = None                    # reference/measured wave celerity (m/s) from the pipe
#                                  config; when None, a_theoretical() computes it from
#                                  the material properties (e, E, ALPHA) instead.

# ── Leak location ──
LOC_POSITIONS = {'LOC1': 2.060, 'LOC2': 7.605, 'LOC3': 10.260}
PT1_OFFSET      = 0.245
LEAK_WIN_FRAC   = 0.50
ECHO_EDGE_GUARD = 0.10      # no longer limits the echo pick (see _echo_picks); kept so
#                             existing configure() calls stay valid
ECHO_PROM_FRAC  = 0.02
ECHO_WEAK_MIN_FRAC = 0.002  # abaixo desta proeminencia (fraccao de |v1|) um minimo
#                             local NAO e um vale: e a ondulacao de ~0.1 % que
#                             aparece no fundo de uma descida e que roubava a escolha
#                             ao joelho que esta mesmo ao lado. Descartados, a escolha
#                             cai na mudanca de declive. A 0.2 % nenhum ensaio de fuga
#                             unica muda (CS e CC) e o eco da LOC3 nos pares
#                             LOC1+LOC3 passa de 10.73 m (forca 0.09 %) para 10.25 m
#                             (2.60 %), contra 10.26 m reais.
ECHO_PICK       = 'deepest' # WHICH dip among those above the prominence bar is
#                             taken as the leak echo:
#                             'deepest'   : the most negative dH/dt (default, and
#                                           the same rule on every facility);
#                             'first'     : the earliest;
#                             'prominent' : the one standing out most from its
#                                           neighbours.
#                             Measured on the whole campaign, changing ONLY this
#                             rule and leaving the window and the edge guard as
#                             they are:
#                               CS  the three rules are IDENTICAL in all 54
#                                   single-leak tests -- the edge guard leaves
#                                   a single candidate in the window almost
#                                   every time, and with one candidate there is
#                                   nothing to choose.
#                               CC  'deepest' 9.76 m vs 'prominent' 15.80 m and
#                                   'first' 46.57 m. It improves at every flow
#                                   and every leak size, 13 tests better and 3
#                                   worse, and takes the tests within 10 m of
#                                   the leak from 48 to 61 of 72.
#                             Hence one rule for both pipes: neutral where it
#                             does not matter, clearly best where it does.
#                             Depth rather than prominence because a leak echo
#                             arriving on an already-falling stretch is deep but
#                             barely prominent.
ECHO_ON         = 'leak'    # WHICH signal the echo dips are searched on:
#                             'leak': dH/dt of the leak test (classic, default);
#                             'diff': dH/dt of (leak - NL), so features the system
#                             produces on its own (fittings, maneuver tail, the
#                             reservoir reflection) cancel and only what the leak
#                             ADDS remains. Needs the NL reference: pass NL_sig to
#                             locate(); without it the search falls back to 'leak'.
#                             t1 is always taken on the leak test itself.
ECHO_MIN_DT     = 0.0       # (s) dips closer than this to t1 are never taken as
#                             the echo. 0 = off (default). A fast valve closure
#                             rings right after the incident front, and on the CC
#                             AUT tests that ringing dip (delta_t ~6-7 ms, present
#                             in the NL too, so not a leak feature) is deeper than
#                             the true echo at ~107 ms: the 'deepest' rule took it
#                             in 10 of 12 AUT tests (x ~ 4 m). A guard of 15 ms
#                             fixes all but one and changes no MAN test. It must
#                             stay per pipe: on the CS the LOC1 echo arrives at
#                             ~3.3 ms, which any such guard would erase. Set via
#                             [processing] echo_min_dt.

# ── Closing valve per maneuver (effective length for M1) ──
VALVE_POS = {}              # position of the valve that closes on each maneuver,
#                             in m from the PAV (positive towards the HPV), keyed
#                             by maneuver code (e.g. {'MAN': -0.790}). Missing
#                             maneuver = the PAV itself (0). Only a_period uses it:
#                             the oscillation spans valve -> HPV, so
#                             a = 4 * (L - pos) / T. The PT1 delta_t is unaffected
#                             as long as the valve is on the PAV side of PT1. Set
#                             via the optional [valves] section of the pipe config.

# ── Transducer-celerity (Method 2) front-timing criterion ──
TDX_TIMING = 'frac_peak'   # how each sensor's front is timed for the time-of-flight:
#                            'frac_peak' -> TDX_FRAC of that sensor's own rise (default);
#                            'absolute'  -> a fixed rise TDX_LEVEL (m) above its baseline,
#                                           i.e. the same physical feature at both sensors.
TDX_FRAC  = 0.5            # fraction of each sensor's own rise ('frac_peak' mode)
TDX_LEVEL = 5.0            # absolute rise in m ('absolute' mode)

# ── Which celerity sets the oscillation period T = 4L/a (see period_time) ──
PERIOD_FROM = 'period'     # 'period' (M1, from the oscillation itself; default),
#                            'transducer' (M2, time of flight PT1->PT2),
#                            'joukowsky' (M3), or 'theoretical' (Korteweg).
#                            Which one is trustworthy is a property of the pipe,
#                            so it belongs in the pipe config, not in the code.
_PERIOD_SOURCES = {'period': 'M1_period', 'transducer': 'M2_transducer',
                   'joukowsky': 'M3_Joukowsky', 'theoretical': 'Theoretical'}

# ── Residual energy / damping ──
FIRST_SURGE_FRAC = 0.5
DROP_FROM = 'onset'       # onde comeca a procura da queda inicial no damping_NL,
#                           por [processing] drop_from: 'onset' procura o maximo
#                           de dH em toda a janela da primeira sobrepressao (desde
#                           o arranque da manobra), 'echo' so a partir do primeiro
#                           maximo local do ensaio SEM FUGA depois de o eco da fuga
#                           chegar ao PT1, t_on + 2 (x_fuga - X_PT1) / a.
#                           Porque existe: na CS a janela toda esta limpa -- as duas
#                           curvas andam coladas ate a queda e o maximo cai sempre
#                           20.8 a 24.5 ms depois do fecho. Na CC nao: a serpentina
#                           devolve reflexoes de geometria muito antes do eco da
#                           fuga, e entre os ~105 e os ~112 ms ha um mergulho
#                           transitorio das curvas com fuga onde o maximo das fugas
#                           pequenas estava a ser apanhado. Dava um indicador NAO
#                           MONOTONO no calibre: a 150 l/h o L0.5 lia 0.2609, acima
#                           do L1.5 (0.2107), e o L1.0 a 350 lia 0.1394 contra os
#                           0.1773 do L1.5. Com 'echo' o maximo passa para a
#                           separacao permanente (131 a 199 ms) e os tres caudais
#                           ficam monotonos: 0.084/0.099/0.211/0.294 a 150,
#                           0.088/0.124/0.212/0.296 a 250 e 0.050/0.091/0.177/0.246
#                           a 350. So mexe nas fugas pequenas -- o L1.5 e o L2.0 ja
#                           estavam depois do eco e nao se alteram.
#                           O instante nao e critico: com a LOC1 nos 65.512 m do
#                           levantamento de 2026-09-17 o eco chega aos 124-128 ms e
#                           o pico seguinte esta nos 130-132 ms; com os 54.0 m
#                           antigos o eco da 108 ms e o pico 115-117, e so o L0.5 a
#                           150 l/h muda (0.0844 contra 0.0868).
DAMP_PROM_FRAC = 0.05     # peak/trough prominence as a fraction of the post-onset span
DAMP_SNR_MIN   = 8.0      # drop the noise tail: keep cycles whose amplitude exceeds
#                           this many standard deviations of the PRE-MANEUVER
#                           baseline OF THE SAME TEST (see _baseline_noise).
#                           Anchored to the measured noise, not to a fraction of
#                           A0, because the noise varies a lot between tests: on
#                           the CS the baseline sd of H* is 0.0130 at 150 l/h and
#                           0.0040 at 850 l/h, so a fixed fraction would cut the
#                           clean records at the same place as the noisy ones.
#                           The previous rule (amplitudes above 3% of A0) never
#                           fired: in 294 series across both facilities it
#                           truncated ZERO of them, because the prominence test
#                           already ends the series when the amplitude reaches
#                           ~5 sd (measured range 3.7 to 13.3). The cut is
#                           therefore a real one, not a rewrite of an equivalent.
#                           Why 8: every undetected cycle in the CS campaign sits
#                           below 6.2 sd, so above that threshold no oscillation
#                           escapes and the cycle number is simply the index of
#                           each amplitude (see amplitude_cycles). Below 10 sd no
#                           test drops under 3 usable cycles. 8 sits between the
#                           two, with margin on both sides. Measured on the CS
#                           (75 pairs) against the previous full-tail fit:
#                             median CV of zeta_leak  3.18% -> 2.32%
#                             R2 of the ln-A fits     0.9978 -> 0.9986 (leak)
#                                                     0.9989 -> 0.9992 (NL)
#                             R2 of zeta_leak -> ratio_Q, global  0.7424 -> 0.7412
#                           i.e. the gain is in repeatability and in the honesty
#                           of the fit, not in the calibration, which is flat.
DAMP_NOISE_MIN_N = 10     # samples required in the baseline window to trust the
#                           noise estimate. Below this the amplitude floor is
#                           skipped (and a note is printed): simulated records
#                           start a few ms before the maneuver and have no
#                           baseline to measure, and there is no noise tail to cut.

# ── Head losses / leak flow ──
CD    = 0.6        # discharge coefficient (-)
MAX_PT_DROP = 3.0  # m. |PT1 - PT2| em permanente acima do qual o PT2 nao esta a
#                    medir e H_L passa a ser o proprio PT1 (com aviso e coluna
#                    HL_from no CSV). O limiar esta no meio de um vazio: a maior
#                    diferenca COERENTE medida e 1.20 m (CC a 350 l/h, que e o
#                    atrito de 51.7 m de tubo, 0.45 m em recto, mais um desvio
#                    fixo de ~0.6 m entre os dois transdutores, visivel porque a
#                    queda cresce com o caudal: 0.71 m a 150, 0.95 a 250, 1.03 a
#                    350). As leituras avariadas estao 8.9 a 10.9 m, sete vezes
#                    acima: tres ensaios da CS (LOC3_L1.0_test01, LOC3_L2.0_test01
#                    e test02) onde a mesma combinacao da 0.11 m nas repeticoes.
#                    Importa porque a LOC3 esta alem do PT2 e a interpolacao passa
#                    a EXTRAPOLACAO (frac = 1.31), amplificando o erro em 31 %:
#                    dava H_L = 28.9 m em vez de ~43 m e QL 22 % abaixo.
HL_FROM = 'interp' # como se obtem a carga estatica na fuga, H_L, por [processing]
#                    hl_from: 'interp' interpola entre PT1 e PT2 (tubo recto e
#                    horizontal, o caso da CS), 'PT1' usa so o PT1. Na CC o tubo e
#                    uma serpentina: a cota varia ao longo do tubo e nao de forma
#                    monotona, sobe e desce a cada volta, por isso a variacao entre
#                    PT1 e PT2 nao e linear e a fuga (65.5 m) esta alem do PT2
#                    (52.0 m), o que tornava aquilo uma extrapolacao de um gradiente
#                    que nem sequer e linear. A queda medida la, -0.71 m a 150 l/h,
#                    -0.95 a 250 e -1.03 a 350, decompoe-se em ~0.6 m fixos (cota
#                    entre os dois pontos, mais eventual desvio de calibracao) e o
#                    atrito, e o sinal diz que o PT2 esta a montante. Nao muda
#                    nenhum resultado publicado da CC: H_L so entra no QL, o QL so
#                    no ratio_Q e esse ja sai n/a por flow_valid = false.
X_PT1 = 0.245      # PT1 position relative to the PABV (m)
X_PT2 = 7.870      # PT2 position relative to the PABV (m)   (X_PT2 - X_PT1 = DX_AB)
AL_OVERRIDE = None # set a leak area (m^2) by hand for non-circular orifices

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# Derived (recomputed by _derive(); do not set these via configure()).
_DERIVED = ('ONSET_BASE_END', 'S_AREA', 'A_CEILING')


def _derive():
    global ONSET_BASE_END, S_AREA, A_CEILING
    # Pre-maneuver window for the onset threshold, on the analysis clock. The
    # steady-state head H0 has its own window, see _base_mask.
    ONSET_BASE_END   = PRE_MANEUVER_MARGIN - 0.01
    S_AREA    = np.pi * D**2 / 4.0
    A_CEILING = np.sqrt(K / rho)


def configure(**overrides):
    """Set analysis knobs from the notebook and recompute derived quantities.

    Unknown names raise (catches typos); derived names raise (set the parent
    knob, e.g. PRE_MANEUVER_MARGIN instead of ONSET_BASE_END).
    """
    g = globals()
    for k, v in overrides.items():
        if k in _DERIVED:
            raise KeyError(f'{k!r} is derived - set its parent knob instead.')
        if k not in g or k.startswith('_') or callable(g[k]):
            raise KeyError(f'Unknown analysis setting: {k!r}')
        g[k] = v
    _derive()


# ══════════════════════════════════════════════════════════════════════════
# Pipe / facility profiles  (externalised to pipe_configs/*.txt)
# ══════════════════════════════════════════════════════════════════════════
# Pipe geometry/material/sensor layout is no longer hardcoded here: each pipe
# is one human-readable INI text file in pipe_configs/ (*.txt). Add a new pipe
# by copying pipe_configs/_template.txt — no edit to this module is needed. The
# signal-processing knobs (MA_WINDOW, SEARCH_S, ...) are unchanged and still
# passed by each notebook via configure_pipe(FACILITY, <knobs>).

PIPE_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipe_configs')

# Accepted flow-rate units for the txt Q column -> factor to convert to m3/s.
_FLOW_UNITS = {
    'l/h':  1 / 3.6e6,
    'kl/h': 1 / 3600.0,     # kl/h = m3/h
    'm3/h': 1 / 3600.0,
    'l/s':  1 / 1000.0,
    'm3/s': 1.0,
}

_NAME_RE = re.compile(r'^[A-Za-z0-9_]+$')


def _cfg_err(path, msg):
    raise ValueError(f'Pipe config {os.path.basename(path)}: {msg}')


def load_pipe_config(path):
    """Load a pipe INI text file into the internal dict consumed by configure_pipe.

    Returns a dict of module knobs (L_pipe, D, A_THEO, sensor/leak positions,
    flow-unit factor, txt column layout, ...) equivalent to the profiles that
    used to be hardcoded here. Raises ValueError with a clear, file-scoped
    message on any missing or invalid field.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',), interpolation=None)
    cfg.optionxform = str                              # preserve case of PT1/LOC1/... keys
    try:
        with open(path, encoding='utf-8') as fh:
            cfg.read_file(fh)
    except configparser.Error as exc:                  # duplicate keys, bad syntax, ...
        _cfg_err(path, str(exc).replace('\n', ' '))

    def _num(section, key, kind):
        try:
            return float(cfg[section][key])
        except KeyError:
            _cfg_err(path, f"missing required field '[{section}] {key}'.")
        except ValueError:
            _cfg_err(path, f"[{section}] {key} must be a number (got {cfg[section][key]!r}).")

    # ── required: [pipe] name, [geometry] L/D/celerity ──
    if not cfg.has_option('pipe', 'name') or not cfg['pipe']['name'].strip():
        _cfg_err(path, "missing required field '[pipe] name'.")
    name = cfg['pipe']['name'].strip()
    L = _num('geometry', 'L', 'length')
    D = _num('geometry', 'D', 'diameter')
    if not L > 0: _cfg_err(path, f'[geometry] L must be > 0 (got {L}).')
    if not D > 0: _cfg_err(path, f'[geometry] D must be > 0 (got {D}).')

    # ── fluid (optional; defaults = water at ~20 C) ──
    K_   = _num('fluid', 'K', 'K') if cfg.has_option('fluid', 'K') else 2.19e9
    rho_ = _num('fluid', 'rho', 'rho') if cfg.has_option('fluid', 'rho') else 1000.0
    g_   = _num('fluid', 'g', 'g') if cfg.has_option('fluid', 'g') else 9.81

    # ── celerity: given in [geometry], OR computed (Korteweg) from [material] ──
    if cfg.has_option('geometry', 'celerity'):
        cel = _num('geometry', 'celerity', 'celerity')
        if not cel > 0: _cfg_err(path, f'[geometry] celerity must be > 0 (got {cel}).')
    elif cfg.has_section('material'):
        e_ = _num('material', 'e', 'wall thickness')
        E_ = _num('material', 'E', "Young's modulus")
        if not (e_ > 0 and E_ > 0):
            _cfg_err(path, '[material] e and E must be > 0.')
        if cfg.has_option('material', 'alpha'):
            alpha_ = _num('material', 'alpha', 'alpha')
        elif cfg.has_option('material', 'poisson'):
            mu = _num('material', 'poisson', 'poisson')   # Wylie & Streeter, pipe fixed throughout
            alpha_ = 2 * e_ / D * (1 + mu) + D * (1 - mu ** 2) / (D + e_)
        else:
            _cfg_err(path, "[material] needs 'alpha' or 'poisson' to compute the celerity.")
        cel = float(np.sqrt(K_ / rho_) / np.sqrt(1 + alpha_ * (K_ * D) / (e_ * E_)))  # Korteweg
    else:
        _cfg_err(path, "provide [geometry] celerity, or a [material] section to compute it.")

    if not cfg.has_section('sensors') or not cfg['sensors']:
        _cfg_err(path, 'at least one sensor is required under [sensors].')
    # [leak_locations] is OPTIONAL: a config for blind analysis describes the
    # pipe but not where the leaks are, because in the field that is unknown.
    # Validation code (batch_analysis, V7, V6) does need them and will fail on
    # a config without them, which is the intended signal.

    def _positions(section, kind):
        out = {}
        for k, v in cfg[section].items():
            if not _NAME_RE.match(k):
                _cfg_err(path, f"{kind} name {k!r} has spaces/special chars "
                               "(use only letters, digits, underscore).")
            try:
                p = float(v)
            except (TypeError, ValueError):
                _cfg_err(path, f'{kind} {k!r} position must be a number (got {v!r}).')
            if p < 0:
                _cfg_err(path, f'{kind} {k!r} position {p} m is negative.')
            out[k] = p
        return out

    sens = _positions('sensors', 'sensor')
    locs = (_positions('leak_locations', 'leak_location')
            if cfg.has_section('leak_locations') else {})
    for nm, p in sens.items():
        if p > L:
            _cfg_err(path, f'sensor {nm!r} at {p} m is beyond pipe length L ({L} m).')
    for nm, p in locs.items():
        if p > L:
            print(f"[warn] {os.path.basename(path)}: leak_location {nm} ({p} m) is beyond "
                  f"pipe length L ({L} m) - fine if this is an intentional extrapolation, "
                  "as documented in the methodology.")

    # ── sensor layout -> internal knobs (file order preserved; PT1 = first) ──
    ordered = list(sens.items())
    (ref_name, ref_pos) = ordered[0]
    second_pos = ordered[1][1] if len(ordered) > 1 else ref_pos

    # ── data format (optional; defaults = facility-1 conventions) ──
    unit_raw = cfg['data_format']['flow_unit'] if cfg.has_option('data_format', 'flow_unit') else 'L/h'
    unit = unit_raw.lower()
    if unit not in _FLOW_UNITS:
        _cfg_err(path, f"unknown flow_unit {unit_raw!r}; options: {sorted(_FLOW_UNITS)}.")
    try:
        flow_valid = (cfg.getboolean('data_format', 'flow_valid')
                      if cfg.has_option('data_format', 'flow_valid') else True)
    except ValueError:
        _cfg_err(path, '[data_format] flow_valid must be true or false.')
    if cfg.has_section('columns'):
        try:
            usecols = (int(cfg['columns']['time']), int(cfg['columns']['PT1']),
                       int(cfg['columns']['PT2']), int(cfg['columns']['Q']))
        except (KeyError, ValueError):
            _cfg_err(path, '[columns] must map time/PT1/PT2/Q to integer column indices.')
    else:
        usecols = (0, 1, 2, 3)

    # ── processing (optional) ──
    echo_pick = (cfg['processing']['echo_pick'] if cfg.has_option('processing', 'echo_pick')
                 else 'first')
    if echo_pick not in ('first', 'prominent', 'deepest'):
        _cfg_err(path, f"[processing] echo_pick must be 'first', 'prominent' or "
                       f"'deepest' (got {echo_pick!r}).")
    timing = (cfg['processing']['timing'] if cfg.has_option('processing', 'timing')
              else 'frac_peak')
    if timing not in ('frac_peak', 'absolute'):
        _cfg_err(path, f"[processing] timing must be 'frac_peak' or 'absolute' (got {timing!r}).")
    tdx_level = _num('processing', 'level', 'level') if cfg.has_option('processing', 'level') else 5.0
    echo_min_dt = (_num('processing', 'echo_min_dt', 'echo_min_dt')
                   if cfg.has_option('processing', 'echo_min_dt') else 0.0)
    if echo_min_dt < 0:
        _cfg_err(path, f'[processing] echo_min_dt must be >= 0 s (got {echo_min_dt}).')

    # ── valves (optional): closing-valve position per maneuver, may be negative ──
    valves = {}
    if cfg.has_section('valves'):
        for k, v in cfg['valves'].items():
            if not _NAME_RE.match(k):
                _cfg_err(path, f"maneuver name {k!r} in [valves] has spaces/special chars.")
            try:
                valves[k] = float(v)
            except (TypeError, ValueError):
                _cfg_err(path, f'[valves] {k} must be a number in m (got {v!r}).')
    hl_from = (cfg['processing']['hl_from'] if cfg.has_option('processing', 'hl_from')
               else 'interp')
    if hl_from not in ('interp', 'PT1'):
        _cfg_err(path, f"[processing] hl_from must be 'interp' or 'PT1' (got {hl_from!r}).")
    period_from = (cfg['processing']['period_from']
                   if cfg.has_option('processing', 'period_from') else 'period')
    if period_from not in _PERIOD_SOURCES:
        _cfg_err(path, f"[processing] period_from must be one of "
                       f"{sorted(_PERIOD_SOURCES)} (got {period_from!r}).")
    drop_from = (cfg['processing']['drop_from']
                 if cfg.has_option('processing', 'drop_from') else 'onset')
    if drop_from not in ('onset', 'echo'):
        _cfg_err(path, f"[processing] drop_from must be 'onset' or 'echo' "
                       f"(got {drop_from!r}).")

    return dict(
        _name=str(name),
        L_pipe=L, D=D, A_THEO=cel,
        PT1_OFFSET=ref_pos, X_PT1=ref_pos, X_PT2=second_pos,
        DX_AB=second_pos - ref_pos,
        LOC_POSITIONS=locs,
        K=K_, rho=rho_, G=g_,
        Q_UNIT_TO_M3S=_FLOW_UNITS[unit], FLOW_VALID=flow_valid,
        USECOLS=usecols, ECHO_PICK=echo_pick,
        TDX_TIMING=timing, TDX_LEVEL=tdx_level, PERIOD_FROM=period_from,
        DROP_FROM=drop_from,
        # always returned (defaults included) so switching pipes in one process
        # never inherits the previous pipe's guard or valve layout
        ECHO_MIN_DT=echo_min_dt, VALVE_POS=valves, HL_FROM=hl_from,
    )


def _load_all_pipes():
    """Build {name: internal_dict} from every pipe_configs/*.txt (skips _*.txt).

    Raises ValueError if two files declare the same name.
    """
    pipes = {}
    if not os.path.isdir(PIPE_CONFIG_DIR):
        return pipes
    for path in sorted(glob(os.path.join(PIPE_CONFIG_DIR, '*.txt'))):
        if os.path.basename(path).startswith('_'):
            continue
        cfg = load_pipe_config(path)
        name = cfg.pop('_name')
        if name in pipes:
            raise ValueError(f"Duplicate pipe name {name!r} in pipe_configs/ "
                             f"(also declared before {os.path.basename(path)}).")
        pipes[name] = cfg
    return pipes


try:
    PIPES = _load_all_pipes()          # available pipes at import (for introspection)
except Exception as _e:                # never let a bad config break `import analysis`
    print(f'[warn] pipe configs not loaded at import: {_e}')
    PIPES = {}


def configure_pipe(pipe, **overrides):
    """Apply a pipe profile (from pipe_configs/*.txt), then extra knob overrides.

    The directory is re-scanned on every call, so a new config file is picked up
    without re-importing analysis (e.g. from a running notebook kernel).
    """
    global PIPES
    PIPES = _load_all_pipes()
    if pipe not in PIPES:
        raise KeyError(f'Unknown pipe {pipe!r}; available: {sorted(PIPES)} '
                       f'(from {PIPE_CONFIG_DIR}).')
    configure(**{**PIPES[pipe], **overrides})


_derive()


# ══════════════════════════════════════════════════════════════════════════
# Pair discovery and loading
# ══════════════════════════════════════════════════════════════════════════

def parse_filename(fn):
    """Attributes parsed from a data filename (flat dataset folders).

    Convention: KIND_FACILITY_MANEUVER_FLOW_(NL | LOCi_L<size> ...)_[test??]_DATE
      EXP_CS_AUT_150_LOC2_L0.5_test01_20260511  -> one leak, test 1
      EXP_CS_AUT_150_NL_test01_20260511         -> no-leak reference
      EXP_CS_AUT_250_LOC1_L1.0_LOC2_L1.0_test03_20260504  -> two leaks
      SIM_CS_AUT_250_LOC1_L0.5_20260818          -> simulated, no test
    Returns dict(kind, facility, maneuver, flow, leaks[(loc,size)...], test,
    date, is_nl), or None when the name does not follow the convention —
    report files written into a dataset folder (e.g. the accuracy summary)
    must not break the callers that scan the directory.
    """
    tok = os.path.splitext(fn)[0].split('_')
    if len(tok) < 4 or not tok[3].isdigit():
        return None
    d = dict(kind=tok[0], facility=tok[1], maneuver=tok[2], flow=int(tok[3]),
             leaks=[], test=None, date=None, is_nl=False)
    i = 4
    while i < len(tok):
        t = tok[i]
        if t == 'NL':
            d['is_nl'] = True; i += 1
        elif (t.startswith('LOC') and i + 1 < len(tok)
              and tok[i + 1][:1] == 'L' and tok[i + 1][1:2].isdigit()):
            d['leaks'].append((t, tok[i + 1][1:])); i += 2
        elif re.fullmatch(r'(test|trial)\d+', t):   # 'trial' = names before 2026-09-14
            d['test'] = int(re.sub(r'\D', '', t)); i += 1
        elif len(t) == 8 and t.isdigit():
            d['date'] = t; i += 1
        else:
            i += 1
    return d


def normalize_leaks(loc, leak):
    """[(loc, size), ...] from the notebook's LOC/LEAK config values.

    Scalars give a one-leak list; parallel lists give one entry per leak:
      normalize_leaks('LOC2', '1.5')                    -> [('LOC2', '1.5')]
      normalize_leaks(['LOC1', 'LOC2'], ['1.0', '1.0']) -> two leaks
    """
    if isinstance(loc, (list, tuple)):
        sizes = list(leak) if isinstance(leak, (list, tuple)) else [leak] * len(loc)
        return [(l, str(s)) for l, s in zip(loc, sizes)]
    return [(loc, str(leak))]


def list_tests(root, maneuver, flow, leaks=None, is_nl=False):
    """Data files in a flat dataset folder matching the selection, test-sorted.

    `leaks` is a list of (loc, size) tuples (ignored when is_nl=True). NL files
    are matched by the SAME maneuver/flow so the reference has the same transient.
    """
    want = None if is_nl else [(l, str(s)) for l, s in leaks]
    out = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(('.txt', '.dat')):
            continue
        a = parse_filename(fn)
        if a is None:                       # not a data file (e.g. a report)
            continue
        if a['maneuver'] != maneuver or a['flow'] != int(flow):
            continue
        if TEST_RE is not None and not TEST_RE.search(fn):
            continue
        if is_nl and a['is_nl']:
            out.append(fn)
        elif (not is_nl) and (not a['is_nl']) and a['leaks'] == want:
            out.append(fn)
    return sorted(out, key=lambda f: (parse_filename(f)['test'] or 0))


def _pick(files, sel):
    if not files:
        raise RuntimeError('No data file found with the current filter.')
    if isinstance(sel, int):
        if sel < 1 or sel > len(files):
            raise IndexError(f'Test index {sel} out of range 1..{len(files)}')
        return files[sel - 1]                       # 1-based index
    hit = [f for f in files if str(sel) in f]
    if not hit:
        raise ValueError(f'Test {sel!r} not found in {files}')
    return hit[0]


def load_file(path):
    """Load one tab-separated test (PT1/PT2/Q) from an explicit path.

    The attribute-based entry point is load_one; this one takes the file
    directly, for callers that do not select by leak attributes (e.g.
    signal_analysis.ipynb, which is handed one path). Returns (label,
    dataframe) with the filename stem as the label.
    """
    df = pd.read_csv(path, sep='\t', skiprows=[1, 2],
                     index_col=0, usecols=list(USECOLS), header=0)
    df.index = df.index.astype(float)
    df.index.name = 'Time (s)'
    df.columns = ['PT1', 'PT2', 'Q']
    return os.path.splitext(os.path.basename(path))[0], df.astype(float)


def load_sim(path):
    """Load one simulated .dat record: (label, t, H) at the measuring section.

    The model writes t followed by one column per node, from the upstream
    reservoir (node1) to the node downstream of the valve (last column). The
    head is recorded at the VALVE, the second-to-last column: node3 in the
    single-leak model (4 nodes), node5 in the two-leak model (6 nodes), where
    node2-node4 are the three leak nodes. The measuring section is therefore
    the valve itself, and the CS_sim configs put PT1 at 0 m.
    """
    d = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None).dropna(axis=1, how='all')
    return (os.path.splitext(os.path.basename(path))[0],
            d.iloc[:, 0].to_numpy(float), d.iloc[:, -2].to_numpy(float))


def load_one(root, maneuver, flow, sel, leaks=None, is_nl=False):
    """Load one tab-separated test (PT1/PT2/Q) selected by attributes.

    `sel` is a 1-based test index or a filename substring. Returns
    (label, dataframe) with the filename stem as the label.
    """
    fn = _pick(list_tests(root, maneuver, flow, leaks, is_nl), sel)
    return load_file(os.path.join(root, fn))


def interp_to_index(df, target_index):
    """Interpolate a dataframe to target_index, preserving time as the interpolation axis."""
    if df.index.equals(target_index):
        return df.copy()
    out = (df.reindex(df.index.union(target_index))
             .interpolate(method='index')
             .reindex(target_index)
             .ffill()
             .bfill())
    out.index.name = target_index.name
    return out


def to_wide(df, label):
    out = df.copy()
    out.columns = [f'{label}_{c}' for c in out.columns]
    return out


def unwrap_single(df, label):
    """Return PT1/PT2/Q columns from a one-test wide dataframe."""
    out = pd.DataFrame(index=df.index)
    for ch in ('PT1', 'PT2', 'Q'):
        out[ch] = df[f'{label}_{ch}']
    out.index.name = df.index.name
    return out


def moving_average(y, w):
    return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


def moving_average_series(s, w):
    return s.rolling(window=w, center=True, min_periods=1).mean()


# ══════════════════════════════════════════════════════════════════════════
# Maneuver detection
# ══════════════════════════════════════════════════════════════════════════

def _dh_threshold(dh, t, base_end_t, multiplier):
    """Robust threshold from the pre-maneuver baseline: median + multiplier*MAD.

    Same idea as V6, with a fallback to the first 15% of the signal if the
    requested baseline window is empty for a specific recording.
    """
    dh = np.asarray(dh, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(dh) == 0:
        return np.nan
    if base_end_t is None:
        base_end_t = -0.01
    base_mask = t[:-1] <= base_end_t
    baseline = dh[base_mask]
    if baseline.size < 5 or not np.all(np.isfinite(baseline)):
        n_base = max(20, int(0.15 * len(dh)))
        baseline = dh[:min(len(dh), n_base)]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0:
        baseline = dh[np.isfinite(dh)]
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))
    if mad == 0:
        # Quantized transducer data often yields MAD = 0 over a quiet baseline,
        # collapsing the threshold to the bare median (single-step noise would
        # then trigger onset). Fall back to the baseline std or one
        # quantization step (smallest nonzero |dH|), whichever is larger.
        nz = baseline[baseline != 0]
        lsb = float(np.min(np.abs(nz))) if nz.size else 0.0
        mad = max(float(np.std(baseline)), lsb)
    # Floor: never below ONSET_FLOOR_FRAC of the largest step in the record.
    # The simulated records have ~5 ms of an almost noiseless baseline, so
    # med + 10*MAD is minute and fired at 0.35 ms on one file and 49 ms on
    # another; the echo window hangs off the onset and missed the LOC3 echo.
    # On the 123 experimental records the floor is below the statistical
    # threshold and not one onset moves.
    floor = ONSET_FLOOR_FRAC * float(np.nanmax(dh)) if np.any(np.isfinite(dh)) else 0.0
    return max(med + multiplier * mad, floor)


def detect_onset(signal, t, base_end_t=None, multiplier=10, consecutive=3):
    """First index where |dH| stays above a robust pre-maneuver threshold."""
    if base_end_t is None:
        base_end_t = ONSET_BASE_END
    y = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < consecutive + 5:
        return None
    dh = np.abs(np.diff(y))
    thr = _dh_threshold(dh, t, base_end_t, multiplier)
    if not np.isfinite(thr):
        return None
    above = dh > thr
    for i in range(len(above) - consecutive + 1):
        if np.all(above[i:i + consecutive]):
            return i
    return None


def a_theoretical():
    """Reference wave celerity (m/s).

    Returns the configured celerity A_THEO when a pipe config supplies one
    (the normal path with pipe_configs/*.txt); otherwise falls back to the
    Halliwell/Korteweg formula from the material properties, preserving the
    behaviour of any code that sets e/E/ALPHA directly via configure().
    """
    if A_THEO is not None:
        return float(A_THEO)
    return np.sqrt(K / rho) / np.sqrt(1 + ALPHA * (K * D) / (e * E))


def detect_events(raw_data, raw_data_nl):
    """Detect onset/offset on PT1 for the L test and the selected NL test.

    The offset is the END OF THE MANOEUVRE (maneuver_end, on the smoothed
    signal), the same definition the indicators use. It bounds the correlation
    template of the horizontal alignment. It used to be a separate criterion
    (|dH| below the onset threshold for 10 samples, on the raw signal), which
    fired 1-3 ms before the end of the manoeuvre and sometimes far off it
    (3.75 ms, mid-rise, on CS 150 NL test03; 7 ms after it on CC AUT 250).
    """
    t_axis = raw_data.index.to_numpy(dtype=float)
    T_theo = 4 * L_pipe / a_theoretical()      # only bounds the crest search
    # The statistical threshold needs a stretch of record before the maneuver to
    # measure the noise on. The experimental acquisitions have 1 s of it; the
    # simulated ones start ~10 ms before the maneuver, and then _dh_threshold
    # falls back to the first 15 % of the record, which already contains the
    # transient: the onset came out at 39.6 ms instead of 9.7 ms. With no
    # baseline, the onset comes from the record's own dynamic range instead
    # (coarse_onset), the same rule signal_analysis uses.
    has_baseline = int(np.sum(t_axis <= -0.01)) >= 5
    events = {}
    for df in (raw_data, raw_data_nl):
        for col in [c for c in df.columns if c.endswith('_PT1')]:
            label = col.replace('_PT1', '')
            sig = df[col].to_numpy(dtype=float)
            on_idx = (detect_onset(sig, t_axis, base_end_t=-0.01) if has_baseline
                      else coarse_onset(moving_average(sig, MA_WINDOW), t_axis))
            off_idx = (maneuver_end(moving_average(sig, MA_WINDOW), t_axis, T_theo,
                                    onset_idx=on_idx) if on_idx is not None else None)
            events[label] = {
                'onset_idx': on_idx,
                'offset_idx': off_idx,
                'onset_t': (t_axis[on_idx] if on_idx is not None else None),
                'offset_t': (t_axis[off_idx] if off_idx is not None else None),
            }
    return events


def print_events(events):
    print('Onset / offset per test (PT1):')
    for label, ev in events.items():
        if ev['onset_idx'] is None:
            print(f'  {label:30s}  no maneuver detected')
        else:
            dur = ev['offset_t'] - ev['onset_t']
            print(f"  {label:30s}  onset = {ev['onset_t']:+.4f} s   "
                  f"offset = {ev['offset_t']:+.4f} s   duration = {dur:.4f} s")


# ══════════════════════════════════════════════════════════════════════════
# Vertical alignment
# ══════════════════════════════════════════════════════════════════════════

def _baseline_window(raw_data, events):
    onset_times = [ev['onset_t'] for ev in events.values() if ev['onset_t'] is not None]
    t_on_min = min(onset_times) if onset_times else 0.0
    stable_t0 = float(raw_data.index.min())
    stable_t1 = t_on_min - PRE_ONSET_MARGIN
    return stable_t0, stable_t1, t_on_min


def _baselines(df, stable_t0, stable_t1):
    """Median of each column over the stable pre-maneuver window, with fallback."""
    mask = (df.index >= stable_t0) & (df.index <= stable_t1)
    if mask.sum() < 5:
        n = max(5, int(0.15 * len(df)))
        return df.iloc[:n].median(axis=0)
    return df[mask].median(axis=0)


def _single_label_from_wide(df, channel='PT1'):
    """Return the unique test label from a one-test wide dataframe."""
    cols = [c for c in df.columns if c.endswith(f'_{channel}')]
    if len(cols) != 1:
        raise ValueError(f'Expected exactly one {channel} column, got {cols}')
    return cols[0].replace(f'_{channel}', '')


def _level_leak_to_nl(leak_df, leak_pre, nl_pre, leak_label, nl_label):
    """Keep NL fixed and shift only the leak test to the NL pre-maneuver baseline.

    The PT1 pre-maneuver level of the selected NL test is the vertical reference.
    The same head shift is applied to leak PT1 and PT2, so the pressure-head
    difference between the two transducers is not artificially changed. Q is
    shifted independently to the selected NL pre-maneuver flow level.
    """
    out = leak_df.copy()
    head_shift = nl_pre[f'{nl_label}_PT1'] - leak_pre[f'{leak_label}_PT1']
    flow_shift = nl_pre[f'{nl_label}_Q']   - leak_pre[f'{leak_label}_Q']

    for ch in ('PT1', 'PT2'):
        out[f'{leak_label}_{ch}'] = leak_df[f'{leak_label}_{ch}'] + head_shift
    out[f'{leak_label}_Q'] = leak_df[f'{leak_label}_Q'] + flow_shift

    return out, head_shift, flow_shift


def vertical_align(raw_data, raw_data_nl, events):
    """Pairwise vertical alignment: NL stays fixed; only L is levelled to NL."""
    stable_t0, stable_t1, t_on_min = _baseline_window(raw_data, events)
    print(f'Baseline window: [{stable_t0:+.4f}, {stable_t1:+.4f}] s '
          f'(ends {PRE_ONSET_MARGIN*1e3:g} ms before earliest onset {t_on_min:+.4f} s)')

    leak_label = _single_label_from_wide(raw_data, 'PT1')
    nl_label   = _single_label_from_wide(raw_data_nl, 'PT1')

    pre_medians_loc = _baselines(raw_data, stable_t0, stable_t1)
    pre_medians_nl  = _baselines(raw_data_nl, stable_t0, stable_t1)

    reference_level = float(pre_medians_nl[f'{nl_label}_PT1'])
    reference_flow  = float(pre_medians_nl[f'{nl_label}_Q'])

    va_data, head_shift, flow_shift = _level_leak_to_nl(
        raw_data, pre_medians_loc, pre_medians_nl, leak_label, nl_label
    )
    va_data_nl = raw_data_nl.copy()     # NL is the fixed vertical reference.

    print(f'Vertical reference = selected NL test ({nl_label})')
    print(f'  NL PT1 baseline kept fixed: {reference_level:.4f} m')
    print(f'  L  PT1 baseline before:     {pre_medians_loc[f"{leak_label}_PT1"]:.4f} m')
    print(f'  Applied L head shift:       {head_shift:+.4f} m  (same shift to PT1 and PT2)')
    print(f'  Applied L Q shift:          {flow_shift:+.4f}')

    return va_data, va_data_nl, dict(
        stable_t0=stable_t0, stable_t1=stable_t1, t_on_min=t_on_min,
        pre_medians_loc=pre_medians_loc, pre_medians_nl=pre_medians_nl,
        reference_level=reference_level, reference_flow=reference_flow,
        leak_label=leak_label, nl_label=nl_label,
        leak_head_shift=head_shift, leak_flow_shift=flow_shift,
    )


# ══════════════════════════════════════════════════════════════════════════
# Horizontal alignment + treated table
# ══════════════════════════════════════════════════════════════════════════

def _sliding_corr(ref_seg, sig_seg):
    n, m = len(ref_seg), len(sig_seg)
    if m < n or n < 3:
        raise ValueError(f'Correlation window too short: ref={n}, search={m}')
    a = (ref_seg - ref_seg.mean()) / (ref_seg.std() + 1e-12)
    vals = np.empty(m - n + 1)
    for k in range(m - n + 1):
        seg = sig_seg[k:k + n]
        seg = (seg - seg.mean()) / (seg.std() + 1e-12)
        vals[k] = np.dot(a, seg) / n
    return np.arange(m - n + 1), vals


def _subsample_peak(lags, vals):
    k = int(np.argmax(vals))
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k-1], vals[k], vals[k+1]
        d = (y0 - 2*y1 + y2)
        delta = 0.5*(y0 - y2)/d if d != 0 else 0.0
    else:
        delta = 0.0
    return lags[k] + delta, float(vals[k])


def _shift_series(y, shift):
    idx = np.arange(len(y))
    return np.interp(idx - shift, idx, y, left=y[0], right=y[-1])


def _estimate_period_samples(sig, i_on, dt):
    """Rough oscillation period (samples) from the autocorrelation of the post-onset
    reference signal. Only used to test +/- one period during alignment."""
    seg = np.asarray(sig[i_on:], dtype=float)
    if len(seg) < 20:
        return None
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode='full')[len(seg) - 1:]
    lo = max(3, int(round(0.010 / dt)))
    if len(ac) <= lo + 2:
        return None
    k = lo + int(np.argmax(ac[lo:]))
    return k if k > lo else None


def _align_to_ref(df, ref_seg, w0, w1, records, events=None, ref_onset_idx=None, T_samp=None):
    """Align every test in df to ref_seg using REF_CHANNEL; apply same shift to PT1/PT2/Q.

    Robust horizontal alignment in three layers:
      1. coarse anchor  = (test onset - reference onset)  -> handles large offsets;
      2. period test     = also try coarse +/- one wave period, keep the highest
                           correlation -> defeats off-by-one-period (a cycle-late onset);
      3. fine refine     = +/- SEARCH_S around the chosen candidate (sub-sample).
    """
    out = pd.DataFrame(index=df.index)
    t_axis = df.index.to_numpy(dtype=float)
    dt = float(np.median(np.diff(t_axis)))
    margin = int(round(SEARCH_S / dt))
    nref = len(ref_seg)
    for lab in [c.replace(f'_{REF_CHANNEL}', '') for c in df.columns
                if c.endswith(f'_{REF_CHANNEL}')]:
        sig = df[f'{lab}_{REF_CHANNEL}'].to_numpy(dtype=float)
        coarse = 0
        if events is not None and ref_onset_idx is not None:
            oi = events.get(lab, {}).get('onset_idx')
            if oi is not None:
                coarse = int(oi) - int(ref_onset_idx)
        cands = [coarse - T_samp, coarse, coarse + T_samp] if T_samp else [coarse]
        # A pre-maneuver disturbance in the record (a pump or valve adjustment
        # while setting up) can trip detect_onset a second before the maneuver.
        # The coarse anchor is then ~1 s out and every candidate above inherits
        # the error, so the correlation never looks where it should. Offer it a
        # candidate that owes nothing to the onset. Skipped when the anchor is
        # already within the fine-refine window, where it would be redundant.
        if abs(coarse) > margin:
            cands.append(0)
        best = None
        for cand in cands:
            c0 = w0 + cand
            j0 = max(0, c0 - margin)
            j1 = min(len(sig), c0 + nref + margin)
            if j1 - j0 < nref:
                continue
            lags, vals = _sliding_corr(ref_seg, sig[j0:j1])
            ke = int(np.argmax(vals))
            edge = (ke == 0 or ke == len(vals) - 1)
            lp, pc = _subsample_peak(lags, vals)
            lp = lp + (j0 - w0)
            # A +/- period candidate must beat the onset-anchored one by a real
            # margin: leak echoes distort the front and can make the wrong
            # period correlate marginally better (seen with two leaks).
            if cand == coarse:
                score = pc
            elif cand == 0:
                score = pc - ONSET_FREE_MARGIN
            else:
                score = pc - PERIOD_TEST_MARGIN
            if (best is None) or (score > best['score']):
                best = dict(lp=lp, pc=pc, cand=cand, edge=edge, score=score)
        if best is None:
            continue
        lag_peak, peak_corr, at_edge = best['lp'], best['pc'], best['edge']
        shift = -lag_peak
        for ch in ('PT1', 'PT2', 'Q'):
            out[f'{lab}_{ch}'] = _shift_series(df[f'{lab}_{ch}'].to_numpy(float), shift)
        onset_free = (best['cand'] == 0 and coarse != 0)
        period_off = (0 if onset_free else
                      (int(round((best['cand'] - coarse) / T_samp)) if T_samp else 0))
        if onset_free:
            print(f'  [warn] {lab}: the detected onset sat {coarse * dt * 1e3:+.1f} ms '
                  f'from the reference and correlated worse than ignoring it; the '
                  f'alignment used the onset-free candidate. The onset detection '
                  f'on this test is unreliable - inspect it.')
        elif period_off != 0:
            print(f'  [info] {lab}: onset looked {-period_off:+d} period(s) off; '
                  f'corrected via the +/- period test.')
        if at_edge:
            print(f'  [warn] {lab}: refine peak at the +/-SEARCH_S edge -> raise SEARCH_S a little.')
        if peak_corr < 0.80:
            print(f'  [warn] {lab}: low peak correlation ({peak_corr:.2f}) on the incident '
                  f'front -> inspect the onset detection / aligned plot.')
        records.append({'test': lab, 'shift_samples': shift,
                        'shift_ms': shift * dt * 1e3, 'peak_corr': peak_corr,
                        'coarse_ms': coarse * dt * 1e3, 'period_off': period_off,
                        'at_search_edge': at_edge, 'onset_free': onset_free})
    return out


def horizontal_align_single_nl(va_data, va_data_nl, events, ref_nl_label,
                               correlate=True):
    """Pairwise horizontal alignment: keep selected NL fixed; shift only L.

    correlate=False skips the correlation and applies NO shift, setting only the
    analysis clock. It is what the numerical model needs: both runs start from
    the same instant by construction, so there is no jitter to correct, while
    the leak does deform the incident front and the correlation would answer
    that deformation with a displacement of up to 0.5 ms (5 samples, 0.31 m of
    position here). On the experimental pairs the shift is physical and stays on.

    Cross-correlation is used only to estimate a time lag on the incident front.
    The reference NL signal is never interpolated/shifted by the correlation step.
    Only the leak dataframe is shifted horizontally, and the same sample shift is
    applied to leak PT1, PT2 and Q.
    """
    t_axis = va_data.index.to_numpy(dtype=float)
    dt = float(np.median(np.diff(t_axis)))
    pad = int(round(PAD_S / dt))

    ev_ref = events[ref_nl_label]
    if ev_ref['onset_idx'] is None or ev_ref['offset_idx'] is None:
        raise RuntimeError(f'Cannot align: onset/offset missing for NL reference {ref_nl_label!r}.')

    i_on, i_off = ev_ref['onset_idx'], ev_ref['offset_idx']
    # Template = flat pre-onset baseline + shared incident front (unique 'flat -> rise').
    # It is built from the fixed NL reference. TEMPLATE_RISE_S caps how much of the
    # rise enters the template, so that leak-reflection echoes (which arrive a few ms
    # after the front and only exist on the L test) cannot bias the correlation.
    base = int(round(PRE_BASE_S / dt))
    w0 = max(0, i_on - base)
    w1 = min(len(t_axis), i_off + pad + 1)
    if TEMPLATE_RISE_S is not None:
        w1 = min(w1, i_on + int(round(TEMPLATE_RISE_S / dt)) + 1)
    ref_full = va_data_nl[f'{ref_nl_label}_{REF_CHANNEL}'].to_numpy(dtype=float)
    ref_seg = ref_full[w0:w1]
    T_samp = _estimate_period_samples(ref_full, i_on, dt)

    l_records = []
    if correlate:
        aligned_data = _align_to_ref(va_data, ref_seg, w0, w1, l_records,
                                     events=events, ref_onset_idx=i_on, T_samp=T_samp)
    else:
        aligned_data = va_data.copy()
        for lab in [c.replace(f'_{REF_CHANNEL}', '') for c in va_data.columns
                    if c.endswith(f'_{REF_CHANNEL}')]:
            l_records.append({'test': lab, 'shift_samples': 0, 'shift_ms': 0.0,
                              'peak_corr': np.nan, 'coarse_ms': 0.0,
                              'period_off': 0, 'at_search_edge': False,
                              'onset_free': False})
        print('Horizontal correlation skipped (correlate=False): the two records '
              'start at the same instant by construction.')
    aligned_data_nl = va_data_nl.copy()     # NL remains fixed: no horizontal correlation shift.

    # Analysis clock: anchor on the selected NL onset. This only defines where t=0
    # sits in the output; it does not change NL relative to L.
    #
    # The convention puts the onset at PRE_MANEUVER_MARGIN and expects the
    # baseline at negative times. A record that starts only a few ms before the
    # maneuver cannot satisfy both -- the simulated runs begin ~10 ms before it
    # -- and forcing the margin would leave nothing below zero, so every later
    # baseline window (onset threshold, H0, noise floor) would fall back to the
    # first 15 % of the record, which already contains the transient. Half of
    # whatever pre-maneuver time exists is kept instead, as signal_analysis does.
    avail = float(ev_ref['onset_t'] - t_axis[0])
    margin = (PRE_MANEUVER_MARGIN if avail >= 2 * PRE_MANEUVER_MARGIN
              else 0.5 * avail)
    if margin < PRE_MANEUVER_MARGIN:
        print(f'  [info] only {avail * 1e3:.1f} ms of record before the maneuver, '
              f'against the {PRE_MANEUVER_MARGIN * 1e3:g} ms the analysis clock '
              f'expects; the onset was placed at {margin * 1e3:.1f} ms.')
    clock_shift = ev_ref['onset_t'] - margin

    aligned_data.index    = aligned_data.index    - clock_shift
    aligned_data_nl.index = aligned_data_nl.index - clock_shift
    aligned_data.index.name = aligned_data_nl.index.name = 'Time (s)'

    va_data_clock = va_data.copy()
    va_data_nl_clock = va_data_nl.copy()
    va_data_clock.index = va_data_clock.index - clock_shift
    va_data_nl_clock.index = va_data_nl_clock.index - clock_shift
    va_data_clock.index.name = va_data_nl_clock.index.name = 'Time (s)'

    print(f'NL reference fixed: {ref_nl_label}  (onset {ev_ref["onset_t"]:+.4f} s)')
    print('\nHorizontal shifts applied only to L → selected NL:')
    print(pd.DataFrame(l_records).to_string(index=False))
    print('\nHorizontal shift applied to NL: 0.000 ms  (fixed reference)')
    print(f'Analysis clock: shift {clock_shift:+.4f} s  ->  NL onset at {margin:g} s')

    rec = l_records[0] if l_records else {'shift_ms': np.nan, 'peak_corr': np.nan}
    return aligned_data, aligned_data_nl, va_data_clock, va_data_nl_clock, dict(
        w0=w0, w1=w1, t_on=ev_ref['onset_t'] - clock_shift,
        t_end=ev_ref['offset_t'] - clock_shift,
        clock_shift=clock_shift,
        lag_ms=rec['shift_ms'], peak=rec['peak_corr'],
        l_records=l_records, nl_records=[],
    )


def _group_single(df, channel):
    cols = [c for c in df.columns if c.endswith(f'_{channel}')]
    if len(cols) != 1:
        raise ValueError(f'Expected one {channel} column, got {cols}')
    return df[cols[0]]


def _base_mask(t, t_on=None):
    """Steady-state window: from the start of the record to PRE_ONSET_MARGIN
    before the maneuver onset, the SAME rule as the vertical alignment
    (_baseline_window). t_on defaults to the NL onset on the analysis clock.

    Era [-1 s, -50 ms] antes do inicio, herdado do V6; a diferenca para esta
    janela e de 4 mm na mediana dos ficheiros (0.01 % da subida), mas assim o
    H0 que alinha e o que normaliza sao o mesmo.
    """
    t = np.asarray(t, dtype=float)
    return t <= (PRE_MANEUVER_MARGIN if t_on is None else t_on) - PRE_ONSET_MARGIN


def _h0_series(s, t_on=None):
    return _h0_pre(s.to_numpy(dtype=float), s.index.to_numpy(dtype=float), t_on)


def first_surge_peak(Hw):
    """Index of the crest of the first overpressure within a search window.

    The first PROMINENT peak, not the largest value: under cavitation (850 l/h)
    the collapse of the vapour cavity produces later spikes higher than the
    first overpressure, and taking the maximum would pick one of those.

    Usado pelo maneuver_end() para delimitar a procura da inflexao: o fim da
    manobra e o primeiro minimo local de dH/dt ENTRE a frente e esta crista.
    """
    prom = max((np.nanmax(Hw) - np.nanmin(Hw)) * 0.02, 1e-9)
    pk, _ = find_peaks(Hw, prominence=prom)
    return int(pk[0]) if len(pk) else int(np.argmax(Hw))


def maneuver_end(H, t, T_period, onset_idx=None):
    """Indice do FIM DA MANOBRA: o 1o ponto de inflexao depois da subida.

    A crista da primeira sobrepressao NAO e o fim da manobra. Na manobra manual
    da CC a valvula acaba de fechar aos 27.0 ms e a carga continua a subir
    devagar mais 20 ms, ate uma crista aos 47.3 ms: a diferenca entre as duas
    referencias e de 2 m (24.06 contra 26.05 m). Nas manobras automaticas as
    duas coincidem (CS 11.0 ms, CC 9.8 ms), por isso esta troca so muda a MAN.

    Uma inflexao de H e um extremo de dH/dt. A subida tem o seu maximo de
    declive na frente; o fim da manobra e o primeiro MINIMO local de dH/dt
    depois dela, onde a curva deixa de subir depressa. Procura-se entre a frente
    e a crista, exigindo proeminencia de 2 % do declive da frente; sem minimo
    local nesse intervalo, fica o ponto de menor declive.

    Ou, se vier antes, o 1o instante em que dH/dt cai abaixo de 2 % do declive
    da frente. Sem isto, uma subida que acaba num patamar a descer devagar, sem
    minimo, so terminava no minimo seguinte: no NL test01 da CS a 150 l/h a
    subida acaba aos ~10 ms (dH/dt a 3 %), segue-se um patamar e uma 2a crista
    7 cm acima aos 18.5 ms, e o fim saia aos 14.0 ms (agora 10.75, como os
    test02/03, 10.25 e 9.50). Nos restantes NL da CS e da CC o fim adianta no
    maximo 1.5 ms e a carga H_fim muda no maximo 9 cm.

    IMPORTANTE: medir isto no ensaio SEM FUGA. No ensaio com fuga o eco cria a
    sua propria inflexao e desloca o resultado -- para a mesma valvula da 12.2 ms
    com a fuga na LOC1, 16.3 na LOC2 e 22.1 na LOC3.
    """
    H = np.asarray(H, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.isfinite(T_period) or len(H) < 7:
        return None
    o = detect_onset(H, t, base_end_t=ONSET_BASE_END) if onset_idx is None else onset_idx
    if o is None:
        return None
    wmask = (t >= t[o]) & (t <= t[o] + FIRST_SURGE_FRAC * T_period)
    if wmask.sum() < 5:
        return None
    i0 = int(np.where(wmask)[0][0])
    g = np.gradient(H, t)
    i_crest = i0 + int(first_surge_peak(H[wmask]))
    i_front = i0 + int(np.argmax(g[wmask]))
    seg = g[i_front:i_crest + 1]
    if len(seg) < 3:
        return i_crest
    ks, _ = find_peaks(-seg, prominence=0.02 * abs(g[i_front]))
    k = int(ks[0] if len(ks) else np.argmin(seg))
    below = np.where(seg < 0.02 * g[i_front])[0]
    return i_front + (min(k, int(below[0])) if len(below) else k)


def nondim_common(L_sm, NL_sm, t, T_period):
    """H* com UM SO denominador: a subida do ensaio SEM FUGA.

    Cada sinal e referido ao SEU H0, mas os dois sao divididos pela mesma
    escala, dHJ_NL, lida no fim da manobra detetado no ensaio sem fuga. E a
    normalizacao da eq. 5.11 na versao de denominador comum.

    Para a energia residual isto tem uma propriedade util: como o numerador e o
    denominador de E_R/E0 sao ambos quadraticos nos sinais normalizados, uma
    escala comum cancela-se e a razao fica IDENTICA a que se obteria com os
    sinais em metros. Ou seja, E_R/E0 deixa de depender da normalizacao.
    """
    L_sm = np.asarray(L_sm, dtype=float)
    NL_sm = np.asarray(NL_sm, dtype=float)
    t = np.asarray(t, dtype=float)
    on = detect_onset(NL_sm, t, base_end_t=ONSET_BASE_END)
    t_on = t[on] if on is not None else None
    k = maneuver_end(NL_sm, t, T_period, onset_idx=on) if on is not None else None
    k_fim = k
    if k is None:
        k = int(np.nanargmax(NL_sm))
    H0_nl = _h0_pre(NL_sm, t, t_on)
    H0_l = _h0_pre(L_sm, t, t_on)
    dH = float(NL_sm[k]) - H0_nl
    if not dH:
        nan = np.full_like(NL_sm, np.nan)
        return nan, nan, k_fim
    return (L_sm - H0_l) / dH, (NL_sm - H0_nl) / dH, k_fim


def nondim_pair(L_sm, NL_sm, t, T_period):
    """H* = (H - H0) / (H_fim - H0) para um par L/NL, a MESMA definicao do damping_NL.

    H0 e o de cada sinal; H_fim e a carga de cada sinal no instante do FIM DA
    MANOBRA detetado no ensaio SEM FUGA (maneuver_end), aplicado aos dois.
    Antes, cada sinal era normalizado no seu proprio fim de manobra, e no ensaio
    com fuga o eco cria a sua propria inflexao (12-22 ms consoante a LOC, ver
    maneuver_end): o H_fim da fuga era lido num ponto ja afetado por ela, e
    dividir por ele cancelava parte da diferenca que a energia residual mede.

    Devolve (L_nd, NL_nd, k_fim); k_fim e None se o fim nao foi detetado e se
    recorreu a crista da primeira sobrepressao do NL.
    """
    L_sm = np.asarray(L_sm, dtype=float)
    NL_sm = np.asarray(NL_sm, dtype=float)
    t = np.asarray(t, dtype=float)
    on = detect_onset(NL_sm, t, base_end_t=ONSET_BASE_END)
    t_on = t[on] if on is not None else None
    k = maneuver_end(NL_sm, t, T_period, onset_idx=on) if on is not None else None
    k_fim = k
    if k is None:
        k = int(np.nanargmax(NL_sm))
        if on is not None and np.isfinite(T_period):
            m = (t >= t[on]) & (t <= t[on] + T_period)
            if m.sum() >= 3:
                k = int(np.where(m)[0][0]) + first_surge_peak(NL_sm[m])
    out = []
    for y in (L_sm, NL_sm):
        H0 = _h0_pre(y, t, t_on)
        dH = float(y[k]) - H0
        out.append((y - H0) / dH if dH else np.full_like(y, np.nan))
    return out[0], out[1], k_fim


def nondimensionalize(result, T_period=None):
    """(Re)escreve as colunas *_PT1_nondim e *_PT2_nondim de make_treated.

    T_period delimita a procura do fim da manobra; None usa 4L/a_teorica. O
    batch volta a chamar isto com o T_period do [processing] period_from, o
    mesmo que o damping_NL recebe, para que os dois leiam o mesmo instante.
    """
    if T_period is None or not np.isfinite(T_period):
        T_period = 4 * L_pipe / a_theoretical()
    t = result.index.to_numpy(dtype=float)
    # A normalizacao e UMA SO e e aplicada aqui, no inicio: cada sinal referido
    # ao SEU H0, os dois divididos pela MESMA escala, a subida do ensaio sem
    # fuga ao fim da manobra (eq. 5.11). Os tres indicadores de dimensionamento
    # sao depois calculados sobre estas curvas, pelo que partilham a mesma
    # adimensionalizacao. As colunas *_nondim_self guardam a alternativa, com
    # cada curva pela sua propria subida.
    for ch in ('PT1', 'PT2'):
        L_nd, NL_nd, _ = nondim_common(result[f'L_{ch}_smooth'].to_numpy(float),
                                       result[f'NL_{ch}_smooth'].to_numpy(float),
                                       t, T_period)
        result[f'L_{ch}_nondim'] = L_nd
        result[f'NL_{ch}_nondim'] = NL_nd
        L_s, NL_s, _ = nondim_pair(result[f'L_{ch}_smooth'].to_numpy(float),
                                   result[f'NL_{ch}_smooth'].to_numpy(float),
                                   t, T_period)
        result[f'L_{ch}_nondim_self'] = L_s
        result[f'NL_{ch}_nondim_self'] = NL_s
    return result


def make_treated(aligned_data, aligned_data_nl, T_period=None):
    """Single-pair equivalent of V6's treated table.

    For one pair, the 'mean' curve is simply the selected test curve. Names are
    kept as L_/NL_ to make later sections close to Analysis_signals. The
    dimensionless curves come from nondim_pair (see nondimensionalize).
    """
    result = pd.DataFrame(index=aligned_data.index)
    result.index.name = 'Time (s)'
    for group_name, df in [('L', aligned_data), ('NL', aligned_data_nl)]:
        for ch in ['PT1', 'PT2']:
            curve = _group_single(df, ch)
            smoothed = moving_average_series(curve, MA_WINDOW)
            result[f'{group_name}_{ch}_mean'] = curve
            result[f'{group_name}_{ch}_smooth'] = smoothed
            result[f'{group_name}_{ch}_nondim'] = np.nan

        q_curve = _group_single(df, 'Q')
        q_smooth = moving_average_series(q_curve, MA_WINDOW)
        Q0 = _h0_series(q_smooth)
        result[f'{group_name}_Q_mean'] = q_curve
        result[f'{group_name}_Q_smooth'] = q_smooth
        result[f'{group_name}_Q_nondim'] = q_smooth / Q0 if Q0 not in (0, None) else np.nan
    return nondimensionalize(result, T_period)


# ══════════════════════════════════════════════════════════════════════════
# Wave celerities
# ══════════════════════════════════════════════════════════════════════════

def pipe_length(maneuver=None):
    """Length (m) of the oscillating column: from the valve that closes on
    `maneuver` to the HPV end. L_pipe is measured from the PAV; VALVE_POS gives
    the closing valve's position from the PAV (negative = beyond it), so a valve
    0.790 m past the PAV makes the column 0.790 m longer. None / unknown
    maneuver -> L_pipe."""
    if maneuver is None:
        return L_pipe
    return L_pipe - float(VALVE_POS.get(maneuver, 0.0))


def a_period(signal, t, a_theo, maneuver=None):
    """Celerity from the oscillation period: a = 4 L / T.

    `maneuver` selects the effective length (see pipe_length / VALVE_POS).
    The peak-picking scale T_theo keeps using L_pipe on purpose: it only spaces
    the peaks, and changing it would let the correction move WHICH peaks are
    picked instead of just rescaling the result.
    """
    dt = float(np.median(np.diff(t)))
    T_theo = 4 * L_pipe / a_theo
    sig_s = moving_average(signal, MA_WINDOW)
    on = detect_onset(sig_s, t)
    if on is None:
        return np.nan
    t_on = t[on]
    seg = (t >= t_on) & (t <= t_on + 8 * T_theo)
    ts, ys = t[seg], np.asarray(signal, dtype=float)[seg]
    if len(ts) < 5:
        return np.nan
    tail = (t >= t_on + 6 * T_theo) & (t <= t_on + 8 * T_theo)
    steady = np.median(np.asarray(signal, dtype=float)[tail]) if tail.sum() else np.median(ys[-max(3, len(ys)//5):])
    y = ys - steady
    amp = y.max() - y.min()
    if amp <= 0:
        return np.nan
    pk, _ = find_peaks(y, prominence=amp * 0.06,
                       distance=max(1, int(round(0.4 * T_theo / dt))))
    tp = ts[pk]
    if len(tp) < 3:
        return np.nan
    T, _ = np.polyfit(np.arange(len(tp)), tp, 1)
    return 4 * pipe_length(maneuver) / T


def period_time(cels):
    """Oscillation period T = 4L/a, from the celerity named by PERIOD_FROM.

    T_period sets the echo search window in locate(), the first-surge window in
    damping() and the cycle length in decay_rate(), so getting it from an
    unreliable celerity poisons everything downstream. Which method is reliable
    depends on the pipe: on a straight run the period method is solid, while on
    a coiled one it can be far off (CC: 1680 m/s at 350 l/h) even though the
    transducer time-of-flight stays steady (~1094 m/s). Hence the pipe config
    chooses, via [processing] period_from.

    `cels` is the dict passed to locate(), keyed Theoretical / M1_period /
    M2_transducer / M3_Joukowsky. Falls back to the theoretical celerity, with
    a printed note, when the chosen one did not resolve.
    """
    a = period_celerity(cels)
    if a is None or not np.isfinite(a) or a <= 0:
        return np.nan
    return 4 * L_pipe / float(a)


def period_celerity(cels, who='period_time'):
    """The celerity named by PERIOD_FROM, with the theoretical one as fallback.

    Split out of period_time() because damping_NL() needs the SAME celerity to
    time the arrival of the leak echo: if the two disagreed, the search window
    of the drop would not match the period that defines it.
    """
    key = _PERIOD_SOURCES.get(PERIOD_FROM, 'M1_period')
    a = cels.get(key, np.nan)
    if a is None or not np.isfinite(a) or a <= 0:
        fb = cels.get('Theoretical') or a_theoretical()
        print(f'[{who}] {key} did not resolve ({a}); falling back to the '
              f'theoretical celerity ({fb:.1f} m/s).')
        a = fb
    return a


ONSET_COARSE_FRAC = 0.01   # of the largest step in the record (see coarse_onset)
ONSET_FLOOR_FRAC  = 0.005  # minimum onset threshold, same scale (see _dh_threshold)


def coarse_onset(signal, t):
    """Onset index from the record's own dynamic range, needing no baseline.

    detect_onset compares against a threshold built from the pre-maneuver
    baseline, which assumes the record HAS one. The experimental acquisitions
    start ~1 s before the maneuver, so they do; simulated runs start a few ms
    before it, so the baseline window is empty, _dh_threshold falls back to the
    first 15% of the record — which by then contains the transient — and the
    detected onset lands several ms late. Worse, it lands late by a different
    amount in each file, so a leak/NL pair comes out misaligned and the
    residual energy, the one indicator that compares the two, is destroyed.

    The maneuver's step dwarfs any steady drift (a factor of ~200 in the
    2026-08-14 simulated runs: 8e-3 against 4e-5), so the first crossing of
    ONSET_COARSE_FRAC of the largest step in the record pins the onset without
    a baseline at all. Only meant for clean records with no pre-trigger time;
    detect_onset stays the right tool when a baseline exists.
    """
    dh = np.abs(np.diff(np.asarray(signal, dtype=float)))
    if not dh.size:
        return None
    peak = np.nanmax(dh)
    if not np.isfinite(peak) or peak <= 0:
        return None
    over = np.flatnonzero(dh > ONSET_COARSE_FRAC * peak)
    return int(over[0]) if over.size else None


def _cfd_time(sig, t, on, T_theo, surge=0.5):
    """Time at which the incident front of `sig` crosses its timing level.

    The level is set by TDX_TIMING: 'frac_peak' (TDX_FRAC of the sensor's own
    rise) or 'absolute' (a fixed TDX_LEVEL m above its baseline). The crossing
    time is linearly interpolated between samples for sub-sample precision.
    """
    if on is None:
        return None
    dt = float(np.median(np.diff(t)))
    half = max(1, int(round(surge * T_theo / dt)))
    base_mask = t < ONSET_BASE_END
    base = float(np.median(sig[base_mask])) if np.any(base_mask) else float(sig[0])
    peak = float(np.max(sig[on:on + half]))
    if TDX_TIMING == 'absolute':
        lvl = base + TDX_LEVEL
        if peak < lvl:              # front never rises TDX_LEVEL m -> criterion undefined here
            return None
    else:                           # 'frac_peak'
        lvl = base + TDX_FRAC * (peak - base)
    for i in range(max(1, on - half), len(sig)):
        if sig[i - 1] < lvl <= sig[i]:
            return t[i - 1] + (lvl - sig[i - 1]) / (sig[i] - sig[i - 1]) * (t[i] - t[i - 1])
    return None


def a_transducer(pt1, pt2, t, a_theo):
    """Wave celerity from the PT1->PT2 time of flight over the known DX_AB.

    Rejected against A_CEILING = sqrt(K/rho), the celerity of the unconfined
    fluid: a pipe with an elastic wall adds compliance, so its celerity is
    always BELOW that limit. A value at or above it is not a slow pipe or a
    noisy test, it is a broken measurement -- the sensor separation, the
    channel timing or the front marking is wrong -- and it must not reach
    x_leak or period_time() dressed as a number. There is deliberately no knob
    to switch this off: the ceiling is physics, not a preference.

    Measured on CS (2026-09-12): all 12 NL tests return 1480-1590 m/s against
    a ceiling of 1479.9, while the period method gives 1209-1237 and Korteweg
    1270.7. The criterion is not the cause -- every front-marking rule tried
    (absolute 2 to 24 m, fractions 0.05 to 0.90, steepest dH/dt, cross
    correlation) lands in the same 1480-1590 band, because the two fronts have
    the same shape (correlation peak 0.9996 to 1.0000) and the shift is
    unambiguous. What the spread does say is that DX_AB is the suspect: the
    separation implied by the period celerity is a steady 6.031 +/- 0.145 m
    (CV 2.4%) against the 7.625 m in the config, while the fixed channel skew
    the same data would need is not steady at all (CV 9.7%). Pending a tape
    measure on PT2, CS returns NaN here. CC is unaffected: 1084 to 1100 m/s on
    all 18 NL tests, comfortably under the ceiling.
    """
    T_theo = 4 * L_pipe / a_theo
    A, B = moving_average(pt1, MA_WINDOW), moving_average(pt2, MA_WINDOW)
    tA = _cfd_time(A, t, detect_onset(A, t), T_theo)
    tB = _cfd_time(B, t, detect_onset(B, t), T_theo)
    if tA is None or tB is None or tB == tA:
        return np.nan
    a = DX_AB / (tB - tA)
    if not np.isfinite(a) or a <= 0 or a >= A_CEILING:
        print(f'  [warn] a_transducer: {a:.1f} m/s is not a possible celerity '
              f'(ceiling sqrt(K/rho) = {A_CEILING:.1f} m/s) -> n/a. '
              f'Check DX_AB = {DX_AB:.3f} m and the channel timing.')
        return np.nan
    return a


def a_joukowsky(signal, q, t, a_theo):
    T_theo = 4 * L_pipe / a_theo
    sig_s = moving_average(signal, MA_WINDOW)
    on = detect_onset(sig_s, t)
    if on is None:
        return np.nan
    t_on = t[on]
    pre = t <= (t_on - 0.005)
    if pre.sum() < 5:
        pre = t < ONSET_BASE_END
    if pre.sum() < 5:
        return np.nan
    H0 = float(np.median(np.asarray(signal, dtype=float)[pre]))
    Q0 = float(np.median(np.asarray(q, dtype=float)[pre])) * Q_UNIT_TO_M3S
    win = (t >= t_on) & (t <= t_on + 1.0 * T_theo)
    if win.sum() < 3:
        return np.nan
    # A sobrepressao de Joukowsky e a do FECHO COMPLETO, portanto le-se ao fim da
    # manobra (maneuver_end) e nao no maximo da janela, que na manobra manual da
    # CC chega 20 ms depois de a valvula fechar e 2 m acima. Sem fim da manobra
    # utilizavel, fica o maximo, como antes.
    sig = np.asarray(signal, dtype=float)
    k_fim = maneuver_end(sig, t, T_theo, onset_idx=on)
    Hmax = float(sig[k_fim]) if k_fim is not None else float(np.max(sig[win]))
    if not _usable_Q0(Q0, 'a_joukowsky'):
        return np.nan
    return (Hmax - H0) * G * S_AREA / Q0


# ══════════════════════════════════════════════════════════════════════════
# Leak location
# ══════════════════════════════════════════════════════════════════════════

def echo_strength(sig, tt, k, kind, v1):
    """How MARKED the feature at index k is, in % of the incident front v1.

    Valleys: prominence of the local minimum. Slope changes: prominence of the
    same point on the detrended gradient dH/dt - MA(dH/dt, 4 ms), which is what
    makes a shoulder measurable. The window is 4 ms and not 2 ms because a leak
    echo arriving on a falling stretch bends the curve over several ms: with
    2 ms the moving average follows the bend and erases it (the LOC3 echo of
    LOC2_L1.0 + LOC3_L1.5 test2, a clear knee at delta_t = 16 ms, measured
    0.15 % with 2 ms and 1.02 % with 4 ms). Reported only (locate() stores it per echo as
    'strength'); no estimate depends on it. Use it to tell a clear echo from a
    faint one: on the CS campaign the single-leak picks span 0.20 to 68 % and
    the weakest picks of the two-leak tests go down to 0.03 %.
    """
    sig = np.asarray(sig, dtype=float)
    tt = np.asarray(tt, dtype=float)
    if len(sig) < 5 or not v1:
        return np.nan
    dt = float(np.median(np.diff(tt)))
    if kind == 'slope_change':
        w = max(3, int(round(0.004 / dt)) | 1)
        ref = sig - moving_average(sig, w)
        tol = 0.0015
    else:
        ref, tol = sig, 0.0005
    ks, props = find_peaks(-ref, prominence=0)
    if not len(ks):
        return 0.0
    j = int(np.argmin(np.abs(ks - k)))
    if abs(ks[j] - k) * dt > tol:
        return 0.0
    return float(props['prominences'][j] / abs(v1) * 100)


def _echo_picks(sig, tt, n, v1, k_from=0, exclude_dt=()):
    """Echo instants on a search segment that starts at t1 (index 0 = t1).

    The echo of a leak is either a VALLEY of dH/dt or, when the echo arrives on a
    stretch that is already falling and leaves no valley, a SLOPE CHANGE. Never
    the deepest sample of a descent: that point is only where the window ends.

    Every pick must sit where dH/dt is already NEGATIVE: a leak echo lowers the
    head, so nothing on the rising or still-positive flank of the incident front
    can be one. Without this, two-leak tests marked the near echo halfway down
    the front (dH/dt ~ +1000 m/s) or on a shoulder still at +200 m/s.

      1. valley       - local minimum that clears the prominence bar
                        (ECHO_PROM_FRAC * |v1|), ranked by ECHO_PICK;
      2. weak_valley  - any other local minimum with prominence of at least
                        ECHO_WEAK_MIN_FRAC * |v1|, deepest first;
      3. slope_change - local maximum of the second derivative of dH/dt, i.e.
                        where the curve bends most from falling to flatter,
                        strongest first.

    Each tier is used only when the previous one has fewer than `n` picks.
    Picks must be at index >= k_from, more than 1 ms from each other and from
    every delta_t in `exclude_dt`. Returns [(k, kind)] in time order.

    Measured on the CS campaign (single leak, Theoretical, mean / median error),
    against the old rule (valley that clears the bar, else deepest sample of the
    window minus a 10 % tail): LOC1 unchanged, LOC2 0.96/0.85 -> 0.87/0.79 m,
    CC unchanged. LOC3 blind 1.84/0.71 -> 2.23/0.95 m, but with the order rule
    (order_constrained_errors) 0.68/0.64 m and no test above 2 m, every pick a
    valley or a slope change. Two-leak tests (not what the rule was tuned for):
    the far leak improves (LOC3 0.60 -> 0.48 m, none above 2 m) and LOC1 gets
    WORSE (1.33 -> 3.61 m), because its good numbers came from bends and
    shoulders on the front itself, which the sign condition now rejects.
    """
    sig = np.asarray(sig, dtype=float)
    tt = np.asarray(tt, dtype=float)
    if len(sig) < 5 or n < 1:
        return []
    dt = float(np.median(np.diff(tt)))
    tol = 0.001
    chosen = []

    def free(k):
        return (all(abs(k * dt - d) > tol for d in exclude_dt)
                and all(abs(k - c) * dt > tol for c, _ in chosen))

    ks, props = find_peaks(-sig, prominence=0)
    prom = props['prominences']
    keep = (ks >= k_from) & (sig[ks] < 0)      # so um eco baixa a carga
    ks, prom = ks[keep], prom[keep]
    bar = ECHO_PROM_FRAC * abs(v1)
    floor = ECHO_WEAK_MIN_FRAC * abs(v1)
    strong, strong_prom = ks[prom >= bar], prom[prom >= bar]
    weak = ks[(prom < bar) & (prom >= floor)]
    if ECHO_PICK == 'prominent':
        strong = strong[np.argsort(strong_prom)[::-1]]
    elif ECHO_PICK == 'first':
        strong = np.sort(strong)
    else:                                               # 'deepest'
        strong = strong[np.argsort(sig[strong])]
    weak = weak[np.argsort(sig[weak])]
    for kind, pool in (('valley', strong), ('weak_valley', weak)):
        for k in pool:
            if len(chosen) >= n:
                break
            if free(int(k)):
                chosen.append((int(k), kind))
    if len(chosen) < n:
        bend = np.gradient(np.gradient(sig, tt), tt)
        cand = [k for k in range(max(2, k_from), len(sig) - 2)
                if sig[k] < 0 and bend[k] >= bend[k - 1] and bend[k] >= bend[k + 1]]
        for k in sorted(cand, key=lambda k: -bend[k]):
            if len(chosen) >= n:
                break
            if free(k):
                chosen.append((k, 'slope_change'))
    return sorted(chosen)


def locate(L_sig, t, T_period, cels, n_leaks=1, force_n=False, NL_sig=None):
    """Reflection-based leak location from the dH/dt of the leak test.

    The incident rise (t1) is the strongest positive gradient in the first
    half-period after the onset; each leak produces a reflection dip after it,
    searched in a window that starts AT t1 (the same origin as delta_t) and
    spans LEAK_WIN_FRAC * T_period. The echo is a valley or, failing that, a
    slope change of dH/dt (see _echo_picks); each echo carries its 'kind'.

    ECHO_ON selects the signal the dips are searched on: the leak test itself
    ('leak', default) or the difference leak - NL ('diff', needs NL_sig), which
    cancels the dips the system produces without any leak. t1, the reported
    gradient and the plots always come from the leak test. With n_leaks > 1
    the first n_leaks significant minima are interpreted as one echo per leak
    (nearest leak first, since echo delay grows with distance).

    Dips closer than ECHO_MIN_DT to t1 (the ringing of a fast closure) are
    listed as candidates but never chosen; 0 disables the guard.

    An echo is preferably a local minimum that clears the ECHO_PROM_FRAC
    prominence bar; when none does, the deepest sample of the window is used
    instead. Returning None in that case was tried and reverted (see the
    fallback below).

    'candidates' reports EVERY local minimum of the window, not only the one
    picked, so a reader can see the alternative positions the trace supports.
    Each carries two flags that say what the point actually is:

        cleared_bar=True                 a clean detection
        is_minimum=True, cleared_bar=F   a real but weak valley: a plausible
                                         position the rule did not select
        is_minimum=False                 no valley there at all -- the deepest
                                         sample of a monotone stretch, which is
                                         NOT a measurement of an echo

    On the 75 CS pairs: 65 clean detections, 4 weak valleys, 6 with no valley.
    The last two tiers are almost all the far leak (LOC3, at 97% of the reach
    this window allows) and the cavitating tests at 850 l/h. build_row exports
    the flags of the selected point as echo_cleared_bar / echo_is_minimum.

    force_n=True guarantees exactly n_leaks echoes: when fewer significant dips
    are found, it pads with the deepest remaining local minima (marked
    'padded': True) so every leak always gets an estimate, however unreliable.
    Used by the notebook to always show both positions in the 2-leak case; the
    batch keeps force_n=False so unresolved echoes stay NaN in the accumulator.

    Legacy keys (t2, v2, delta_t, x_leak) always refer to the FIRST echo, so
    single-leak behaviour and outputs are unchanged; the full list is in
    'echoes'. Each extra echo carries a 'rereflection_risk' flag when its
    delta_t is close to an integer multiple of the first echo's delta_t: a
    re-reflection of the nearest leak can masquerade as a farther leak.
    """
    if not np.isfinite(T_period):
        return None
    grad = np.gradient(L_sig, t)
    o = detect_onset(L_sig, t, base_end_t=ONSET_BASE_END)
    if o is None:
        return None
    t_on = t[o]
    # Echo window anchored on the ONSET, and t1 is the strongest positive
    # gradient inside it. Anchoring on t1 instead was tried and reverted: it
    # extends the window by (t1 - t_on) ~ 5 ms, which changed nothing in 56 of
    # 58 CS tests but let the CAVITATION COLLAPSE at 850 l/h into the window.
    # That collapse is a dip ~20x deeper than any leak echo, at a fixed
    # delta_t = 23.5 ms regardless of leak size, and it captured two LOC2
    # tests whose error went from 0.91 m to 4.92 m.
    wmask = (t >= t_on) & (t <= t_on + LEAK_WIN_FRAC * T_period)
    tw, gw = t[wmask], grad[wmask]
    if len(gw) < 5:
        return None
    i_max = int(np.argmax(gw))
    t1, v1 = tw[i_max], gw[i_max]
    # Signal the dips are searched on (see ECHO_ON); gw stays the leak test,
    # so v2 and the plots keep referring to the measured dH/dt.
    if ECHO_ON == 'diff' and NL_sig is not None:
        grad_search = np.gradient(np.asarray(L_sig, float) - np.asarray(NL_sig, float), t)
    else:
        if ECHO_ON == 'diff':
            print('  [warn] locate: ECHO_ON=\'diff\' but no NL_sig given '
                  '-> searching on the leak test.')
        grad_search = grad
    sw = grad_search[wmask]
    hi = len(gw)                        # valleys and slope changes up to the window end
    search = sw[i_max:hi]
    want = max(1, int(n_leaks))
    # ECHO_MIN_DT: dips closer than this to t1 may be SEEN (candidates) but never
    # CHOSEN. Index k of `search` sits at delta_t = k * dt from t1.
    k_min = 0
    if ECHO_MIN_DT > 0 and len(tw) > 1:
        k_min = min(len(search) - 1,
                    int(np.ceil(ECHO_MIN_DT / float(np.median(np.diff(tw))) - 1e-9)))
    picks = _echo_picks(search, tw[i_max:hi], want, v1, k_from=k_min)
    kind_of = {k: kind for k, kind in picks}
    chosen = np.array(sorted(kind_of), dtype=int)
    real_k = set(kind_of)                    # echoes found by the valley / slope-change rule
    n_real = len(real_k)
    # Every dip that cleared the prominence bar, in time order, whether or not
    # it was chosen. Reported as a DIAGNOSTIC: the estimate itself is still the
    # one picked by ECHO_PICK, and x_leak/delta_t/t2 are untouched. The point is
    # to make the ambiguity visible instead of silently resolving it.
    #   CS: only 8 of 49 resolved tests have more than one candidate, and the
    #       chosen dip is already the best available in 48 of 49 -- the echo is
    #       essentially unambiguous on the straight pipe.
    #   CC: all 72 tests carry a median of 7 candidates and the earliest is
    #       NEVER the closest to the leak, which is why the coil localises so
    #       badly: on a coiled pipe the bends reflect continuously and 'the
    #       first dip' stops meaning 'the leak'.
    def _cand(k, pr, cleared, is_min=True, in_win=True):
        i_c = i_max + int(k)
        dt_c = abs(tw[i_c] - t1)
        return dict(
            t2=float(tw[i_c]), v2=float(gw[i_c]), delta_t=float(dt_c),
            prominence=float(pr), prominence_rel=float(pr / abs(v1)) if v1 else np.nan,
            x_leak={n: (a * dt_c / 2 + PT1_OFFSET) if (a and np.isfinite(a)) else np.nan
                    for n, a in cels.items()},
            cleared_bar=bool(cleared), is_minimum=bool(is_min),
            in_search_window=bool(in_win), chosen=int(k) in real_k)

    # EVERY local minimum of the window, not only the ones that cleared the bar.
    # The sub-threshold ones matter: on the CS LOC3 1.0 mm tests the true echo
    # IS a local minimum (t = 30.5 to 31.0 ms, x = 9.47 to 9.77 m against 10.26
    # real) that misses the bar at 0.005 to 0.015 against ECHO_PROM_FRAC = 0.02.
    # Reporting them lets a reader see the alternative positions the trace
    # actually supports instead of only the one the rule happened to pick.
    # Candidates are found on the FULL window, NOT on the guarded 'search'
    # array that feeds the selection. The guard limits what may be CHOSEN; it
    # should not limit what is SEEN, and prominence is measured inside whatever
    # array find_peaks is given, so a dip sitting near the cut has its right
    # flank clipped and reads as far weaker than it is. The CS LOC3 2.0 mm
    # test 1 is the case: the true echo at delta_t = 16.25 ms scores a
    # prominence of 6 when the array stops just after it, 43 a little later and
    # 109 -- above the bar -- on the full window, where it gives x = 10.23 m
    # against 10.26 m real.
    #
    # This costs almost nothing and reveals exactly what it should: over the
    # whole CS campaign it adds TWO candidates to the 78 already there, and
    # both are the far leak (errors 0.03 and 0.44 m). On the CC it adds 0.8 per
    # test to a list that already holds ~21, none of them near the leak.
    # Selection, and therefore every reported number, is untouched.
    full = gw[i_max:]
    all_min, all_props = find_peaks(-full, prominence=1e-12)
    bar = ECHO_PROM_FRAC * abs(v1)
    n_search = hi - i_max
    candidates = [_cand(k, pr, pr >= bar, in_win=int(k) < n_search)
                  for k, pr in zip(all_min, all_props['prominences'])]
    if len(chosen) and int(chosen[0]) not in set(int(k) for k in all_min):
        # The fallback landed on a plain sample, not on a local minimum: there
        # is no valley there at all. Flagged so it never reads as a detection.
        candidates.append(_cand(int(chosen[0]), 0.0, False, is_min=False))
        candidates.sort(key=lambda c: c['delta_t'])
    # 'from_fallback' kept for callers: the first echo is not a proper valley
    from_fallback = (not picks) or picks[0][1] != 'valley'
    if force_n and n_real < want:
        # Pad up to n_leaks with the deepest dips not already chosen (any
        # prominence), then the deepest remaining samples if still short.
        extra, _ = find_peaks(-search)
        extra = [int(k) for k in extra if k not in real_k and k >= k_min]
        extra.sort(key=lambda k: search[k])                 # deepest (most negative) first
        if len(real_k) + len(extra) < want:
            pool = [int(k) for k in np.argsort(search)
                    if int(k) not in real_k and int(k) not in set(extra) and k >= k_min]
            extra += pool
        chosen = np.sort(np.concatenate([np.array(sorted(real_k), dtype=int),
                                         np.array(extra[:want - n_real], dtype=int)]))
    echoes = []
    for k in chosen:
        i_min = i_max + int(k)
        t2, v2 = tw[i_min], float(gw[i_min])
        delta_t = abs(t2 - t1)
        x = {n: (a * delta_t / 2 + PT1_OFFSET) if (a and np.isfinite(a)) else np.nan
             for n, a in cels.items()}
        rr = False
        if echoes:
            dt0 = echoes[0]['delta_t']
            m = round(delta_t / dt0) if dt0 else 0
            rr = m >= 2 and abs(delta_t - m * dt0) <= 0.15 * dt0
        # padded flag by IDENTITY (search-index), not by position in the list:
        # a padded dip earlier in time must still read as padded.
        echoes.append(dict(t2=t2, v2=v2, delta_t=delta_t, x_leak=x,
                           kind=kind_of.get(int(k), 'padded'),
                           strength=echo_strength(search, tw[i_max:hi], int(k),
                                                  kind_of.get(int(k), 'padded'), v1),
                           rereflection_risk=bool(rr), padded=int(k) not in real_k,
                           from_fallback=from_fallback))
    if n_leaks > 1 and n_real < n_leaks:
        note = 'padded to show all leaks' if force_n else 'unresolved leaks stay NaN'
        print(f'  [warn] locate: found only {n_real} clear echo(es) for '
              f'{n_leaks} leaks ({note}).')
    if not echoes:
        return None
    e0 = echoes[0]
    # t_win0/t_win1 delimit the ECHO search window (anchored on t1); t_on is
    # kept for callers that plot the maneuver itself.
    return dict(t1=t1, t2=e0['t2'], v1=float(v1), v2=e0['v2'],
                delta_t=e0['delta_t'], x_leak=e0['x_leak'], echoes=echoes,
                candidates=candidates, from_fallback=from_fallback,
                echo_kind=picks[0][1] if picks else None,
                grad=grad, grad_search=grad_search, echo_on=ECHO_ON, t_on=t_on,
                t_search_end=float(tw[hi - 1]),
                t_win0=t_on, t_win1=t_on + LEAK_WIN_FRAC * T_period)


# ══════════════════════════════════════════════════════════════════════════
# First-overpressure damping + head losses
# ══════════════════════════════════════════════════════════════════════════

def _h0_pre(sig, t, t_on=None):
    """H0 = median over _base_mask; with too little record before the onset
    (the simulated runs start ~5 ms before it) the whole pre-onset stretch."""
    sig = np.asarray(sig, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = _base_mask(t, t_on)
    if mask.sum() >= 5:
        return float(np.median(sig[mask]))
    mask = t < (PRE_MANEUVER_MARGIN if t_on is None else t_on)
    if mask.sum() >= 5:
        return float(np.median(sig[mask]))
    return float(np.median(sig[:max(5, int(0.15 * len(sig)))]))


MIN_Q0 = 1e-6      # m3/s (~3.6 L/h). Below this the flow channel is not measuring.
FLOW_VALID = True  # set per pipe by [data_format] flow_valid; False = never trust Q
_FLOW_INVALID_WARNED = False


def _usable_Q0(Q0, who=''):
    """True when the steady flow is physical enough to divide by.

    A disconnected or unpowered flow meter reads a constant near-zero offset,
    which can be negative. Dividing by it turns quantities like ratio_Q = QL/Q0
    and the Joukowsky celerity into large nonsense instead of a visible
    failure, and that nonsense then travels into the results CSV. The CC
    campaign of 2026-08-04 was recorded with the flow cable disconnected
    (Q0 = -0.004 kl/h, identical in 75 of 90 files), which is what motivated
    this guard.

    MIN_Q0 only catches a DEAD meter. A meter that reads, but reads wrong,
    passes it: the CC campaign of 2026-09-14 reads 0.081 kl/h (81 L/h) at a
    nominal 250 L/h, and M3_Joukowsky came out with a 547.8 m mean location
    error on a 105 m coil. No threshold on Q0 alone can tell 81 L/h from a
    genuine low flow, so the pipe config says it instead: with
    [data_format] flow_valid = false every quantity that divides by Q0 is
    n/a, and the warning is printed once per run rather than once per pair.
    """
    global _FLOW_INVALID_WARNED
    if not FLOW_VALID:
        if not _FLOW_INVALID_WARNED:
            print('[flow] this pipe config marks the flow channel as unreliable '
                  '([data_format] flow_valid = false): M3_Joukowsky and ratio_Q '
                  'are reported as n/a.')
            _FLOW_INVALID_WARNED = True
        return False
    if Q0 is None or not np.isfinite(Q0) or Q0 < MIN_Q0:
        tag = f' [{who}]' if who else ''
        print(f'[flow]{tag} steady flow Q0 = {Q0} m3/s is not usable '
              f'(< {MIN_Q0:g}): the flow channel looks disconnected or unpowered. '
              f'Quantities that divide by Q0 are reported as n/a.')
        return False
    return True


def _drop_search_start(Nw, tw, t_on, x_leak, cels):
    """Indice, na janela, a partir do qual se procura a queda com DROP_FROM='echo'.

    O eco da fuga so pode chegar ao PT1 em t_on + 2 (x_fuga - X_PT1) / a; antes
    disso nenhuma separacao entre as duas curvas e atribuivel a fuga. A procura
    comeca no PRIMEIRO MAXIMO LOCAL do ensaio sem fuga depois desse instante, e
    nao no instante em si, porque o eco chega a meio de uma oscilacao: cortar ali
    deixaria o inicio da janela dentro do proprio mergulho do eco, que e
    transitorio. O pico seguinte e o ponto em que as curvas voltam a juntar-se, a
    partir do qual a separacao ja e a permanente.

    Devolve None (procura na janela toda, como em 'onset') quando faltam a
    posicao da fuga ou a celeridade, quando o eco cai fora da janela ou quando
    nao ha nenhum maximo local depois dele. Com varias fugas conta a mais
    proxima, que e a primeira a responder.
    """
    if x_leak is None or cels is None:
        return None
    xs = [float(x) for x in (x_leak if isinstance(x_leak, (list, tuple, np.ndarray))
                             else [x_leak]) if x is not None and np.isfinite(x)]
    if not xs:
        return None
    a = period_celerity(cels, who='damping_NL')
    if a is None or not np.isfinite(a) or a <= 0:
        return None
    t_echo = t_on + 2 * (min(xs) - X_PT1) / float(a)
    after = np.where(tw >= t_echo)[0]
    if len(after) < 3:
        return None
    pk, _ = find_peaks(Nw[after])
    if not len(pk):
        return None
    return int(after[pk[0]])


def damping_NL(H, H_nl, t, T_period, x_leak=None, cels=None):
    """QUEDA INICIAL: cada curva normalizada pela SUA subida, depois a maior separacao.

        H*           = (H - H0) / (H_fim - H0)   para cada ensaio, com H_fim a
                                                 carga ao FIM DA MANOBRA;
        ratio_H_self = max(H*_NL - H*_L)         o INDICADOR, procurado do fim da
                                                 manobra ate meio periodo.

    Cada sinal e dividido pela sua propria subida antes da subtraccao, por isso
    o indicador compara FORMAS. Exporta-se tambem dH_over_dHJ = max(H_NL - H_L)
    / dHJ_NL, com uma so normalizacao (a do ensaio sem fuga), que conta ainda a
    reducao que a fuga provoca na propria subida: nos 48 pares de uma fuga da CS
    da declives de 3.32/3.25/3.57 contra QL/Q0 (contra 11.79/3.39/3.35) e R2
    global 0.93 contra 0.46. Fica como diagnostico, para a comparacao entre as
    duas leituras.

    `x_leak` (m da PAV, escalar ou lista) e `cels` (o dict das celeridades) so
    sao precisos com [processing] drop_from = echo, que atrasa o inicio da
    procura ate a chegada do eco da fuga; sem eles, ou com drop_from = onset
    (a omissao), a procura e na janela toda, como sempre foi.

    CONDICAO DE VALIDADE:  x > a * t_fecho / 2  (7.0 m na CS, 6.2 m na CC AUT,
    17.1 m na CC MAN). Abaixo disso o eco da fuga regressa ANTES de a manobra
    acabar e a subida usada para normalizar ja vem afectada pela fuga.

    Devolve dict(ratio_H_self, H_star_max, H_star_min, dH_over_dHJ, dH_m,
    t_drop_dh, dt_drop_dh, dHJ_NL, dHJ_L, H0_NL, H0_L, t_on, t_drop, dt_drop,
    t_fim_manobra, dt_fim_manobra) ou None quando o inicio da manobra ou a
    janela nao sao utilizaveis.
    """
    if not np.isfinite(T_period):
        return None
    H = np.asarray(H, dtype=float)
    H_nl = np.asarray(H_nl, dtype=float)
    o = detect_onset(H, t, base_end_t=ONSET_BASE_END)
    if o is None:
        return None
    t_on = t[o]
    # H0 with the NL onset, as in nondim_pair, so both read the same window
    o_nl = detect_onset(H_nl, t, base_end_t=ONSET_BASE_END)
    t_on_nl = t[o_nl] if o_nl is not None else t_on
    H0_nl = _h0_pre(H_nl, t, t_on_nl)
    H0_l = _h0_pre(H, t, t_on_nl)
    # Referencia = FIM DA MANOBRA do ensaio SEM FUGA (maneuver_end), nao a crista:
    # na MAN da CC a crista esta 20 ms depois de a valvula fechar.
    k_fim = maneuver_end(H_nl, t, T_period)
    # FIM DA JANELA: o instante em que o ensaio SEM FUGA volta a descer a H0,
    # ou seja o fim da primeira sobrepressao lido no proprio sinal. Antes era
    # t_on + T/2, a previsao teorica do patamar para um fecho instantaneo, que
    # nesta bancada fica ~5 ms curta porque a manobra dura 10 ms. Se o
    # cruzamento nao for encontrado, volta-se ao T/2.
    t_lim = t_on + FIRST_SURGE_FRAC * T_period
    if k_fim is not None and k_fim + 1 < len(t):
        pos = np.arange(k_fim, len(t) - 1)
        cruz = pos[(H_nl[k_fim:-1] >= H0_nl) & (H_nl[k_fim + 1:] < H0_nl)]
        if len(cruz):
            t_lim = float(t[cruz[0] + 1])
    wmask = (t >= t_on) & (t <= t_lim)
    if wmask.sum() < 7:
        return None
    tw, Hw, Nw = t[wmask], H[wmask], H_nl[wmask]
    i0 = int(np.where(wmask)[0][0])
    i_e = (k_fim - i0) if k_fim is not None else int(np.argmax(Nw))
    i_e = int(min(max(i_e, 0), len(Nw) - 1))
    dHJ_nl = float(Nw[i_e]) - H0_nl
    dHJ_l = float(Hw[i_e]) - H0_l
    if not dHJ_nl or not dHJ_l:
        return None
    # INDICADOR (dH_over_dHJ): a maior separacao entre as duas curvas a dividir
    # pela subida de Joukowsky do ensaio SEM FUGA -- o dH_d/dH_j da Pauseiro.
    # Denominador comum aos dois ensaios, procurado do fim da manobra ate meio
    # periodo.
    # Cada curva referida ao SEU proprio H0 (e a subtracao que a normalizacao de
    # cada curva faz), e so o denominador e comum: a subida do ensaio sem fuga.
    # Nao e o mesmo que Nw - Hw, que so coincide se os dois H0 forem iguais.
    # Procurado ao longo de TODA a janela, desde o arranque da manobra. Comecar
    # no fim da manobra daria exactamente o mesmo: durante a subida as duas
    # curvas estao coladas e o maximo cai sempre depois do fecho (20.8 a
    # 24.5 ms contra um fecho aos 20.25 ms, nos doze casos da CS). Numa conduta
    # onde a janela nao esta limpa, o DROP_FROM='echo' atrasa esse inicio ate o
    # eco da fuga (ver o knob).
    # INICIO DA PROCURA: a janela toda ('onset') ou so depois do eco da fuga
    # ('echo', ver o knob DROP_FROM). O fim da manobra, i_e, continua a ser onde
    # as duas subidas sao lidas -- o que muda e so onde o maximo pode cair.
    k_ini = 0
    if DROP_FROM == 'echo':
        k_ini = _drop_search_start(Nw, tw, t_on, x_leak, cels) or 0
    dH = (Nw - H0_nl) - (Hw - H0_l)
    k_dh = k_ini + int(np.argmax(dH[k_ini:]))
    # Alternativa (ratio_H_self): cada curva pela SUA subida. Tem o seu proprio
    # maximo, que pode cair noutro instante. Dentro da zona cega a subida do
    # ensaio com fuga esta ela propria truncada pela reflexao, pelo que o
    # denominador colapsa junto com o numerador: na LOC1 da um declive de
    # calibracao de 60.9 contra os 3.87 do dH_d/dH_j. Por isso deixou de ser o
    # indicador principal (ver calibration_coeffs.py).
    dif = (Nw - H0_nl) / dHJ_nl - (Hw - H0_l) / dHJ_l
    k_self = max(i_e, k_ini)
    k = k_self + int(np.argmax(dif[k_self:]))
    return dict(ratio_H_self=float(dif[k]),
                H_star_max=float((Nw[k] - H0_nl) / dHJ_nl),
                H_star_min=float((Hw[k] - H0_l) / dHJ_l),
                dH_over_dHJ=float(dH[k_dh] / dHJ_nl),
                # Variante relativa: a mesma queda, mas a dividir pelo SINAL do
                # ensaio sem fuga ja normalizado, NO MESMO INSTANTE, em vez de
                # pela subida ao fim da manobra. Fica adimensional sem
                # referencia externa.
                dH_over_HNL=(float(dH[k_dh] / (Nw[k_dh] - H0_nl))
                             if (Nw[k_dh] - H0_nl) else float('nan')),
                HNL_at_drop=float(Nw[k_dh] - H0_nl),
                dH_m=float(dH[k_dh]),
                # as duas cargas, em metros, no instante da separacao maxima
                H_max_m=float(Nw[k_dh]), H_min_m=float(Hw[k_dh]),
                dH0=float(H0_nl - H0_l),
                t_drop_dh=float(tw[k_dh]),
                dt_drop_dh=float(tw[k_dh] - tw[i_e]),
                dHJ_NL=dHJ_nl, dHJ_L=dHJ_l, H0_NL=H0_nl, H0_L=H0_l,
                t_on=float(t_on), t_drop=float(tw[k]),
                dt_drop=float(tw[k] - tw[i_e]),
                # inicio da procura: igual ao arranque da janela com
                # DROP_FROM='onset', o pico a seguir ao eco com 'echo'
                t_drop_min=float(tw[k_ini]),
                t_fim_manobra=float(tw[i_e]), t_lim=float(t_lim),
                dt_fim_manobra=float(tw[i_e] - t_on))


def head_losses(fsd, pre_medians, leak_label, leak_mm, x_real):
    """Carga estatica na fuga, caudal de orificio QL e a razao QL/Q0.

    Uses the UNSHIFTED pre-maneuver medians of the leak test (physical levels),
    the known leak position(s), and the CD / X_PT1 / X_PT2 / AL_OVERRIDE knobs.

    Multiple simultaneous leaks: pass parallel lists for leak_mm and x_real.
    QL is then the SUM of the orifice flows, each computed with the steady head
    interpolated at its own position ('per_leak' holds the breakdown; the flat
    HL key refers to the first leak). AL_OVERRIDE only applies to single leaks.
    """
    sizes = list(leak_mm) if isinstance(leak_mm, (list, tuple)) else [leak_mm]
    xs    = list(x_real) if isinstance(x_real, (list, tuple)) else [x_real]

    dHJ = float(fsd['dHJ_L'])          # subida de carga ao fim da manobra

    H0_PT1 = float(pre_medians[f'{leak_label}_PT1'])
    H0_PT2 = float(pre_medians[f'{leak_label}_PT2'])

    # PT2 avariado (ver MAX_PT_DROP): a interpolacao entre PT1 e PT2 herdaria a
    # leitura impossivel, por isso nesses ensaios H_L e o proprio PT1. Nos ensaios
    # saos as duas vias diferem menos de 0.2 % em QL, porque QL ~ sqrt(H_L) e a
    # perda real entre os transdutores e ~0.13 m em 45 m de carga.
    pt_drop = H0_PT1 - H0_PT2
    hl_from = HL_FROM
    if hl_from == 'interp' and abs(pt_drop) > MAX_PT_DROP:
        hl_from = 'PT1'
        print(f'[flow] [{leak_label}] PT1 - PT2 = {pt_drop:.3f} m em permanente, '
              f'acima de MAX_PT_DROP = {MAX_PT_DROP:g} m: o PT2 nao esta a medir. '
              f'H_L passa a ser o PT1 ({H0_PT1:.3f} m) em vez da interpolacao.')

    per_leak = []
    for sz, xr in zip(sizes, xs):
        if AL_OVERRIDE is not None and len(sizes) == 1:
            AL = float(AL_OVERRIDE)
        else:
            AL = np.pi * (float(sz) / 1000.0) ** 2 / 4.0   # circular orifice area (m^2)
        frac_leak = (xr - X_PT1) / (X_PT2 - X_PT1)
        HL_i = H0_PT1 if hl_from == 'PT1' else H0_PT1 - frac_leak * pt_drop
        QL_i = CD * AL * np.sqrt(2 * G * HL_i) if HL_i > 0 else np.nan
        per_leak.append(dict(x=xr, size_mm=float(sz), HL=HL_i, QL=QL_i))

    QL = float(np.nansum([p['QL'] for p in per_leak]))
    HL = per_leak[0]['HL']
    Q0 = float(pre_medians[f'{leak_label}_Q']) * Q_UNIT_TO_M3S
    ratio_Q = QL / Q0 if _usable_Q0(Q0, leak_label) else np.nan
    if np.isfinite(ratio_Q) and ratio_Q > 1:
        # In steady state the leak cannot discharge more than the pipe carries,
        # so this is not a marginal case: the flow channel is under-reading.
        # It catches a partly-working meter, which _usable_Q0 lets through.
        print(f'[flow] [{leak_label}] ratio_Q = QL/Q0 = {ratio_Q:.3f} > 1: the '
              f'orifice flow exceeds the measured steady flow (Q0 = {Q0:.3e} '
              f'm3/s), which is impossible. Treat Q0 as unreliable here.')

    return dict(dHJ=dHJ, H0_PT1=H0_PT1, H0_PT2=H0_PT2,
                HL=HL, QL=QL, Q0=Q0, ratio_Q=ratio_Q, per_leak=per_leak,
                HL_from=hl_from, pt_drop=pt_drop)


# ══════════════════════════════════════════════════════════════════════════
# Residual energy
# ══════════════════════════════════════════════════════════════════════════

def _cross_time(s, t, i):
    y0, y1 = s[i], s[i + 1]
    return t[i] + (-y0 / (y1 - y0) if y1 != y0 else 0.0) * (t[i + 1] - t[i])


def residual_energy(NL_nd, L_nd, NL_sm, t, T_period):
    o = detect_onset(NL_sm, t, base_end_t=ONSET_BASE_END)
    if o is None:
        print('[residual_energy] NOT RESOLVED: onset not found on NL_sm '
              f'(ONSET_BASE_END={ONSET_BASE_END}, t range=[{t[0]:.4f}, {t[-1]:.4f}] s)')
        return None
    t_on = t[o]
    base = np.median(NL_nd[t < t_on]) if np.any(t < t_on) else 0.0
    s = NL_nd - base
    down = np.where((s[:-1] >= 0) & (s[1:] < 0))[0]
    down = down[down > o]
    t_end = None
    if len(down):
        up = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
        up = up[up > down[0]]
        if len(up):
            t_end = _cross_time(s, t, up[0])
    if t_end is None:
        t_end = t_on + T_period if np.isfinite(T_period) else None
    if t_end is None:
        print('[residual_energy] NOT RESOLVED: no NL period window. '
              f'onset@{t_on:.4f}s, down-crossings after onset={len(down)}, '
              f'T_period={T_period} (nan -> period method failed in section 6)')
        return None
    mask = (t >= t_on) & (t <= t_end)
    if mask.sum() < 3:
        print(f'[residual_energy] NOT RESOLVED: window too short '
              f'({int(mask.sum())} samples in [{t_on:.4f}, {t_end:.4f}] s)')
        return None
    diff_sq = (NL_nd[mask] - L_nd[mask]) ** 2            # residual integrand (NL - L)^2
    nl_sq   = NL_nd[mask] ** 2                           # E0 integrand: squared NL curve
    E_raw   = float(_trapz(diff_sq, t[mask]))            # leak residual energy (int over one NL period)
    E0_raw  = float(_trapz(nl_sq,   t[mask]))            # no-leak reference energy (same window)
    T_meas  = float(t_end - t_on)
    ratio   = (E_raw / E0_raw) if E0_raw else np.nan     # E_R / E0  (dimensionless)
    return dict(E_raw=E_raw, E_star=E_raw / T_meas if T_meas else np.nan,
                E0_raw=E0_raw, E0_star=E0_raw / T_meas if T_meas else np.nan,
                E_R_over_E0=ratio, t_grid=t[mask], diff_sq=diff_sq, nl_sq=nl_sq,
                t0=t_on, t1=t_end)


# ══════════════════════════════════════════════════════════════════════════
# Leak damping rate (zeta) via the ln-amplitude decay slope
# ══════════════════════════════════════════════════════════════════════════
# Friction removes a constant fraction of energy per cycle, so ln A vs cycle is a
# straight line whose slope is the damping rate. The leak adds dissipation on top,
# making its line steeper. Subtracting the NL slope isolates the leak's rate.
#   delta = -slope(ln A vs n)                zeta_leak = zeta_total - zeta_NL
#   zeta  = delta / sqrt(4*pi^2 + delta^2)   (exact; ~ delta/(2*pi) for light damping)

def _baseline_noise(sig, t, t_on):
    """Standard deviation of the signal before the maneuver, or None.

    The amplitude floor of amplitude_cycles is expressed in units of this, so
    that the usable part of each record is decided by ITS OWN noise rather than
    by a fixed duration or a fixed fraction of the first amplitude.

    Prefers the steady-state window (_base_mask, ~1 s of the experimental
    records); falls back to whatever precedes the onset. Returns
    None when neither holds DAMP_NOISE_MIN_N samples, which is the normal case
    for simulated records: they start a few ms before the maneuver.
    """
    sig = np.asarray(sig, float)
    pre = t < t_on if t_on is not None else np.ones(len(t), bool)
    for mask in (_base_mask(t, t_on) & pre, pre):
        if mask.sum() >= DAMP_NOISE_MIN_N:
            sd = float(np.nanstd(sig[mask]))
            if np.isfinite(sd) and sd > 0:
                return sd
    return None


def amplitude_cycles(H, t, T_period, t_on=None, prom_frac=None, snr_min=None):
    """Peak-to-trough amplitude per cycle of a decaying transient.

    A_n = 0.5 * (H_peak_n - H_trough_n), one value per oscillation after onset.
    Peak-minus-trough (not the distance to an assumed H_ss) keeps it robust to
    slow drift of the steady-state plateau.

    Returns (A, n, t_on, t_pk) with n = 0, 1, 2, ... — the plain index of each
    amplitude. That is only legitimate because the series is truncated before
    any oscillation can escape detection: a weak cycle can fail the prominence
    test, and numbering the survivors sequentially past that point would
    compress the abscissa and overstate the decay rate. In the CS campaign
    every undetected cycle sits below 6.2 baseline sd, so the DAMP_SNR_MIN = 8
    floor removes all of them (0 missed in 294 series over both facilities,
    against 131 before). The caller does not have to take that on trust:
    decay_rate reports n_span, computed from the peak TIMES, and warns when it
    disagrees with the number of peaks found.

    Truncation is at the FIRST cycle that fails, never a non-contiguous filter:
    dropping a cycle from the middle would renumber the ones after it, and a
    non-positive amplitude would break np.log downstream. The A <= 0 guard is
    not hypothetical — two CC tests pair a peak with a HIGHER trough on the
    second cycle and are dropped by it.
    """
    if prom_frac is None:
        prom_frac = DAMP_PROM_FRAC
    if snr_min is None:
        snr_min = DAMP_SNR_MIN
    if not np.isfinite(T_period):
        return None
    if t_on is None:
        o = detect_onset(H, t, base_end_t=ONSET_BASE_END)
        if o is None:
            return None
        t_on = t[o]
    m = t >= t_on
    tw, Hw = t[m], np.asarray(H, float)[m]
    if len(Hw) < 10:
        return None
    dt   = float(np.median(np.diff(tw)))
    dist = max(int(0.5 * T_period / dt), 1)              # >= half a period between extrema
    span = float(np.nanmax(Hw) - np.nanmin(Hw))
    prom = max(prom_frac * span, 1e-9)
    pk, _ = find_peaks(Hw,  prominence=prom, distance=dist)   # maxima
    tr, _ = find_peaks(-Hw, prominence=prom, distance=dist)   # minima
    if len(pk) < 2 or len(tr) < 1:
        return None
    A, t_pk = [], []
    for ip in pk:                                        # pair each maximum with the next minimum
        nxt = tr[tr > ip]
        if not len(nxt):
            break
        A.append(0.5 * (Hw[ip] - Hw[int(nxt[0])]))
        t_pk.append(tw[ip])
    A = np.asarray(A, float)
    t_pk = np.asarray(t_pk, float)
    if len(A) < 2 or A[0] <= 0:
        return None
    sd = _baseline_noise(H, t, t_on)
    if sd is None:
        # No baseline to measure the noise against, so no floor can be set and
        # every detected cycle is kept. Said out loud because the sequential
        # cycle numbering leans on the floor: without it, decay_rate's n_span
        # check is the only thing left guarding against a missed oscillation.
        print(f'[warn] [amplitude_cycles] fewer than {DAMP_NOISE_MIN_N} samples '
              f'before the maneuver: the noise floor is skipped and all '
              f'{len(A)} detected cycles are kept.')
        floor = 0.0
    else:
        floor = snr_min * sd
    bad = np.where((A < floor) | (A <= 0))[0]
    if len(bad):
        A, t_pk = A[:int(bad[0])], t_pk[:int(bad[0])]
    if len(A) < 2:
        return None
    return A, np.arange(len(A)), t_on, t_pk


def _line_fit(x, y):
    """Least-squares straight line; returns (slope, intercept, R^2, slope_se).

    slope_se is the standard error of the fitted slope (sqrt of the residual
    variance over Sxx), i.e. the statistical uncertainty of the decay rate.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    s, b = np.polyfit(x, y, 1)
    yh   = s * x + b
    ss_r = float(np.sum((y - yh) ** 2))
    ss_t = float(np.sum((y - np.mean(y)) ** 2))
    dof  = len(x) - 2
    sxx  = float(np.sum((x - x.mean()) ** 2))
    se   = float(np.sqrt(ss_r / dof / sxx)) if (dof > 0 and sxx > 0) else np.nan
    return float(s), float(b), (1 - ss_r / ss_t if ss_t else np.nan), se


def decay_rate(H, t, T_period, t_on=None, who=''):
    """Damping ratio (and its standard error) from ln(amplitude) vs cycle."""
    out = amplitude_cycles(H, t, T_period, t_on=t_on)
    if out is None:
        return None
    A, n, t_on, t_pk = out
    lnA        = np.log(A)
    s, b, r2, se = _line_fit(n, lnA)
    delta    = -s                            # per-cycle logarithmic decrement
    # Exact relation, not the light-damping approximation zeta ~ delta/(2*pi):
    # over the campaign the two differ by up to 1.05 % in zeta_total (CS, the
    # tests with zeta ~ 0.15) and 1.10 % in zeta_leak, so the exact form costs
    # nothing and spares the thesis the 'low damping' caveat.
    root     = np.sqrt(4 * np.pi ** 2 + delta ** 2)
    zeta     = delta / root
    # d(zeta)/d(delta) = 4*pi^2 / (4*pi^2 + delta^2)^(3/2)
    zeta_u   = se * 4 * np.pi ** 2 / root ** 3
    # Independent audit of the sequential numbering: how many periods the fit
    # actually spans, counted from the peak SPACINGS. It equals n_cycles only if
    # no oscillation escaped detection, which is what the DAMP_SNR_MIN floor buys.
    #
    # Counted step by step, never as round((t_last - t_first) / T_period): the
    # period comes from 4L/a and is not exactly the observed peak spacing, so a
    # global index accumulates drift — 7.6 cycles over the ~56 cycles of a CS
    # reference, which flags 21 of 150 series as missing a cycle when none is.
    # Each incremental step only has to tell one period from two.
    n_span = int(np.sum(np.maximum(1, np.round(np.diff(t_pk) / T_period)))) + 1
    if n_span != len(A):
        tag = f' [{who}]' if who else ''
        print(f'[decay_rate]{tag} {n_span - len(A)} cycle(s) went undetected: the '
              f'fit spans {n_span} periods but only {len(A)} peaks were found, so '
              f'n = 0,1,2,... mislabels them and the decay rate is overstated. '
              f'Raise DAMP_SNR_MIN (currently {DAMP_SNR_MIN:g}) or check the trace.')
    return dict(n=n, lnA=lnA, slope=s, intercept=b, r2=r2,
                zeta=zeta, zeta_u=zeta_u, n_cycles=len(A),
                n_span=n_span, t_on=t_on, t_pk=t_pk)


# ══════════════════════════════════════════════════════════════════════════
# Per-test exports
# ══════════════════════════════════════════════════════════════════════════

def paths_for_test(leak_dir, lab_L):
    """The four per-test export paths inside <leak_dir>/outputs/ (folder created)."""
    out_dir = os.path.join(leak_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return {kind: os.path.join(out_dir, f'{lab_L}_{kind}.txt')
            for kind in ('raw', 'aligned', 'nondimensional', 'summary')}


def write_wide_txt(df, path):
    """Tab-separated text export with header; the time index is the first column."""
    df.to_csv(path, sep='\t', float_format='%.6f')


def _fmt(v, spec='.3f', scale=1.0):
    """Format a number, or return 'N/A' for None / NaN / non-numeric values."""
    try:
        v = float(v) * scale
    except (TypeError, ValueError):
        return 'N/A'
    return format(v, spec) if np.isfinite(v) else 'N/A'


def _ms(v):
    """Segundos -> ms, tolerante a None (o resumo imprime '-' nesse caso)."""
    return None if v is None else float(v) * 1e3


def _get(d, *keys):
    """Nested dict lookup that returns None when any level is missing/None."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fmt_pm(v, u, spec='.4f'):
    """'value ± uncertainty', omitting the ± part when u is unavailable."""
    s, su = _fmt(v, spec), _fmt(u, spec)
    return s if su == 'N/A' else f'{s} ± {su}'


def location_errors(loc, x_real):
    """Per-method localisation errors from a locate() result.

    Returns {method: {'x_leak', 'err_m', 'err_L'}} with err_m = |x_leak - x_real|
    (m) and err_L the same error as % of the pipe length. Empty dict if loc is None.
    """
    if not loc:
        return {}
    out = {}
    for name, x_leak in loc['x_leak'].items():
        err_m = abs(x_leak - x_real)
        out[name] = {'x_leak': x_leak, 'err_m': err_m, 'err_L': err_m / L_pipe * 100}
    return out


def assign_echoes(loc, x_reals):
    """One detected echo (or None) per configured leak, assigned by proximity.

    Each echo is matched to the leak whose known position is closest to the
    echo's Theoretical-celerity estimate (greedy unique matching). A leak left
    without an echo means 'not detected in this test' - with merged echoes
    this is the honest attribution, instead of forcing the k-th echo onto the
    k-th leak. Validation-mode helper: it uses the known x_real positions.
    """
    echoes = (loc or {}).get('echoes', []) if isinstance(loc, dict) else []
    out = [None] * len(x_reals)
    if not echoes:
        return out
    cands = []
    for ei, echo in enumerate(echoes):
        xe = echo['x_leak'].get('Theoretical', np.nan)
        for li, xr in enumerate(x_reals):
            d = abs(xe - xr)
            if np.isfinite(d):
                cands.append((d, ei, li))
    used_e, used_l = set(), set()
    for d, ei, li in sorted(cands):
        if ei in used_e or li in used_l:
            continue
        out[li] = echoes[ei]
        used_e.add(ei)
        used_l.add(li)
    return out


def location_errors_all(loc, x_reals):
    """One location_errors-style dict per configured LEAK (proximity-assigned).

    Echoes are assigned to leaks with assign_echoes(); a leak whose echo was
    not detected gets an empty dict. Non-empty dicts also carry the assigned
    echo under the reserved '_echo' key. Single-leak pairs behave exactly as
    before (the only echo is assigned to the only leak).
    """
    assigned = assign_echoes(loc, x_reals)
    out = []
    for xr, echo in zip(x_reals, assigned):
        if echo is None:
            out.append({})
            continue
        errs = {}
        for name, x_est in echo['x_leak'].items():
            err_m = abs(x_est - xr)
            errs[name] = {'x_leak': x_est, 'err_m': err_m,
                          'err_L': err_m / L_pipe * 100}
        errs['_echo'] = echo
        out.append(errs)
    return out


def order_constrained_errors(loc, x_reals, cels, t):
    """VALIDATION ONLY: a leak cannot echo before the location nearer to PT1.

    Uses the KNOWN positions, so report these numbers apart from the blind
    location_errors_all(). For a leak at x, the nearer configured location
    x_prev (LOC_POSITIONS) sets the earliest admissible echo,

        delta_t_min = 2 (x_prev - PT1_OFFSET) / a,

    with the celerity that sets T_period (PERIOD_FROM). Leaks are handled
    nearest first and never share an echo. If the blind echo arrives after
    delta_t_min it is kept; otherwise the echo is picked AGAIN with the same
    rule as the blind one (_echo_picks: valley, weak valley, slope change), only
    after delta_t_min. The nearest location has no nearer one, so its blind
    echo is always kept; with one configured location (CC) nothing changes.

    Returns one dict per leak like location_errors_all(), plus '_echo'
    {'delta_t'}, '_source' ('kept' or the kind of the new pick, or 'none') and
    '_dt_min'.
    """
    n = len(x_reals)
    if not loc:
        return [{'_source': 'none'} for _ in range(n)]
    a_ref = cels.get(_PERIOD_SOURCES.get(PERIOD_FROM, 'M1_period'), np.nan)
    if a_ref is None or not np.isfinite(a_ref) or a_ref <= 0:
        a_ref = cels.get('Theoretical') or a_theoretical()
    positions = sorted(float(v) for v in LOC_POSITIONS.values())
    t = np.asarray(t, dtype=float)
    idx = np.where((t >= loc['t1'] - 1e-12) & (t <= loc['t_search_end'] + 1e-12))[0]
    sig, tt = np.asarray(loc['grad_search'])[idx], t[idx]
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else np.nan
    k_min = int(np.ceil(ECHO_MIN_DT / dt - 1e-9)) if ECHO_MIN_DT > 0 else 0
    blind = assign_echoes(loc, x_reals)
    used, out = [], [None] * n
    for li in sorted(range(n), key=lambda i: x_reals[i] if np.isfinite(x_reals[i]) else np.inf):
        xr = x_reals[li]
        if not np.isfinite(xr):
            out[li] = {'_source': 'none'}
            continue
        prev = max([p for p in positions if p < xr - 1e-9], default=None)
        dt_min = max(2 * (prev - PT1_OFFSET) / a_ref if prev is not None else 0.0, ECHO_MIN_DT)
        echo = blind[li]
        if (echo is not None and echo['delta_t'] > dt_min
                and all(abs(echo['delta_t'] - u) > 0.001 for u in used)):
            dt_pick, src = echo['delta_t'], 'kept'
        else:
            k_from = max(k_min, int(np.floor(dt_min / dt)) + 1)
            picks = _echo_picks(sig, tt, 1, loc['v1'], k_from=k_from, exclude_dt=used)
            if not picks:
                out[li] = {'_source': 'none', '_dt_min': dt_min}
                continue
            dt_pick, src = picks[0][0] * dt, picks[0][1]
        strength = echo_strength(sig, tt, int(round(dt_pick / dt)),
                                 src if src != 'kept' else (echo or {}).get('kind', ''), loc['v1'])
        used.append(dt_pick)
        errs = {}
        for name, a in cels.items():
            x_est = (a * dt_pick / 2 + PT1_OFFSET) if (a and np.isfinite(a)) else np.nan
            err_m = abs(x_est - xr)
            errs[name] = {'x_leak': x_est, 'err_m': err_m, 'err_L': err_m / L_pipe * 100}
        errs.update({'_echo': {'delta_t': dt_pick}, '_source': src, '_dt_min': dt_min,
                     '_strength': strength})
        out[li] = errs
    return out


def build_row(*, flow, loc_name, leak, lab_NL, lab_L, x_real, al, cel, T_period,
              va_info, loc=None, leak_errors=None, er=None, hl=None,
              ldc=None, leaks=None, x_reals=None, leak_errors_list=None,
              facility=None, maneuver=None, ordered_errors_list=None, fsd=None, er_common=None):
    """Flat per-pair results row for the accumulator CSV.

    Keeps the CSV schema in one place so the notebook and the batch runner
    cannot drift apart. `al` needs the keys lag_ms/peak/t_on/t_end (the dict
    returned by horizontal_align_single_nl works directly). Optional sections
    (loc/er/hl/ldc) are included only when resolved.

    Multi-leak pairs: pass `x_reals` and `leak_errors_list` (one entry per
    leak); the base columns keep referring to the first (nearest) leak, and
    each additional echo adds the same columns with a _leakK suffix.
    """
    row = {
        'facility': facility, 'maneuver': maneuver,
        'flow': flow, 'loc': loc_name, 'leak': leak,
        'NL_test': lab_NL, 'leak_test': lab_L,
        'x_real': x_real,
        'align_shift_ms': al['lag_ms'], 'peak_corr': al['peak'],
        'nl_ref_window_ms': (al['t_end'] - al['t_on']) * 1e3,
        'T_period_ms': T_period * 1e3,
        'reference_level_m': va_info['reference_level'],
        'reference_flow': va_info['reference_flow'],
        **{f'a_{k.split("_")[-1]}': v for k, v in cel.items()},
    }
    leak_errors = leak_errors or {}
    if loc:
        # One column group per configured leak, filled from its proximity-
        # assigned echo (see assign_echoes); leaks without an echo stay empty.
        # Base (unsuffixed) columns = leak 1, _leakK suffix for the others.
        row['n_echoes'] = len(loc.get('echoes', []))
        # t1 after the onset: the echo clock starts at t1, the manoeuvre at the
        # onset, so the blind zone is delta_t < t_fim_manobra - t1 (Accuracy_V6)
        row['t1_ms'] = (loc['t1'] - loc['t_on']) * 1e3
        # Diagnostic, not an estimate: how many dips cleared the prominence bar,
        # and the delta_t of the runner-up when there was one. n_echo_cands > 1
        # marks a test where x_leak rests on the ECHO_PICK rule rather than on
        # an unambiguous echo. Multiply delta_t by a/2 (+ PT1_OFFSET) to get the
        # position the runner-up would have implied under any celerity.
        cands = loc.get('candidates') or []
        # n_echo_cands counts only the dips ABOVE the prominence bar (the ones
        # the rule could choose from); n_echo_minima counts every local minimum
        # in the window, including the sub-threshold ones, which are the other
        # positions the trace supports.
        # What the REPORTED pick actually is, which is what a reader needs:
        #   cleared bar      -> a proper detection
        #   minimum, no bar  -> a weak valley; a plausible echo, not a clean one
        #   not a minimum    -> no valley at all; not a measurement of an echo
        sel = next((c for c in cands if c.get('chosen')), None)
        row['echo_kind'] = loc.get('echo_kind')
        first = (loc.get('echoes') or [{}])[0]
        row['echo_strength_pct'] = first.get('strength')
        row['echo_is_minimum'] = bool(sel.get('is_minimum', True)) if sel else None
        row['echo_cleared_bar'] = bool(sel.get('cleared_bar')) if sel else None
        row['n_echo_cands'] = sum(1 for c in cands if c.get('cleared_bar'))
        row['n_echo_minima'] = sum(1 for c in cands if c.get('is_minimum', True))
        other = [c['delta_t'] for c in cands if not c.get('chosen')]
        row['delta_t_s_cand2'] = other[0] if other else None
        errs_list = leak_errors_list
        if errs_list is None:
            errs_list = [leak_errors] if leak_errors else []
        for k, errs in enumerate(errs_list, start=1):
            sfx = '' if k == 1 else f'_leak{k}'
            if k > 1 and x_reals and k - 1 < len(x_reals):
                row[f'x_real{sfx}'] = x_reals[k - 1]
            echo = (errs or {}).get('_echo')
            if echo:
                row[f'delta_t_s{sfx}'] = echo['delta_t']
                if k > 1:
                    row[f'rereflection_risk{sfx}'] = echo.get('rereflection_risk', False)
            for n in ('Theoretical', 'M1_period', 'M2_transducer', 'M3_Joukowsky'):
                if n in (errs or {}):
                    row[f'x_{n}{sfx}'] = errs[n]['x_leak']
                    row[f'x_err_{n}_m{sfx}'] = errs[n]['err_m']
                    row[f'x_err_{n}_pctL{sfx}'] = errs[n]['err_L']
        if 'x_M1_period' in row:
            row['x_err_period_m'] = row['x_M1_period'] - x_real
        # VALIDATION-ONLY columns (_ord), see order_constrained_errors().
        for k, errs in enumerate(ordered_errors_list or [], start=1):
            sfx = '' if k == 1 else f'_leak{k}'
            errs = errs or {}
            row[f'echo_src_ord{sfx}'] = errs.get('_source', 'none')
            row[f'echo_strength_pct_ord{sfx}'] = errs.get('_strength')
            echo = errs.get('_echo')
            row[f'delta_t_s_ord{sfx}'] = echo['delta_t'] if echo else np.nan
            for n in ('Theoretical', 'M1_period', 'M2_transducer', 'M3_Joukowsky'):
                if n in errs:
                    row[f'x_{n}_ord{sfx}'] = errs[n]['x_leak']
                    row[f'x_err_{n}_m_ord{sfx}'] = errs[n]['err_m']
    if fsd:
        row['ratio_H_self'] = fsd['ratio_H_self']
        row['dH_over_dHJ'] = fsd['dH_over_dHJ']
        row['dH_over_HNL'] = fsd['dH_over_HNL']
        row['H0_m'] = fsd['H0_L']
        row['H_max_m'] = fsd['H_max_m']
        row['H_min_m'] = fsd['H_min_m']
        row['HNL_at_drop'] = fsd['HNL_at_drop']
        row['dt_drop_dh_ms'] = fsd['dt_drop_dh'] * 1e3
        row['H_star_max'] = fsd['H_star_max']
        row['H_star_min'] = fsd['H_star_min']
        row['dH_m'] = fsd['dH_m']
        row['dHJ_NL'] = fsd['dHJ_NL']
        row['dHJ_L'] = fsd['dHJ_L']
        row['t_fim_manobra_ms'] = fsd['dt_fim_manobra'] * 1e3
        row['t_lim_ms'] = (fsd['t_lim'] - fsd['t_on']) * 1e3
        row['dt_drop_ms'] = fsd['dt_drop'] * 1e3
    if hl:
        row['dHJ'] = hl['dHJ']
        row['HL'] = hl['HL']; row['QL'] = hl['QL']; row['Q0'] = hl['Q0']; row['ratio_Q'] = hl['ratio_Q']
        row['HL_from'] = hl.get('HL_from'); row['pt_drop_m'] = hl.get('pt_drop')
    if er:
        row['E_star'] = er['E_star']
        row['E0_star'] = er['E0_star']
        row['E_R_over_E0'] = er['E_R_over_E0']
    if er_common:
        # alternativa: a mesma energia com cada curva pela sua propria subida
        row['E_R_over_E0_self'] = er_common['E_R_over_E0']
    if ldc:
        row['zeta_total'] = ldc['zeta_total']; row['zeta_NL'] = ldc['zeta_NL']
        row['zeta_leak'] = ldc['zeta_leak']
        row['zeta_total_u'] = ldc.get('zeta_total_u')
        row['zeta_NL_u'] = ldc.get('zeta_NL_u')
        row['zeta_leak_u'] = ldc.get('zeta_leak_u')
        row['r2_lnA_L'] = ldc['r2_lnA_L']; row['r2_lnA_NL'] = ldc['r2_lnA_NL']
        row['n_cycles_L'] = ldc['n_cycles_L']; row['n_cycles_NL'] = ldc['n_cycles_NL']
        # Periods spanned by each fit, measured on the peak times. Equal to
        # n_cycles unless an oscillation escaped detection, which the
        # DAMP_SNR_MIN floor is set to prevent; decay_rate warns if it happens.
        row['n_span_L'] = ldc.get('n_span_L'); row['n_span_NL'] = ldc.get('n_span_NL')
    return row


def write_pair_exports(*, leak_dir, lab_L, lab_NL, va_data_clock, va_data_nl_clock,
                       result, loc_name, leak, x_real, T_period, al, cel,
                       loc=None, leak_errors=None, hl=None, er=None,
                       ldc=None, leaks=None, x_reals=None, leak_errors_list=None,
                       fsd=None):
    """Write the four per-test files into <leak_dir>/outputs/; returns their paths.

    a) raw            : analysis clock (t=0 at maneuver onset), no correlation shift
    b) aligned        : cross-correlation aligned + moving average (smooth columns)
    c) nondimensional : H* columns (PT channels only)
    d) summary        : lab-report-style formatted results block
    """
    paths = paths_for_test(leak_dir, lab_L)

    raw_L, raw_NL = unwrap_single(va_data_clock, lab_L), unwrap_single(va_data_nl_clock, lab_NL)
    raw_out = pd.DataFrame(index=raw_L.index)
    raw_out.index.name = 'Time (s)'
    for grp, df_ in (('L', raw_L), ('NL', raw_NL)):
        for ch in ('PT1', 'PT2', 'Q'):
            raw_out[f'{grp}_{ch}'] = df_[ch]
    write_wide_txt(raw_out, paths['raw'])

    write_wide_txt(result[[f'{g}_{ch}_smooth' for g in ('L', 'NL')
                           for ch in ('PT1', 'PT2', 'Q')]], paths['aligned'])
    write_wide_txt(result[[f'{g}_{ch}_nondim' for g in ('L', 'NL')
                           for ch in ('PT1', 'PT2')]], paths['nondimensional'])

    report = summary_report(fsd=fsd, lab_L=lab_L, lab_NL=lab_NL, loc_name=loc_name, leak=leak,
                            x_real=x_real, T_period=T_period, al=al, cel=cel, loc=loc,
                            leak_errors=leak_errors, hl=hl, er=er, ldc=ldc,
                            leaks=leaks, x_reals=x_reals,
                            leak_errors_list=leak_errors_list)
    with open(paths['summary'], 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(report) + '\n')
    return paths


def summary_report(*, lab_L, lab_NL, loc_name, leak, x_real, T_period,
                   al, cel, loc, leak_errors=None, hl=None, er=None, fsd=None,
                   ldc=None, leaks=None, x_reals=None, leak_errors_list=None):
    """Lab-report-style results block (74-char box layout), 'N/A' when missing.

    All inputs are passed explicitly (no notebook globals): `al`/`cel` are the
    alignment/celerity dicts, `loc`/`leak_errors`/`hl`/`er`/`ldc` may be
    None when the corresponding section was not resolved. For multi-leak pairs
    pass `leaks` [(loc, size), ...], `x_reals` and `leak_errors_list`: the
    header lists every leak and LEAK LOCATION shows one table per echo.
    """
    W = 74
    dbl, sep = '═' * W, '─' * W
    lerr = leak_errors if isinstance(leak_errors, dict) else {}
    al = al if isinstance(al, dict) else {}
    cel = cel if isinstance(cel, dict) else {}
    multi = bool(leaks) and len(leaks) > 1

    def section(title):
        return ['', sep, f'  {title}', sep]

    if multi:
        loc_desc  = ' + '.join(f'{l} ({_fmt(LOC_POSITIONS.get(l))} m)' for l, _ in leaks)
        size_desc = ' + '.join(s for _, s in leaks)
        xr_desc   = ', '.join(_fmt(x) for x in (x_reals or []))
    else:
        loc_desc  = f'{loc_name} ({_fmt(LOC_POSITIONS.get(loc_name))} m)'
        size_desc = f'{leak}'
        xr_desc   = _fmt(x_real)

    lines = [dbl, f'  RESULTS SUMMARY — {lab_L}', dbl]
    lines += [
        f"  Generated        : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f'  NL test         : {lab_NL}',
        f'  L test          : {lab_L}',
        f'  Location         : {loc_desc}',
        f'  Leak size        : {size_desc} mm',
        f'  x_real           : {xr_desc} m',
        f"  T_period         : {_fmt(T_period, '.3f', 1e3)} ms",
    ]
    lines += section('ALIGNMENT')
    lines += [
        f"  Shift (L -> NL)  : {_fmt(al.get('lag_ms'), '+.3f')} ms",
        f"  Peak correlation : {_fmt(al.get('peak'), '.4f')}",
        f"  NL ref. window   : [{_fmt(al.get('t_on'), '.4f')}, {_fmt(al.get('t_end'), '.4f')}] s",
    ]
    lines += section('WAVE CELERITY')
    lines += [
        f"  (a) Theoretical (Korteweg)           : {_fmt(cel.get('a_theoretical'), '.1f')} m/s",
        f"  (b) Experimental M1 (Period)         : {_fmt(cel.get('a_period'), '.1f')} m/s",
        f"  (c) Experimental M2 (Transducers)    : {_fmt(cel.get('a_transducer'), '.1f')} m/s",
        f"  (d) Experimental M3 (Joukowsky)      : {_fmt(cel.get('a_joukowsky'), '.1f')} m/s",
    ]
    lines += section('LEAK LOCATION')
    _METHODS = (('Theoretical',   'Theoretical (Korteweg)'),
                ('M1_period',     'Experimental M1 (Period)'),
                ('M2_transducer', 'Experimental M2 (Transducers)'),
                ('M3_Joukowsky',  'Experimental M3 (Joukowsky)'))

    def _loc_table(x_map, errs):
        lines.append('  Method                        x_leak (m)    err (m)   err (%L)')
        for key, name in _METHODS:
            x_ = _fmt((x_map or {}).get(key))
            em = _fmt(_get((errs or {}).get(key), 'err_m'))
            eL = _fmt(_get((errs or {}).get(key), 'err_L'), '.2f')
            lines.append(f'  {name:<30s}{x_:>8s}   {em:>8s}   {eL:>6s} %')

    if multi:
        errs_list = leak_errors_list or []
        for k, (ln, sz) in enumerate(leaks):
            errs = errs_list[k] if k < len(errs_list) else {}
            echo = (errs or {}).get('_echo')
            xr_k = (x_reals or [None] * len(leaks))[k] if x_reals else None
            flag = ('   [!] delta_t ~ multiple of echo 1 -> possible re-reflection'
                    if echo and echo.get('rereflection_risk') else '')
            lines.append(f'  Leak {k + 1} @ {ln}  (D = {sz} mm, '
                         f'x_real = {_fmt(xr_k)} m)' + flag)
            if echo is None:
                lines.append('    echo not detected (leak not resolved in this test)')
            else:
                _loc_table(echo.get('x_leak'), errs)
            if k < len(leaks) - 1:
                lines.append('')
    else:
        lines.append(f'  Leak @ {loc_name}  (D = {leak} mm, x_real = {_fmt(x_real)} m)')
        _loc_table(_get(loc, 'x_leak'), lerr)

    # All dips that cleared the prominence bar, chosen or not. One line each, so
    # a reader can see whether the echo was unambiguous or picked out of a
    # crowd. Only the marked one feeds x_leak above.
    cands = _get(loc, 'candidates') or []
    lines.append('')
    if not cands:
        lines.append('  Echo candidates: none (location unresolved)')
    else:
        n_bar = sum(1 for c in cands if c.get('cleared_bar'))
        lines.append(f'  Echo candidates: {len(cands)} local minimum/minima in the '
                     f'window, {n_bar} of them above the '
                     f'{ECHO_PROM_FRAC:g} x |dH/dt|_max prominence bar.')
        lines.append('     delta_t (ms)   dH/dt at dip   prominence   x (M1_period, m)   status')
        for c in cands:
            xm = (c.get('x_leak') or {}).get('M1_period')
            if not c.get('is_minimum', True):
                st = 'NOT a minimum (deepest sample; no valley here)'
            elif not c.get('in_search_window', True):
                st = ('above bar' if c.get('cleared_bar') else 'below bar') +                      ', PAST the edge guard - reported, never selectable'
            elif c.get('cleared_bar'):
                st = 'above bar'
            else:
                st = 'below bar - possible position, not selected'
            mark = ' <-' if c.get('chosen') else '   '
            lines.append(f'  {mark}{c["delta_t"] * 1e3:10.2f}   {c["v2"]:14.0f}   '
                         f'{c["prominence"]:10.0f}   {_fmt(xm):>16s}   {st}')
        lines.append('     the marked row is the reported x_leak; the others are the '
                     'alternative positions this trace supports.')

    lines += section('FIRST-OVERPRESSURE DAMPING')
    lines += [
        f"  End of manoeuvre : {_fmt(_ms(_get(fsd, 'dt_fim_manobra')), '.2f')} ms "
        f"after onset",
        f"  dHJ (NL / L)     : {_fmt(_get(fsd, 'dHJ_NL'))} / {_fmt(_get(fsd, 'dHJ_L'))} m",
        f"  H*max / H*min    : {_fmt(_get(fsd, 'H_star_max'), '.4f')} / "
        f"{_fmt(_get(fsd, 'H_star_min'), '.4f')}  (-)",
        f"  ratio_H_self     : {_fmt(_get(fsd, 'ratio_H_self'), '.4f')}  (-)   <- indicator",
        f"  dH / dHJ_NL      : {_fmt(_get(fsd, 'dH_over_dHJ'), '.4f')}  (-)   (single-scale reading)",
        f"  dH (in metres)   : {_fmt(_get(fsd, 'dH_m'))} m",
        f"  HL      : {_fmt(_get(hl, 'HL'))} m",
        f"  QL      : {_fmt(_get(hl, 'QL'), '.3e')} m3/s",
        f"  Q0      : {_fmt(_get(hl, 'Q0'), '.3e')} m3/s",
        f"  QL/Q0   : {_fmt(_get(hl, 'ratio_Q'), '.4f')}",
    ]
    lines += section('RESIDUAL ENERGY')
    lines += [
        f"  Window (first T) : [{_fmt(_get(er, 't0'), '.4f')}, {_fmt(_get(er, 't1'), '.4f')}] s",
        f"  E_R              : {_fmt(_get(er, 'E_raw'), '.6e')} (s*H*^2)",
        f"  E_R*  = E_R / T  : {_fmt(_get(er, 'E_star'), '.6e')} (-)",
        f"  E0               : {_fmt(_get(er, 'E0_raw'), '.6e')} (s*H*^2)",
        f"  E0*   = E0 / T   : {_fmt(_get(er, 'E0_star'), '.6e')} (-)",
        f"  E_R / E0         : {_fmt(_get(er, 'E_R_over_E0'), '.4f')} (-)",
    ]
    lines += section('DAMPING RATE (ln-amplitude decay)')
    n_L, n_NL = _get(ldc, 'n_cycles_L'), _get(ldc, 'n_cycles_NL')
    lines += [
        f"  n_cycles_L  : {int(n_L) if n_L is not None else 'N/A'}",
        f"  n_cycles_NL : {int(n_NL) if n_NL is not None else 'N/A'}",
        f"  zeta_total  : {_fmt_pm(_get(ldc, 'zeta_total'), _get(ldc, 'zeta_total_u'))}",
        f"  zeta_NL     : {_fmt_pm(_get(ldc, 'zeta_NL'), _get(ldc, 'zeta_NL_u'))}",
        f"  zeta_leak   : {_fmt_pm(_get(ldc, 'zeta_leak'), _get(ldc, 'zeta_leak_u'))}",
        f"  r2_lnA_L    : {_fmt(_get(ldc, 'r2_lnA_L'), '.4f')}",
        f"  r2_lnA_NL   : {_fmt(_get(ldc, 'r2_lnA_NL'), '.4f')}",
        dbl,
    ]
    return lines
