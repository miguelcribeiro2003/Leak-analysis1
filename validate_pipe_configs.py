# -*- coding: utf-8 -*-
"""Validate every pipe config in pipe_configs/*.txt (skips _*.txt).

Run this after adding or editing a pipe file, to confirm it is well-formed
without opening a notebook:

    python validate_pipe_configs.py

Reports one line per file (OK / FAILED), plus any warnings (e.g. a leak beyond
the pipe length), and checks that no two files declare the same `name`.
Exit code is non-zero if any file failed.
"""
import os
import sys
from glob import glob

import analysis as an

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass


def main():
    files = [f for f in sorted(glob(os.path.join(an.PIPE_CONFIG_DIR, '*.txt')))
             if not os.path.basename(f).startswith('_')]
    if not files:
        print(f'No pipe configs found in {an.PIPE_CONFIG_DIR}.')
        return 0

    print(f'Validating {len(files)} pipe config(s) in {an.PIPE_CONFIG_DIR}\n')
    names, failed = {}, 0
    for path in files:
        base = os.path.basename(path)
        try:
            cfg = an.load_pipe_config(path)          # prints warnings inline
        except Exception as exc:
            failed += 1
            print(f'  [FAILED] {base}: {exc}')
            continue
        name = cfg['_name']
        dup = f"  [!] duplicate name '{name}' (also in {names[name]})" if name in names else ''
        names.setdefault(name, base)
        n_sen = 1 + sum(1 for k in cfg if k == 'X_PT2')  # informational
        n_loc = len(cfg['LOC_POSITIONS'])
        locs = (f'{n_loc} leak location(s)' if n_loc
                else 'no leak locations (blind analysis only)')
        print(f"  [OK]     {base}  -> name='{name}', L={cfg['L_pipe']:g} m, "
              f"D={cfg['D']:g} m, celerity={cfg['A_THEO']:g} m/s, {locs}{dup}")
        if dup:
            failed += 1

    print()
    if failed:
        print(f'{failed} problem(s) found.')
        return 1
    print('All pipe configs are valid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
