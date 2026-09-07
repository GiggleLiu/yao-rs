"""Stable host identity and explicit compiler configuration for saved runs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


def stable_lscpu(text):
    # Clock frequency and scaling percentage change while an idle host runs.
    # Keep topology/cache/model information, including changes to online CPUs.
    lines = [line for line in text.splitlines()
             if 'mhz' not in line.partition(':')[0].lower()
             and line.partition(':')[0].strip().lower() != 'bogomips']
    if not any(line.startswith('Model name:') for line in lines):
        raise ValueError('lscpu did not identify the CPU model')
    return '\n'.join(lines) + '\n'


def hardware():
    if sys.platform == 'darwin':
        return subprocess.check_output(['sysctl', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.physicalcpu'], text=True)
    raw = subprocess.check_output(['lscpu'], text=True, env={**os.environ, 'LC_ALL': 'C'})
    memory = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    return stable_lscpu(raw) + f'Physical memory bytes: {memory}\n'


def configuration(root, environ=None, cargo_home=None):
    environ = os.environ if environ is None else environ
    names = ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER',
             'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_TARGET', 'JULIA_CPU_TARGET',
             'CC', 'CFLAGS', 'CXX', 'CXXFLAGS', 'LD_PRELOAD', 'DYLD_INSERT_LIBRARIES',
             'OPENBLAS_CORETYPE', 'MKL_DEBUG_CPU_TYPE']
    names += [name for name in environ if name.startswith(('CARGO_PROFILE_RELEASE_', 'CARGO_PROFILE_BENCH_'))
              or (name.startswith('CARGO_TARGET_') and name.endswith('_RUSTFLAGS'))]
    flags = {name: environ[name] for name in sorted(set(names)) if name in environ}
    home = Path(cargo_home or environ.get('CARGO_HOME', Path.home() / '.cargo'))
    locations = [('cargo_home', home)] + [(f'ancestor_{i}', path / '.cargo') for i, path in enumerate([root, *root.parents])]
    configs = {}
    for label, folder in locations:
        for name in ['config', 'config.toml']:
            path = folder / name
            if path.is_file():
                # Configuration may contain credentials; retain only its hash.
                configs[label + '/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(flags=flags, cargo_configs=configs, julia_startup=False,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
