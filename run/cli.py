"""One external entry point; computation and plotting remain independent."""
from __future__ import annotations

import argparse
import importlib
import json
import os

from .config import load_config, project_path

COMMANDS = {
    'generate': 'run.datasets.generate',
    'optimize': 'run.experiments.optimize',
    'onion': 'run.experiments.onion',
    'evaluate': 'run.experiments.evaluate',
    'analyze': 'run.analysis.report',
    'rank': 'run.analysis.rankings',
}


def main(argv=None):
    # Apply before lazy imports initialize BLAS or the FINDER session.
    from .reproducibility import THREAD_ENVIRONMENT
    for key, value in THREAD_ENVIRONMENT.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description='RESS experiment stages, controlled by one JSON config.')
    parser.add_argument('command', choices=list(COMMANDS) + ['pipeline', 'config'])
    parser.add_argument('--config', help='JSON path; relative paths are resolved from the project root')
    parser.add_argument('--out-dir', help='Override the experiment output directory')
    parser.add_argument('--with-onion', action='store_true', help='Include Onion construction in pipeline')
    args = parser.parse_args(argv)
    if args.with_onion and args.command != 'pipeline':
        parser.error('--with-onion is only used with pipeline')
    config = load_config(project_path(args.config) if args.config else None)
    if args.out_dir:
        config['output_dir'] = args.out_dir
    if args.command == 'config':
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if args.command == 'pipeline':
        if config['evaluation']['source'] != 'final_graphs' or config['evaluation']['final_graph_dir']:
            parser.error('pipeline evaluates its own final_graphs; use separate commands for other inputs')
        if args.with_onion and 'onion' not in config['evaluation']['graph_types']:
            config['evaluation']['graph_types'].append('onion')
        commands = ['optimize'] + (['onion'] if args.with_onion else []) + ['evaluate', 'analyze']
    else:
        commands = [args.command]
    for command in commands:
        importlib.import_module(COMMANDS[command]).run(config)
