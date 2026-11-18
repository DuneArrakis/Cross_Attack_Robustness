"""External experiment configuration; paths are relative to the project root."""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SA_DEFAULTS = dict(T_max=4.5 * 0.02, T_min=1e-10, N_per_iter=10,
                   batch_reference_edges=1500, alpha=0.99, max_iter=2070,
                   swap_pairing='both')

DEFAULTS = {
    'data': {'input_dir': 'data/raw', 'pattern': 'graph_*.txt'},
    'output_dir': 'work/reruns/default',
    'optimization_attacks': ['HDA', 'CI', 'CoreHD', 'EI', 'FINDER', 'Zheng', 'DomiRank'],
    'evaluation_attacks': ['HDA', 'CI', 'CoreHD', 'EI', 'FINDER', 'Zheng', 'DomiRank'],
    'attack_options': {},
    'workers': None,  # None retains max(1, int(cpu_count() * 0.3)).
    'seeds': {'optimization': 20261014, 'attack': 42},
    'resume': True,
    'sa': SA_DEFAULTS,
    'snapshots': {'enabled': True, 'save_every': 100, 'only_if_improved': False,
                  'topk_enabled': False, 'top_ratio': 0.5, 'max_keep': None,
                  'topk_only_if_improved': False},
    'evaluation': {'source': 'final_graphs', 'final_graph_dir': None,
                   'graph_types': ['Original', 'HDA', 'CI', 'CoreHD', 'EI', 'FINDER', 'Zheng', 'DomiRank']},
    'analysis': {'points': 101, 'show_ratio': 0.5, 'topology': True, 'group_by': 'auto'},
    'generation': {'type': 'WS', 'count': 10, 'n': 500, 'seed': 20261014,
                   'output_dir': 'data/synthetic', 'require_connected': True,
                   'relabel': True, 'max_tries': 1000, 'ba_n0': 3, 'ba_m': 3,
                   'er_p': None, 'ws_dim': 1, 'ws_nei': 3, 'ws_p': 0.1},
    'onion': {'a': 3, 'seed_base': 42, 'min_pair_attempts': 200000,
              'pair_attempts_factor': 4000, 'max_reshuffle_tries': 100000,
              'restarts': 20, 'verbose': False},
}


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding='utf-8') as f:
            changes = json.load(f)
        unknown = set(changes) - set(DEFAULTS)
        if unknown:
            raise ValueError('Unknown configuration keys: {}'.format(sorted(unknown)))
        _merge(config, changes)
    validate_config(config)
    return config


def validate_config(config):
    from .attacks import ATTACK_NAMES
    for key, default in DEFAULTS.items():
        if isinstance(default, dict) and key != 'attack_options':
            if not isinstance(config[key], dict) or set(config[key]) != set(default):
                raise ValueError('Unexpected keys in {}'.format(key))
    for key in ('optimization_attacks', 'evaluation_attacks'):
        names = config[key]
        if not names or len(set(names)) != len(names) or set(names) - set(ATTACK_NAMES):
            raise ValueError('Invalid or duplicate attack names in {}'.format(key))
    if set(config['attack_options']) - set(ATTACK_NAMES):
        raise ValueError('Unknown method in attack_options')
    for options in config['attack_options'].values():
        if not isinstance(options, dict):
            raise ValueError('Each attack_options entry must be an object')
        if 'seed' in options:
            raise ValueError('Use seeds.attack to control attack randomness')
    if type(config['resume']) is not bool:
        raise ValueError('resume must be boolean')
    for value in config['seeds'].values():
        if type(value) is not int or value < 0:
            raise ValueError('Seeds must be nonnegative integers')
    sa = config['sa']
    if set(sa) != set(SA_DEFAULTS):
        raise ValueError('Unexpected SA parameter keys')
    if sa['swap_pairing'] not in ('both', 'legacy'):
        raise ValueError('swap_pairing must be both or legacy')
    if not (0 < sa['T_min'] < sa['T_max'] and 0 < sa['alpha'] < 1):
        raise ValueError('Invalid SA temperature or cooling parameters')
    for key in ('N_per_iter', 'max_iter'):
        if not isinstance(sa[key], int) or sa[key] < 1:
            raise ValueError('{} must be a positive integer'.format(key))
    reference = sa['batch_reference_edges']
    if reference is not None and (type(reference) is not int or reference < 1):
        raise ValueError('batch_reference_edges must be null or a positive integer')
    if config['workers'] is not None and (not isinstance(config['workers'], int) or config['workers'] < 1):
        raise ValueError('workers must be null or a positive integer')
    if (not isinstance(config['analysis']['points'], int) or config['analysis']['points'] < 2
            or not 0 < config['analysis']['show_ratio'] <= 1):
        raise ValueError('Invalid analysis points/show_ratio')
    if config['analysis']['group_by'] not in ('auto', 'batch', 'instance'):
        raise ValueError('analysis.group_by must be auto, batch or instance')
    if config['snapshots']['save_every'] < 1:
        raise ValueError('snapshots.save_every must be positive')
    generation = config['generation']
    if generation['type'].upper() not in ('BA', 'ER', 'WS'):
        raise ValueError('generation.type must be BA, ER or WS')
    for key in ('count', 'n', 'max_tries'):
        if not isinstance(generation[key], int) or generation[key] < 1:
            raise ValueError('generation.{} must be a positive integer'.format(key))
    if generation['type'].upper() == 'WS' and generation['ws_dim'] != 1:
        raise ValueError('The existing WS protocol uses ws_dim=1 so n means the total vertex count')
    if config['evaluation']['source'] not in ('inputs', 'final_graphs'):
        raise ValueError('evaluation.source must be inputs or final_graphs')
    graph_types = config['evaluation']['graph_types']
    if (not graph_types or len(set(graph_types)) != len(graph_types)
            or set(graph_types) - set(('Original', 'onion') + ATTACK_NAMES)):
        raise ValueError('Invalid evaluation.graph_types')


def input_files(config):
    directory = project_path(config['data']['input_dir'])
    paths = sorted(directory.glob(config['data']['pattern']))
    if not paths:
        raise FileNotFoundError('No input graphs in {}'.format(directory))
    return paths


def final_graph_root(config):
    value = config['evaluation']['final_graph_dir']
    return project_path(value) if value else project_path(config['output_dir']) / 'final_graphs'
