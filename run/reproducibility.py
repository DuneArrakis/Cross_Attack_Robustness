"""Stable task seeds; never seed a process-global generator during optimization."""
import hashlib
import json


def derive_seed(base, *parts):
    payload = json.dumps([int(base)] + list(parts), ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:4], 'big')


def optimization_seeds(config, instance, method):
    seeds = config['seeds']
    return (derive_seed(seeds['optimization'], instance, method, 'optimization'),
            derive_seed(seeds['attack'], instance, method, 'objective'))


def evaluation_seed(config, instance, design, method):
    return derive_seed(config['seeds']['attack'], instance, design, method, 'evaluation')


THREAD_ENVIRONMENT = {key: '1' for key in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TF_NUM_INTEROP_THREADS',
    'TF_NUM_INTRAOP_THREADS', 'RESS_FINDER_THREADS')}
THREAD_ENVIRONMENT.update(CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')

