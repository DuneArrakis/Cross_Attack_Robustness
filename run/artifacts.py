"""Resolve graph references written by the experiment pipeline."""
from functools import lru_cache
import json
from pathlib import Path

INDEX = 'bundle_manifest.json'


def _relative(path, parent):
    try:
        return path.relative_to(parent)
    except ValueError:
        return None


@lru_cache(maxsize=8)
def _read_index(path, size, mtime_ns):
    # A formal archive has many graph references. Parse its checksum index once,
    # and invalidate the cache if the index file changes.
    return json.loads(path.read_text())


def _bundle_index(root):
    root = Path(root).resolve()
    for parent in (root,) + tuple(root.parents):
        path = parent / INDEX
        if path.is_file():
            stat = path.stat()
            return parent, _read_index(path, stat.st_size, stat.st_mtime_ns)
    return None, None


def resolve_artifact(reference, root, original_root=None):
    """Prefer the archive copy even when an old absolute path still exists."""
    root, reference = Path(root).resolve(), Path(reference)
    bundle, index = _bundle_index(root)
    if not reference.is_absolute():
        resolved = (root / reference).resolve()
        if bundle is not None and _relative(resolved, bundle) is None:
            raise ValueError('Artifact escapes the result bundle: ' + str(reference))
        return resolved
    if index is not None:
        mappings = sorted(index['path_mappings'], key=lambda m: len(Path(m['source']).parts), reverse=True)
        for mapping in mappings:
            relative = _relative(reference, Path(mapping['source']))
            if relative is not None:
                resolved = (bundle / mapping['destination'] / relative).resolve()
                if _relative(resolved, bundle) is None:
                    raise ValueError('Invalid bundle path mapping')
                return resolved
        raise ValueError('Absolute path is not archived: ' + str(reference))
    if original_root is not None:
        relative = _relative(reference, Path(original_root))
        if relative is not None:
            return root / relative
    return reference
