from typing import List, Tuple, Iterable, Mapping, Any, Callable, Union
from dataclasses import dataclass, replace
from huggingface_hub import HfApi, hf_hub_download
from os import PathLike
from pathlib import Path

def merge_shapes(*dims) -> Tuple[int, ...]:
    """
    Concatenate multiple dims into one.
    Args:
        dims: either an int or a list of ints.
    Returns:
        concatenated dims.
    """
    out: List[int] = []
    for d in dims:
        if isinstance(d, Iterable):
            out.extend(d)
        else:
            out.append(d)
    return tuple(out)

def recursive_replace_str(src: dataclass,
                          key: str, value: Any):
    """ replace() support for x.y = z """
    keys = key.split('.', maxsplit=1)
    if len(keys) == 1:
        try:
            return replace(src, **{key: value})
        except Exception:
            print(F'replace failed for key={key}')
            raise
    replaced = recursive_replace_str(getattr(src, keys[0]),
                                     keys[1], value)
    return replace(src, **{keys[0]: replaced})

def recursive_replace_map(
        src: dataclass, entries: Mapping[str, Any]):
    out = src
    for k, v in entries.items():
        out = recursive_replace_str(out, k, v)
    return out

def last_ckpt(root: Union[str, PathLike, Path],
              pattern: str = '*.ckpt',
              key: Callable[[Path], Any] = None):

    # By default, sort by file modification time.
    if key is None:
        lambda f: f.stat().st_mtime

    path = Path(root)
    if path.is_file():
        return path

    try:
        last_ckpt = max(path.rglob(pattern), key=key)
    except ValueError:
        # Fallback to huggingface
        repo_id, ckpt_name = str(root).split(':', maxsplit=1)
        last_ckpt = hf_hub_download(repo_id, ckpt_name)

    return last_ckpt