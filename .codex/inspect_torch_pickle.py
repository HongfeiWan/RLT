"""Inspect PyTorch checkpoint metadata without materializing tensor storage."""

from __future__ import annotations

import collections
import pickle
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class Storage:
    dtype: str
    key: str
    location: str
    numel: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def rebuild_tensor(storage: Storage, offset: int, shape: tuple[int, ...], stride: tuple[int, ...], *args: Any) -> Tensor:
    return Tensor(storage, offset, tuple(shape), tuple(stride))


def rebuild_parameter(tensor: Tensor, *args: Any) -> Tensor:
    return tensor


class MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: tuple[Any, ...]) -> Storage:
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")
        _, storage_type, key, location, numel = pid[:5]
        return Storage(getattr(storage_type, "__name__", str(storage_type)), str(key), str(location), int(numel))

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        if module == "torch" and name == "device":
            return str
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("collections", "defaultdict"):
            return collections.defaultdict
        return super().find_class(module, name)


def summarize(value: Any, prefix: str = "", max_depth: int = 3) -> None:
    if isinstance(value, Tensor):
        print(f"{prefix}: tensor dtype={value.storage.dtype} shape={value.shape} storage={value.storage.key} numel={value.storage.numel}")
        return
    if isinstance(value, dict):
        print(f"{prefix or '<root>'}: dict len={len(value)} keys={list(value)[:20]}")
        if max_depth > 0:
            for key, item in value.items():
                summarize(item, f"{prefix}.{key}" if prefix else str(key), max_depth - 1)
        return
    if isinstance(value, (list, tuple)):
        print(f"{prefix}: {type(value).__name__} len={len(value)}")
        if max_depth > 0:
            for index, item in enumerate(value[:20]):
                summarize(item, f"{prefix}[{index}]", max_depth - 1)
        return
    print(f"{prefix}: {type(value).__name__} {value!r}")


summarize(MetadataUnpickler(sys.stdin.buffer).load(), max_depth=4)
