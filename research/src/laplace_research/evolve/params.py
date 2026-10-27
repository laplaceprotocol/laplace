"""Parameter files: public arrays a program reads as ``ctx.params``.

A program whose model is too large to write into its source (a few thousand trees) carries a parameter file. The file
is canonical JSON (sorted keys, no whitespace, ASCII), one entry per array with its dtype, shape and little-endian
bytes in base64, so the same arrays always give the same bytes and the file's hash commits to every number. A
program's identity then covers both its source and its parameters.

Inside a program the arrays are read-only (they are views of immutable bytes) and the mapping cannot be changed, so
one forecast cannot leave anything behind for the next.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

FORMAT = "laplace-params/1"
DTYPES = ("<f8", "<i4")  # float64 and int32 are enough for trees and weights
MAX_BYTES = 32_000_000
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")

Params = Mapping[str, np.ndarray[Any, Any]]
EMPTY: Params = MappingProxyType({})


class BadParams(ValueError):
    """A parameter file that is not canonical, too large or malformed."""


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def encode(arrays: Mapping[str, np.ndarray[Any, Any]]) -> bytes:
    """The parameter file for ``arrays`` (float arrays are stored as float64, integer arrays as int32)."""
    entries = {}
    for name, value in arrays.items():
        if not NAME.fullmatch(name):
            raise BadParams(f"bad array name {name!r}")
        a = np.asarray(value)
        if a.dtype.kind == "f":
            a = a.astype("<f8")
        elif a.dtype.kind in "iu":
            if a.size and (a.min() < -(2**31) or a.max() >= 2**31):
                raise BadParams(f"{name}: integers beyond int32")
            a = a.astype("<i4")
        else:
            raise BadParams(f"{name}: dtype {a.dtype} is not stored")
        entries[name] = {
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "data": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(),
        }
    out = _canonical({"arrays": entries, "format": FORMAT})
    if len(out) > MAX_BYTES:
        raise BadParams(f"parameter file of {len(out)} bytes is over {MAX_BYTES}")
    return out


def decode(raw: bytes) -> Params:
    """Read-only arrays from a parameter file; refuses anything ``encode`` would not have written."""
    if len(raw) > MAX_BYTES:
        raise BadParams(f"parameter file of {len(raw)} bytes is over {MAX_BYTES}")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadParams(f"not JSON: {exc}") from None
    if not isinstance(body, dict) or set(body) != {"arrays", "format"} or body["format"] != FORMAT:
        raise BadParams(f"not a {FORMAT} file")
    if _canonical(body) != raw:
        raise BadParams("not canonical JSON")
    if not isinstance(body["arrays"], dict):
        raise BadParams("arrays must be an object")
    out: dict[str, np.ndarray[Any, Any]] = {}
    for name, entry in body["arrays"].items():
        if not NAME.fullmatch(name) or not isinstance(entry, dict) or set(entry) != {"data", "dtype", "shape"}:
            raise BadParams(f"bad entry {name!r}")
        dtype, shape = entry["dtype"], entry["shape"]
        if dtype not in DTYPES or not isinstance(shape, list):
            raise BadParams(f"{name}: bad dtype or shape")
        if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape):
            raise BadParams(f"{name}: bad shape {shape}")
        try:
            data = base64.b64decode(entry["data"], validate=True)
        except (binascii.Error, TypeError, ValueError):
            raise BadParams(f"{name}: data is not base64") from None
        if base64.b64encode(data).decode() != entry["data"]:
            raise BadParams(f"{name}: data is not canonical base64")
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if len(data) != size * np.dtype(dtype).itemsize:
            raise BadParams(f"{name}: {len(data)} bytes do not fit shape {shape} of {dtype}")
        out[name] = np.frombuffer(data, dtype=dtype).reshape(shape)  # a view of immutable bytes: read-only
    return MappingProxyType(out)
