import base64
import json
from collections.abc import Callable

import numpy as np
import pytest

from laplace_research.evolve import params


def test_arrays_come_back_exactly_read_only_and_in_canonical_bytes() -> None:
    arrays = {"h1.threshold": np.array([0.5, -1.25, 3e-9]), "h1.left": np.array([1, -1, -2]), "scale": np.float64(2)}
    raw = params.encode(arrays)
    assert raw == params.encode(dict(reversed(list(arrays.items()))))
    back = params.decode(raw)
    assert back["h1.threshold"].tolist() == [0.5, -1.25, 3e-9]
    assert back["h1.left"].dtype == np.dtype("<i4")
    assert back["h1.left"].tolist() == [1, -1, -2]
    assert back["scale"].shape == ()
    with pytest.raises(ValueError, match="read-only"):
        back["h1.threshold"][0] = 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        back["h1.threshold"].flags.writeable = True
    with pytest.raises(TypeError):
        back["new"] = np.zeros(1)  # type: ignore[index]


def _noncanonical_base64(raw: bytes) -> bytes:
    """The same bytes, with a padding bit set in the base64 of the one array (decoders that are not strict ignore
    it)."""
    body = json.loads(raw)
    data = body["arrays"]["a"]["data"]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    i = data.index("=") - 1
    loose = data[:i] + alphabet[alphabet.index(data[i]) + 1] + data[i + 1 :]
    assert base64.b64decode(loose) == base64.b64decode(data)
    body["arrays"]["a"]["data"] = loose
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _with(**changes: object) -> Callable[[bytes], bytes]:
    def edit(raw: bytes) -> bytes:
        body = json.loads(raw)
        body["arrays"]["a"] |= changes
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    return edit


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw.replace(b",", b", "),
        lambda raw: raw.replace(params.FORMAT.encode(), b"laplace-params/0"),
        _with(dtype="<f4"),
        _with(shape=[2]),
        _with(shape=[-1]),
        _with(extra=1),
        _noncanonical_base64,
        lambda raw: raw[:-1],
    ],
    ids=["whitespace", "format", "dtype", "shape", "negative shape", "extra key", "loose base64", "truncated"],
)
def test_anything_encode_would_not_write_is_refused(edit: Callable[[bytes], bytes]) -> None:
    raw = params.encode({"a": np.array([1.0])})
    params.decode(raw)
    with pytest.raises(params.BadParams):
        params.decode(edit(raw))


def test_large_or_odd_arrays_are_refused_when_written() -> None:
    with pytest.raises(params.BadParams):
        params.encode({"Bad Name": np.zeros(1)})
    with pytest.raises(params.BadParams):
        params.encode({"a": np.array([2**40])})
    with pytest.raises(params.BadParams):
        params.encode({"a": np.array(["text"])})


@pytest.mark.parametrize(
    "body",
    [
        {"a": {"data": "", "dtype": "<f8", "shape": [2**32, 2**32]}},
        {"a": {"data": "", "dtype": "<f8", "shape": [2**70]}},
        {"a": {"data": "", "dtype": "<f8", "shape": [1] * 65}},
    ],
    ids=["product overflows int64", "beyond int64", "too many dimensions"],
)
def test_shapes_that_could_overflow_are_refused(body: dict[str, object]) -> None:
    raw = json.dumps({"arrays": body, "format": params.FORMAT}, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(params.BadParams):
        params.decode(raw)


def test_deeply_nested_json_is_refused() -> None:
    with pytest.raises(params.BadParams):
        params.decode(b"[" * 100_000 + b"]" * 100_000)
