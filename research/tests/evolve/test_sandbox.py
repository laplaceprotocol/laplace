import pytest

from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.sandbox import ProgramRejected, check, load

BAD = {
    "import os": "import os\ndef forecast(ctx):\n    return [1] * 10\n",
    "from-import": "from os import path\ndef forecast(ctx):\n    return [1] * 10\n",
    "open": "def forecast(ctx):\n    open('x')\n    return [1] * 10\n",
    "dunder": "def forecast(ctx):\n    return ctx.__class__\n",
    "numpy io": "import numpy as np\ndef forecast(ctx):\n    np.load('x')\n    return [1] * 10\n",
    "numpy random": "import numpy as np\ndef forecast(ctx):\n    return np.random.rand(10)\n",
    "aliased numpy": "import numpy as xp\ndef forecast(ctx):\n    return xp.lib.npyio\n",
    "class": "class A:\n    pass\ndef forecast(ctx):\n    return [1] * 10\n",
    "eval": "def forecast(ctx):\n    return eval('1')\n",
    "no forecast": "def predict(ctx):\n    return [1] * 10\n",
    "syntax": "def forecast(ctx)\n    return 1\n",
    "str.format": "def forecast(ctx):\n    s = '{0.lib}'.format(ctx)\n    return [1] * 10\n",
    "format_map": "def forecast(ctx):\n    s = '{x}'.format_map({'x': 1})\n    return [1] * 10\n",
}


@pytest.mark.parametrize("source", BAD.values(), ids=BAD.keys())
def test_rejects(source: str) -> None:
    with pytest.raises(ProgramRejected):
        check(source)


def test_accepts_the_seed_and_runs_it_with_restricted_builtins() -> None:
    fn = load(SEED_PROGRAM)
    assert callable(fn)


def test_restricted_namespace_blocks_dynamic_imports() -> None:
    # A name check alone would miss an import smuggled through a string; the restricted __import__ refuses it.
    fn = load("def forecast(ctx):\n    import math\n    return [math.pi] * 10\n")
    assert fn(None)[0] == pytest.approx(3.14159, rel=1e-5)


ESCAPES = {
    "backing array": "def forecast(ctx):\n    return [float(ctx.close.base[0, -1])] * 10\n",
    "module alias": (
        "import numpy as np\ndef forecast(ctx):\n    x = np\n    return x.lib.stride_tricks.as_strided(ctx.close)\n"
    ),
    "module in a list": "import numpy as np\ndef forecast(ctx):\n    return [np][0].lib\n",
    "module default argument": "import math\ndef forecast(ctx, m=math):\n    return [1] * 10\n",
    "array file write": "def forecast(ctx):\n    ctx.close.tofile('/tmp/x')\n    return [1] * 10\n",
    "array pickle": "def forecast(ctx):\n    ctx.close.dump('/tmp/x')\n    return [1] * 10\n",
    "flag change": "def forecast(ctx):\n    ctx.close.setflags(write=True)\n    return [1] * 10\n",
    "raw pointer": "def forecast(ctx):\n    return ctx.close.ctypes\n",
    "attribute store": "def forecast(ctx):\n    ctx.horizon = 1\n    return [1] * 10\n",
    "dunder name": "def forecast(ctx):\n    return __builtins__\n",
    "ufunc.at write": (
        "import numpy as np\ndef forecast(ctx):\n    np.add.at(ctx.close, [0], 1.0)\n    return [1] * 10\n"
    ),
}


@pytest.mark.parametrize("source", ESCAPES.values(), ids=ESCAPES.keys())
def test_rejects_escapes(source: str) -> None:
    with pytest.raises(ProgramRejected):
        check(source)


def test_allows_ordinary_numpy_and_the_helper_library() -> None:
    check(
        "import numpy as np\nimport math\n\n\ndef _sd(r):\n    return math.sqrt(float(np.mean(r * r)))\n\n\n"
        "def forecast(ctx):\n    r = ctx.lib.log_returns(ctx.close)\n"
        "    return ctx.lib.student_t_probs(_sd(r[-168:]) * math.sqrt(ctx.horizon), 5.0, ctx.edges)\n"
    )
