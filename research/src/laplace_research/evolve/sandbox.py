"""Static checks and a restricted namespace for language-model-written forecasting programs.

The guarantee that a program cannot see the future is structural (see ``history``): its process holds no future
candle. These checks are defence in depth against everything else a program might try. Every program comes from our
own mutation calls in research runs; production runs execute programs in isolated containers without network access.

- Imports are limited to ``math`` and ``numpy``, and a module may only be used as ``module.attribute``, never
  assigned, passed or returned (an alias would slip past the attribute checks below).
- Attribute names that reach files, raw memory, pickling or randomness are refused on any object.
- No private or dunder names, no attribute assignment, no dynamic code, classes or global state.
- Builtins are restricted; ``__import__`` returns the allowed modules and lets numpy's own internal submodule imports
  through.
"""

from __future__ import annotations

import ast
import builtins
import math
from collections.abc import Callable
from types import CodeType, ModuleType
from typing import Any

import numpy as np

MAX_SOURCE_BYTES = 16_000
MAX_LINES = 300
ALLOWED_MODULES: dict[str, ModuleType] = {"math": math, "numpy": np}
FORBIDDEN_NAMES = frozenset(
    {
        "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec", "exit", "getattr", "globals",
        "help", "input", "locals", "memoryview", "open", "print", "quit", "setattr", "super", "type", "vars",
    }
)  # fmt: skip
# Refused on any object: file access, raw memory and buffers, pickling, flag changes, randomness, format strings
# (str.format walks attributes, private ones included, past every check here; f-strings are checked like code), and
# ufunc.at, which writes into read-only arrays (numpy does not check the flag there).
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "at", "base", "ctypes", "ctypeslib", "DataSource", "distutils", "dump", "dumps", "f2py", "format", "format_map",
        "fromfile", "fromregex", "genfromtxt", "load", "loads", "loadtxt", "memmap", "random", "save", "savetxt",
        "savez", "savez_compressed", "setflags", "testing", "tofile",
    }
)  # fmt: skip
# Refused when reached from a module (``numpy.lib`` holds stride tricks and file I/O); ``ctx.lib`` stays allowed.
FORBIDDEN_MODULE_ATTRIBUTES = frozenset({"lib"})
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
        "map", "max", "min", "pow", "range", "reversed", "round", "set", "slice", "sorted", "sum", "tuple", "zip",
        "ArithmeticError", "Exception", "FloatingPointError", "IndexError", "KeyError", "OverflowError",
        "TypeError", "ValueError", "ZeroDivisionError",
    )
}  # fmt: skip


class ProgramRejected(ValueError):
    """The source failed static checks or does not define ``forecast(ctx)``."""


def _module_aliases(tree: ast.Module) -> set[str]:
    aliases = set(ALLOWED_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return aliases


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def check(source: str) -> ast.Module:
    """Parse and statically check ``source``; raise ``ProgramRejected`` with the reason."""
    if len(source.encode()) > MAX_SOURCE_BYTES or source.count("\n") > MAX_LINES:
        raise ProgramRejected("program too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramRejected(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    modules = _module_aliases(tree)
    parents = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in ALLOWED_MODULES]
            if bad:
                raise ProgramRejected(f"import of {', '.join(bad)} is not allowed (only math and numpy)")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_MODULES or node.level:
                raise ProgramRejected(f"import from {node.module} is not allowed (only math and numpy)")
            if any(a.name in FORBIDDEN_ATTRIBUTES | FORBIDDEN_MODULE_ATTRIBUTES for a in node.names):
                raise ProgramRejected("that numpy import is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
                raise ProgramRejected(f"use of {node.id} is not allowed")
            parent = parents.get(node)
            if node.id in modules and not (isinstance(parent, ast.Attribute) and parent.value is node):
                raise ProgramRejected(f"a module may only be used as {node.id}.<attribute>")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ProgramRejected(f"access to private attribute {node.attr} is not allowed")
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise ProgramRejected(f"attribute {node.attr} is not allowed")
            if node.attr in FORBIDDEN_MODULE_ATTRIBUTES and _rooted_in(node.value, modules):
                raise ProgramRejected(f"module attribute {node.attr} is not allowed")
            if not isinstance(node.ctx, ast.Load):
                raise ProgramRejected("assigning or deleting attributes is not allowed")
        elif isinstance(node, ast.ClassDef | ast.Global | ast.Nonlocal | ast.AsyncFunctionDef | ast.Await):
            raise ProgramRejected(f"{type(node).__name__} is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forecast" for n in tree.body):
        raise ProgramRejected("no top-level forecast(ctx) function")
    return tree


def _rooted_in(node: ast.expr, names: set[str]) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in names


def _restricted_import(
    name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0
) -> ModuleType:
    if level == 0 and name in ALLOWED_MODULES:
        return ALLOWED_MODULES[name]
    # numpy's C code imports some of its own submodules lazily (e.g. numpy._core._methods on the first ``ndarray.clip``
    # in a process), and such imports go through the builtins of whichever Python frame is running: the program's.
    # Program source can never ask for these: the static check refuses every import except ``math`` and ``numpy``.
    if level == 0 and name.startswith("numpy."):
        return builtins.__import__(name, globals_, locals_, fromlist, level)
    raise ImportError(f"import of {name} is not allowed")


def compile_checked(source: str) -> CodeType:
    """Check ``source`` and compile it, ready for ``instantiate``."""
    return compile(check(source), "<program>", "exec")


def instantiate(code: CodeType) -> Callable[[Any], Any]:
    """Run compiled program code in a fresh restricted namespace and return its ``forecast`` function. Nothing a
    program keeps at module level survives into the next instance."""
    namespace: dict[str, Any] = {"__builtins__": {**SAFE_BUILTINS, "__import__": _restricted_import}}
    exec(code, namespace)
    fn = namespace.get("forecast")
    if not callable(fn):
        raise ProgramRejected("forecast is not callable")
    return fn  # type: ignore[no-any-return]


def load(source: str) -> Callable[[Any], Any]:
    """Check ``source`` and return its ``forecast`` function, bound to a restricted namespace."""
    return instantiate(compile_checked(source))
