from skverify.maps import numpy as _numpy_map
from skverify.maps import special as _special_map
from skverify.pair import IDX, Pair

from .api import to_sympy

__all__ = ["Pair", "IDX", "to_sympy"]


def _tidy(expr):
    """Print-tier cleanups that change nothing mathematically:
    collapse sum axes of extent one, and state negated comparisons
    positively (-a/n > -b/n becomes a/n < b/n)."""
    import sympy as _sym

    def collapse(s):
        fun, limits = s.function, []
        for v, lo, hi in s.limits:
            if lo == hi:
                # the axis has one slot: drop the dummy from indexed
                # terms outright (a dead index position), substitute
                # anywhere it appears in arithmetic
                fun = fun.replace(
                    lambda x, v=v: isinstance(x, _sym.Indexed)
                    and v in x.indices,
                    lambda x, v=v, lo=lo: x.base[
                        tuple(i for i in x.indices if i != v)
                    ] if len(x.indices) > 1 else x.base[lo],
                )
                fun = fun.subs(v, lo)
            else:
                limits.append((v, lo, hi))
        if fun.could_extract_minus_sign():
            # a sum of negated terms IS the negated sum: pull the sign
            # out so comparisons can read positively
            inner = _sym.Sum(-fun, *limits) if limits else -fun
            return -inner
        return _sym.Sum(fun, *limits) if limits else fun

    expr = expr.replace(lambda x: isinstance(x, _sym.Sum), collapse)
    rel_flip = {
        _sym.Lt: _sym.Gt, _sym.Le: _sym.Ge,
        _sym.Gt: _sym.Lt, _sym.Ge: _sym.Le,
    }
    if type(expr) in rel_flip:
        lhs, rhs = expr.lhs, expr.rhs
        if lhs.could_extract_minus_sign() and rhs.could_extract_minus_sign():
            expr = rel_flip[type(expr)](-lhs, -rhs)
    return expr


def latex(expr, aliases=None):
    """sympy.latex with readable defaults for certificates: code-ish
    names render upright, sum axes of extent one collapse, negated
    comparisons read positively. Pass a dict as `aliases` to shorten
    long names to T1, T2, ...; the dict fills with alias -> full name
    so you can print a legend."""
    import sympy as _sym

    if isinstance(expr, _sym.MatrixBase):
        names = {
            s_: r"\mathtt{{{}}}".format(str(s_).replace("_", r"\_"))
            for s_ in expr.free_symbols
            if "_" in str(s_)
        }
        return _sym.latex(expr, symbol_names=names)
    if not isinstance(expr, _sym.Basic):
        return str(expr)
    expr = _tidy(expr)
    names = {}
    for s in sorted(expr.atoms(_sym.Symbol), key=str):
        n = str(s)
        if "_" not in n:
            continue
        if aliases is not None and len(n) > 14:
            short = next(
                (k for k, v in aliases.items() if v == n),
                f"T{len(aliases) + 1}",
            )
            aliases[short] = n
            names[s] = rf"\mathtt{{{short}}}"
        else:
            names[s] = r"\mathtt{{{}}}".format(n.replace("_", r"\_"))
    return _sym.latex(expr, symbol_names=names)
