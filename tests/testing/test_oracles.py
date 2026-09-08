"""Two standing bug-finding oracles, run at CI scale.

Ground truth: the formula comes FIRST and the code is generated from
it, so 'matches' is the only correct verdict (and 'differs' after a
mutation). Conversion differential: sympy truth and z3 truth of every
converted guard atom must agree at random exact points, in positive
AND negated contexts -- the property whose violation was the
aux-under-negation unsoundness.
"""

import linecache

import numpy as np
import pytest
import sympy
import z3

from skverify.explore import _to_z3
from skverify.testing import check_formula

V = sympy.IndexedBase("v")
N = 4


class TestGroundTruth:
    def _make(self, src, tag):
        code = f"def gen(v):\n    return float({src})\n"
        fname = f"<gt {tag}>"
        linecache.cache[fname] = (len(code), None,
                                  code.splitlines(True), fname)
        ns = {"np": np}
        exec(compile(code, fname, "exec"), ns)
        return ns["gen"]

    def _rand(self, rng, depth=2):
        ch = rng.integers(0, 7)
        if depth == 0 or ch == 0:
            k = int(rng.integers(0, N))
            return V[k], f"v[{k}]"
        if ch == 1:
            c = int(rng.integers(1, 5))
            f, s = self._rand(rng, depth - 1)
            return c * f, f"({c} * {s})"
        if ch == 2:
            a, sa = self._rand(rng, depth - 1)
            b, sb = self._rand(rng, depth - 1)
            return a + b, f"({sa} + {sb})"
        if ch == 3:
            a, sa = self._rand(rng, depth - 1)
            b, sb = self._rand(rng, depth - 1)
            return a * b, f"({sa} * {sb})"
        if ch == 4:
            a, sa = self._rand(rng, depth - 1)
            return a ** 2, f"({sa} ** 2)"
        j = sympy.Dummy("j", integer=True)
        if ch == 5:
            return sympy.Sum(V[j], (j, 0, N - 1)), "v.sum()"
        return sympy.Sum(V[j], (j, 0, N - 1)) / N, "v.mean()"

    def test_forty_generated_programs(self):
        rng = np.random.default_rng(11)
        args = (np.array([0.7, -1.2, 2.5, 0.3]),)
        for t in range(40):
            f, src = self._rand(rng)
            fn = self._make(src, t)
            v = check_formula(fn, args, f)
            assert v.matches, (src, str(f), v.message())
            vm = check_formula(fn, args, f + sympy.Rational(1, 7))
            assert not vm.matches, (src, str(f), vm.tier)


class TestConversionDifferential:
    def _rand_arith(self, rng, depth=2):
        a, b, c = sympy.symbols("a b c", real=True)
        ch = rng.integers(0, 8)
        if depth == 0 or ch == 0:
            return [a, b, c][int(rng.integers(0, 3))]
        if ch == 1:
            return sympy.Rational(int(rng.integers(-30, 30)),
                                  int(rng.integers(1, 9)))
        if ch == 2:
            return self._rand_arith(rng, depth-1) + self._rand_arith(rng, depth-1)
        if ch == 3:
            return self._rand_arith(rng, depth-1) * self._rand_arith(rng, depth-1)
        if ch == 4:
            return sympy.Abs(self._rand_arith(rng, depth-1))
        if ch == 5:
            return sympy.sqrt(sympy.Abs(self._rand_arith(rng, depth-1)) + 1)
        if ch == 6:
            return sympy.Max(self._rand_arith(rng, depth-1),
                             self._rand_arith(rng, depth-1))
        return self._rand_arith(rng, depth-1) ** 2

    def _truth_via_z3(self, expr, point):
        varmap = {}
        zf = _to_z3(expr, z3, varmap)
        if zf is None:
            return None
        zsub = [
            (var, z3.RealVal(f"{point[sym].p}/{point[sym].q}"))
            for sym, var in varmap.items()
            if not isinstance(sym, sympy.Dummy)
        ]
        solver = z3.Solver()
        solver.add(z3.substitute(zf, *zsub))
        return solver.check() == z3.sat

    def test_hundred_atoms_positive_and_negated(self):
        rng = np.random.default_rng(5)
        a, b, c = sympy.symbols("a b c", real=True)
        rels = [sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne]
        for _ in range(100):
            atom = rels[int(rng.integers(0, 6))](
                self._rand_arith(rng), self._rand_arith(rng)
            )
            point = {
                s: sympy.Rational(int(rng.integers(-40, 40)),
                                  int(rng.integers(1, 7)))
                for s in (a, b, c)
            }
            truth = bool(atom.xreplace(point))
            got = self._truth_via_z3(atom, point)
            if got is not None:
                assert got == truth, (str(atom), point, truth, got)
            ngot = self._truth_via_z3(sympy.Not(atom), point)
            if ngot is not None:
                assert ngot == (not truth), (str(atom), point)
