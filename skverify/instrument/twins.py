"""Building instrumented twins of functions, classes and methods.

The entry point is :func:`instrument`. A twin is a re-executed copy
of the original's source with the rewriter's call-site replacements,
its namespace seeded with the runtime twins. Classes twin their whole
base chain so ``super()`` stays instrumented; decorated members peel
to their inner functions; closure cells snapshot as trace constants.
Caches live on the active TraceSession, so no twin outlives a trace's
assumptions.
"""

import __future__

import ast
import inspect
import textwrap


from ..pair import _loop_end, _loop_iter
from ..session import current as _session
from .registries import OPAQUE_CALLABLES
from .rewriter import _Rewriter
from .runtime import (
    _skv_at,
    _skv_classof,
    _skv_clip,
    _skv_cmp,
    _skv_concrete,
    _skv_concrete_call,
    _skv_dict,
    _skv_empty,
    _skv_eye,
    _skv_finfo,
    _skv_float,
    _skv_full,
    _skv_getitem,
    _skv_identity,
    _skv_isinstance,
    _skv_isscalar,
    _skv_method,
    _skv_namespace,
    _skv_neutral,
    _skv_ones,
    _skv_opaque_out,
    _skv_scalarize,
    _skv_set,
    _skv_zeros,
)
from .triage import _skv_maybe, _skv_opaque, _twinnable


def instrument(fn, depth=3):
    """A semantically identical copy of fn with math-neutral calls
    replaced by traced twins. Returns (fn_copy, sites); on any failure
    returns (fn, ()) so tracing proceeds uninstrumented."""
    try:
        if getattr(fn, "__closure__", None):
            inner = fn
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__  # peel stacked decorators
            if inner is fn:
                inner = None
            if inspect.isfunction(inner) and not inner.__closure__:
                # a wrapped function is just two functions: trace the
                # inner one; to_sympy verifies the wrapper was neutral
                # for this call by rerunning the real thing on values
                sub, sites = _instrument(inner, depth, seen=set())
                return sub, (f"{fn.__name__}: decorator unwrapped",) + sites
            try:
                cells = {
                    name: cell.cell_contents
                    for name, cell in zip(
                        fn.__code__.co_freevars, fn.__closure__
                    )
                }
            except ValueError:
                return fn, ()  # unfilled cells
            if "__class__" not in cells:
                # a plain closure: its cells are this call's constants.
                # Snapshot them into the twin's namespace -- the re-exec'd
                # def resolves the names there
                sub, sites = _instrument(fn, depth, seen=set(), extra=cells)
                if sites:
                    return sub, (
                        f"{fn.__name__}: closure cells bound as trace constants",
                    ) + sites
                return fn, ()
            return fn, ()  # methods with __class__ cells stay untouched here
        return _instrument(fn, depth, seen=set())
    except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
        return fn, ()


def _instrument(fn, depth, seen, extra=None):
    source = textwrap.dedent(inspect.getsource(fn))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # getsource of a lambda inside a larger expression can return a
        # fragment; grab the lambda subexpression instead
        tree = None
    if tree is None or not isinstance(tree.body[0], (ast.FunctionDef,)):
        lam = None
        if tree is not None:
            # a source line can hold SEVERAL lambdas; pick the one
            # whose parameters match the live function object, or the
            # first as a last resort
            want = fn.__code__.co_varnames[: fn.__code__.co_argcount]
            candidates = [
                n for n in ast.walk(tree) if isinstance(n, ast.Lambda)
            ]
            for n in candidates:
                if tuple(a.arg for a in n.args.args) == tuple(want):
                    lam = n
                    break
            if lam is None and candidates:
                lam = candidates[0]
        if lam is None:
            raise TypeError("no function definition found in source")
        fdef = ast.FunctionDef(
            name="__skv_lambda__",
            args=lam.args,
            body=[ast.Return(value=lam.body)],
            decorator_list=[],
            type_params=[],
        )
        tree = ast.Module(body=[fdef], type_ignores=[])
        ast.fix_missing_locations(tree)
    else:
        fdef = tree.body[0]
    if fdef.decorator_list:
        # transparent decorators (metadata-only, returned the original
        # function: no closure, no __wrapped__) are safe to strip; a
        # wrapping decorator changes semantics, so bail. A __class__
        # cell alone is fine: super() is rewritten explicit
        only_class_cell = all(
            v == "__class__" for v in fn.__code__.co_freevars
        )
        if getattr(fn, "__wrapped__", None) is not None or (
            fn.__closure__ and not only_class_cell
        ):
            raise TypeError("wrapped functions are not instrumented")
        fdef.decorator_list = []
    rewriter = _Rewriter(fn.__globals__, tag=fn.__name__)
    if fdef.args.args:
        rewriter.first_arg = fdef.args.args[0].arg
    rewriter.visit(tree)
    ast.fix_missing_locations(tree)

    namespace = dict(fn.__globals__)
    if extra:
        namespace.update(extra)
    namespace["__skv_zeros__"] = _skv_zeros
    namespace["__skv_empty__"] = _skv_empty
    namespace["__skv_ones__"] = _skv_ones
    namespace["__skv_full__"] = _skv_full
    namespace["__skv_eye__"] = _skv_eye
    namespace["__skv_identity__"] = _skv_identity
    namespace["__skv_neutral__"] = _skv_neutral
    namespace["__skv_opaque__"] = _skv_opaque
    namespace["__skv_method__"] = _skv_method
    namespace["__skv_maybe__"] = _skv_maybe
    namespace["__skv_finfo__"] = _skv_finfo
    namespace["__skv_namespace__"] = _skv_namespace
    namespace["__skv_at__"] = _skv_at
    namespace["__skv_concrete__"] = _skv_concrete
    namespace["__skv_scalarize__"] = _skv_scalarize
    namespace["__skv_isinstance__"] = _skv_isinstance
    namespace["__skv_set__"] = _skv_set
    namespace["__skv_dict__"] = _skv_dict
    namespace["__skv_getitem__"] = _skv_getitem
    namespace["__skv_cmp__"] = _skv_cmp
    namespace["__skv_float__"] = _skv_float
    namespace["__skv_clip__"] = _skv_clip
    namespace["__skv_classof__"] = _skv_classof
    namespace["__skv_isscalar__"] = _skv_isscalar
    namespace["__skv_concrete_call__"] = _skv_concrete_call
    namespace["__skv_opaque_out__"] = _skv_opaque_out
    namespace["__skv_loop_iter__"] = _loop_iter
    namespace["__skv_loop_end__"] = _loop_end

    sites = list(rewriter.sites)
    if rewriter.wrapped and not sites:
        sites.append(f"auto-opaque guard on {rewriter.wrapped} calls")
    if depth > 0:
        for name in rewriter.callees:
            callee = fn.__globals__.get(name)
            if getattr(callee, "__name__", None) in OPAQUE_CALLABLES:
                continue  # will be sealed at the boundary, not entered
            _inner = callee
            while getattr(_inner, "__wrapped__", None) is not None:
                _inner = _inner.__wrapped__  # peel stacked decorators
            if (
                callable(callee)
                and getattr(callee, "__closure__", None)
                and _inner is not callee
                and inspect.isfunction(_inner)
                and not _inner.__closure__
                and callee.__name__ not in seen
            ):
                # decorator-wrapped callee: instrument the inner function
                seen.add(callee.__name__)
                try:
                    sub, sub_sites = _instrument(_inner, depth - 1, seen)
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    continue
                namespace[name] = sub
                sites.append(f"{name}: decorator unwrapped")
                sites.extend(f"{name}: {t}" for t in sub_sites)
                continue
            if inspect.isclass(callee):
                from .registries import CONCRETE_BY_NAME

                if callee.__name__ in CONCRETE_BY_NAME:
                    # sparse constructors and inventory classes run on
                    # CONCRETE values by doctrine; a pre-built twin
                    # here would shadow the runtime treatment and
                    # trace the construction anyway
                    continue
                try:
                    twin, twin_sites = _instrument_class(callee, depth, seen)
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    continue
                if twin_sites:
                    namespace[name] = twin
                    sites.extend(f"{name}: {t}" for t in twin_sites)
                continue
            if (
                inspect.isfunction(callee)
                and not getattr(callee, "__closure__", None)
                and callee.__module__ not in (None, "builtins")
            ):
                if callee in _FN_MEMO:
                    sub, sub_sites = _FN_MEMO[callee]
                elif callee.__name__ in seen:
                    continue  # cycle guard only; memo handles reuse
                else:
                    seen.add(callee.__name__)
                    try:
                        sub, sub_sites = _instrument(callee, depth - 1, seen)
                    except (
                        OSError,
                        TypeError,
                        SyntaxError,
                        KeyError,
                        AttributeError,
                    ):
                        continue
                    _FN_MEMO[callee] = (sub, sub_sites)
                if sub_sites:
                    namespace[name] = sub
                    sites.extend(f"{name}: {s}" for s in sub_sites)

    # Compile under PEP 563 (string annotations): source modules often
    # guard annotation-only names behind TYPE_CHECKING, and evaluating
    # them at re-exec time raises NameError on Pythons where
    # annotations are eager (3.11/3.12). We never introspect a twin's
    # annotations, so lazy strings are always safe.
    code = compile(
        tree,
        filename=f"<instrumented {fn.__name__}>",
        mode="exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace[fdef.name], tuple(sites)


from ..session import current as _session

# Historical aliases: the session owns the twin caches, so every trace
# starts with blank ones (order-independence by construction).
_CLASS_TWINS = _session.class_twins
_FN_MEMO = _session.fn_twins


def _instrument_class(C, depth, seen):
    """A parallel twin of the class: every method instrumented, the
    base twinned too so super() chains stay instrumented. Returns
    (twin_or_C, sites)."""
    if C in _CLASS_TWINS:
        return _CLASS_TWINS[C]
    if not inspect.isclass(C) or C is object or not _twinnable(C):
        return C, ()
    key = f"class:{C.__module__}.{C.__qualname__}"
    if key in seen:
        return C, ()
    seen.add(key)

    bases, base_sites = [], []
    for b in C.__bases__:
        if b is object:
            bases.append(b)
            continue
        tb, ts = _instrument_class(b, depth, seen)
        bases.append(tb)
        base_sites.extend(ts)
    if all(b is object for b in bases):
        bases = [object]
    members = {
        k: v
        for k, v in vars(C).items()
        if k not in ("__dict__", "__weakref__")
    }
    slots = members.pop("__slots__", None)
    if slots is not None:
        # the twin gets a plain __dict__; slot descriptors would clash
        for slot in ((slots,) if isinstance(slots, str) else slots):
            members.pop(slot, None)
    sites = [f"{C.__name__}(base): {t}" for t in base_sites]
    patch_namespaces = []
    for name, raw in list(members.items()):
        if name in (
            "__init_subclass__",
            "__class_getitem__",
            "__subclasshook__",
            "__set_name__",
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
        ):
            continue  # attribute plumbing: instrumenting it recurses
        wrap, target = None, None
        if isinstance(raw, staticmethod):
            target, wrap = raw.__func__, staticmethod
        elif isinstance(raw, classmethod):
            target, wrap = raw.__func__, classmethod
        elif isinstance(raw, property):
            if raw.fset is None and raw.fdel is None and inspect.isfunction(raw.fget):
                target, wrap = raw.fget, property
        elif (
            hasattr(raw, "fget")
            and inspect.isfunction(getattr(raw, "fget", None))
            and type(raw).__module__ not in ("builtins",)
        ):
            # cache_readonly-style descriptors (statsmodels, pandas):
            # twin the wrapped function, rewrap in the SAME descriptor
            # type so caching semantics survive
            target = raw.fget
            wrap = type(raw)
        elif inspect.isfunction(raw):
            target = raw
        elif callable(raw) and getattr(raw, "__wrapped__", None) is not None:
            # decorator-wrapped method (xp_capabilities): peel to the
            # inner function, same as module-level callees
            inner = raw
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__
            if inspect.isfunction(inner):
                target = inner
        if target is None or not inspect.isfunction(target):
            continue  # classmethod(GenericAlias) and similar non-functions
        freevars = getattr(target.__code__, "co_freevars", ())
        if any(v != "__class__" for v in freevars):
            # a decorated method: the wrapper is a closure, but the
            # real function underneath may not be -- peel it
            inner = target
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__
            if inner is target and target.__closure__:
                # wrappers without __wrapped__ (deprecation shims) hide
                # the real method in a closure cell: find it by name
                for cell in target.__closure__:
                    try:
                        held = cell.cell_contents
                    except ValueError:
                        continue
                    if inspect.isfunction(held) and held.__name__ == name:
                        inner = held
                        break
            if (
                inner is not target
                and inspect.isfunction(inner)
                and all(
                    v == "__class__"
                    for v in getattr(inner.__code__, "co_freevars", ())
                )
            ):
                target = inner
            else:
                chain = [target]
                while getattr(chain[-1], "__wrapped__", None) is not None:
                    chain.append(chain[-1].__wrapped__)
                if any(
                    "__class__"
                    in getattr(getattr(f, "__code__", None), "co_freevars", ())
                    for f in chain
                ):
                    # an unpeelable super()-using member: a parallel twin
                    # would break its super chain. No twin for this class
                    _CLASS_TWINS[C] = (C, ())
                    return C, ()
                continue  # real closures don't survive re-exec
        try:
            tree_fn = _instrument(target, depth - 1, seen)
        except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
            continue
        sub, sub_sites = tree_fn
        if sub_sites:
            members[name] = wrap(sub) if wrap else sub
            patch_namespaces.append(sub.__globals__)
            sites.extend(f"{C.__name__}.{name}: {t}" for t in sub_sites)
    if not sites:
        _CLASS_TWINS[C] = (C, ())
        return C, ()
    # methods carried over UNinstrumented but holding the original
    # class in their __class__ cell would break super(): clone them
    # with a fresh cell and rebind it to the twin
    import types

    rebind = []
    for name, raw in list(members.items()):
        target = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        if (
            inspect.isfunction(target)
            and getattr(target.__code__, "co_freevars", ()) == ("__class__",)
            and target.__closure__ is not None
        ):
            cell = types.CellType(None)
            clone = types.FunctionType(
                target.__code__,
                target.__globals__,
                target.__name__,
                target.__defaults__,
                (cell,),
            )
            clone.__kwdefaults__ = target.__kwdefaults__
            wrapper_kind = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
            members[name] = wrapper_kind(clone) if wrapper_kind else clone
            rebind.append(cell)
    # the twin's namespace holds EVERY member of C (instrumented or
    # raw), so inheriting from C itself adds no behavior -- but it puts
    # the original in the MRO, so isinstance(twin_instance, C) and
    # every library type-gate built on it hold. Twinned bases stay
    # ahead of C so instrumented inherited methods still win.
    final_bases = [b for b in bases if b is not object]
    if C not in final_bases:
        final_bases.append(C)
    meta = type(C)
    twin = None
    for attempt in (tuple(final_bases), tuple(bases)):
        for m in (meta, type):
            try:
                twin = m(C.__name__, attempt, dict(members))
                break
            except TypeError:
                continue
        if twin is not None:
            break
    if twin is None:
        _CLASS_TWINS[C] = (C, ())
        return C, ()
    for cell in rebind:
        cell.cell_contents = twin
    # explicit ancestor references -- super(Base, self) and direct
    # Base.__init__(self, ...) calls -- resolve by NAME in the method's
    # namespace; left alone they name the raw class, whose presence in
    # the twin's MRO (the isinstance keystone) would route the rest of
    # the chain through uninstrumented code
    ancestors = {C.__name__: twin}
    for anc in C.__mro__[1:]:
        hit = _CLASS_TWINS.get(anc)
        if hit and hit[0] is not anc:
            ancestors[anc.__name__] = hit[0]
    for ns in patch_namespaces:
        ns["__skv_class__"] = twin
        ns.update(ancestors)
    _CLASS_TWINS[C] = (twin, tuple(sites))
    return twin, tuple(sites)
