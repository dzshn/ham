import abc
import cmath
import dataclasses
import math
import operator as op
import re
from collections import deque
from collections.abc import Callable, Iterable
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from numbers import Rational
from typing import Any, TypeVar, Union

import numpy as np

# (´；ω；`) <https://github.com/python/mypy/issues/3186>
RationalT = int | Fraction
RealT = float | Decimal | RationalT
ComplexT = complex | RealT

R = TypeVar("R", bound=RealT)
C = TypeVar("C", bound=ComplexT)
Poly = Union[C, np.ndarray[C, Any]]
Obj = Poly[ComplexT]

NUMBER = re.compile(
    r"""
    [+-]?
    (
        0b[01_]+(\.[0-1_]+)?           # binary       (0b1000_1111)
        |0o[0-7_]+(\.[0-7_]+)?         # octal        (0o333)
        |(0x[0-9a-f_]+(\.[0-9a-f_]+)?) # hexadecimal  (0xfa50a0)
        (p[+-]?[0-9_]+)?               # hex exponent (0x1.ABCp123)
        |(
            [0-9][0-9_]*(\.[0-9_]*)?   # decimal      (1337)
            |\.[0-9_]+ )               # trailing dot (.123)
        (e[+-]?[0-9_]+)?               # exponent     (1e-4)
    )
    [ij]?
    """,
    re.VERBOSE | re.IGNORECASE,
)
PACK_OP = re.compile(r"@([0-9]+,?)+")
SEPARATORS = re.compile(r"[ ;\n]+")
WORD = re.compile(r"[a-z]+|[^ ;\n]+|")
BASES = {
    "0b": 2,
    "0o": 8,
    "0x": 16,
}


def convert(
    t: type[ComplexT], args: Iterable[Poly[ComplexT]]
) -> tuple[Poly[ComplexT], ...]:
    if t is Decimal:
        def converter(x: Any) -> Poly[ComplexT]:
            if isinstance(x, Rational):
                return Decimal(x.numerator) / Decimal(x.denominator)
            if isinstance(x, np.ndarray):
                return x.astype(object) * Decimal(1)
            return Decimal(x)
    elif t is Fraction:
        def converter(x: Any) -> Poly[ComplexT]:
            if isinstance(x, np.ndarray):
                return x.astype(object) * Fraction(1)
            return Fraction(x)
    else:
        def converter(x: Any) -> Poly[ComplexT]:
            if isinstance(x, np.ndarray):
                return x.astype(t)
            return t(x)
    return tuple(map(converter, args))


class Symbol(abc.ABC):
    @abc.abstractmethod
    def dispatch(self, stack: deque[Obj]) -> None:
        ...


@dataclasses.dataclass
class Single(Symbol):
    fn: Callable[..., Obj]
    args: int | None = None
    require: type[ComplexT] | None = None
    err: Callable | None = None

    def dispatch(self, stack: deque[Obj]) -> None:
        if self.args is None:
            try:
                res = self.fn(stack)
            except (ArithmeticError, ValueError) as e:
                if self.err is None:
                    raise e from None
                res = self.err()
        else:
            try:
                args = tuple(stack.pop() for _ in range(self.args))[::-1]
                if self.require is not None:
                    args = convert(self.require, args)
            except IndexError as e:
                if self.err is None:
                    raise e from None
                res = self.err()
            else:
                try:
                    res = self.fn(*args)
                except (ArithmeticError, ValueError) as e:
                    if self.err is None:
                        raise e from None
                    res = self.err(*args)
        if isinstance(res, tuple):
            stack.extend(res)
        else:
            stack.append(res)


@dataclasses.dataclass
class Polymorphic(Symbol):
    rational: Callable
    complex: Callable
    args: int
    r_require: type | None = None
    c_require: type | None = None
    err: Callable | None = None

    def dispatch(self, stack: deque[Obj]) -> None:
        try:
            args = [stack.pop() for _ in range(self.args)][::-1]
            r_args = args
            c_args = args
        except IndexError as e:
            if self.err is None:
                raise e from None
            res = self.err()
        else:
            try:
                if all(isinstance(x, Rational) for x in args):
                    try:
                        if self.r_require is not None:
                            r_args = convert(self.r_require, args)
                        res = self.rational(*r_args)
                    except (ArithmeticError, TypeError, ValueError):
                        if self.c_require is not None:
                            c_args = convert(self.c_require, args)
                        res = self.complex(*c_args)
                else:
                    if self.c_require is not None:
                        c_args = convert(self.c_require, args)
                    res = self.complex(*c_args)
            except (ArithmeticError, ValueError) as e:
                if self.err is None:
                    raise e from None
                res = self.err(*args)
        if isinstance(res, tuple):
            stack.extend(res)
        else:
            stack.append(res)


@dataclasses.dataclass
class Constant(Symbol):
    value: ComplexT

    def dispatch(self, stack: deque[Obj]) -> None:
        stack.append(self.value)


def e_inf(*args: ComplexT) -> float:
    return float("inf")


def e_sinf(*args: RationalT) -> float:
    return args[0] * float("inf")


def e_nan(*args: ComplexT) -> float:
    return float("nan")


def e_nop(*args: C) -> tuple[C, ...]:
    return args


def swap(a: ComplexT, b: ComplexT) -> tuple[ComplexT, ComplexT]:
    return b, a


def dup(a) -> tuple[Obj, Obj]:
    return a, a


def pop(a: Obj) -> tuple[()]:
    return ()


def splat(a: np.ndarray[ComplexT, Any]) -> tuple[ComplexT, ...]:
    return tuple(a.flat)


stdlib = {
    "+": Single(op.add, 2),
    "-": Single(op.sub, 2),
    "*": Single(op.mul, 2),
    "/": Polymorphic(op.truediv, op.truediv, 2, Fraction, None, e_sinf),
    "%": Single(op.mod, 2, Fraction, e_nan),
    "**": Single(op.pow, 2, Decimal, e_inf),
    "^": Single(op.xor, 2, int),
    "&": Single(op.and_, 2, int),
    "|": Single(op.or_, 2, int),
    ">>": Single(op.rshift, 2, int),
    "<<": Single(op.lshift, 2, int),
    "~": Single(op.invert, 1, int),
    "abs": Single(abs, 1),
    "sqrt": Polymorphic(Decimal.sqrt, np.emath.sqrt, 1, Decimal),
    "ln": Polymorphic(Decimal.ln, np.emath.log, 1, Decimal, err=e_inf),
    "sin": Polymorphic(math.sin, np.sin, 1),
    "cos": Polymorphic(math.cos, np.cos, 1),
    "tan": Polymorphic(math.tan, np.tan, 1),
    "rad": Single(math.radians, 1),
    "deg": Single(math.degrees, 1),
    "phase": Single(cmath.phase, 1),
    "pi": Constant(math.pi),
    "tau": Constant(math.tau),
    "e": Constant(math.e),
    "swap": Single(swap, 2, err=e_nop),
    "dup": Single(dup, 1, err=e_nop),
    "pop": Single(pop, 1, err=e_nop),
    "splat": Single(splat, 1, err=e_nop),
}


def parse_number(num: str) -> ComplexT:
    exponent: RealT = -1 if num.startswith("-") else +1
    num = num.lstrip("+-")
    base = BASES.get(num[:2], 10)
    num = num.lstrip("0box")
    imag = num.endswith(("i", "j"))
    num = num.rstrip("ij")
    if base == 10 and "e" in num:
        num, x = num.split("e")
        x = x.replace("_", "")
        try:
            10 ** float(x)
        except OverflowError:
            exponent *= Decimal(10) ** int(x)
        else:
            exponent *= 10 ** abs(int(x))
            if x.startswith("-"):
                exponent = Fraction(1, exponent)
    elif base == 16 and "p" in num:
        num, x = num.split("p")
        x = x.replace("_", "")
        try:
            2 ** float(x)
        except OverflowError:
            exponent *= Decimal(2) ** int(x)
        else:
            exponent *= 2 ** abs(int(x))
            if x.startswith("-"):
                exponent = Fraction(1, exponent)
    intp, _, frac = num.partition(".")
    value: RationalT = int(intp or "0", base)
    value += Fraction(int(frac or "0", base), base ** len(frac))
    if isinstance(exponent, RationalT):
        value *= exponent
    else:
        value = Decimal(value.numerator) / Decimal(value.denominator)
    if imag:
        return value * 1j
    if value.denominator == 1:
        return int(value)
    return value


def unparse_number(num: ComplexT) -> str:
    if num == float("inf"):
        return "inf"
    if num == float("-inf"):
        return "-inf"
    if num == float("nan"):
        return "nan"
    if isinstance(num, Fraction):
        return str(Decimal(num.numerator) / Decimal(num.denominator))
    if isinstance(num, complex):
        if num.real == 0 and num.imag != 0:
            return f"{num.imag:g}i"
        if num.imag > 0:
            return f"{num.real:g} {num.imag:g}i +"
        if num.imag < 0:
            return f"{num.real:g} {num.imag:g}i -"
        return f"{num.real:g}"
    return str(num)


def evaluate(src: str, stack: Iterable[Obj] = []) -> deque[Obj]:
    stack = deque(stack)

    def take(n: int) -> list[Obj]:
        return [stack.pop() for _ in range(n)][::-1]  # type: ignore

    with localcontext(Context(prec=32, capitals=False, traps=[])):
        p = 0
        while p < len(src):
            if m := SEPARATORS.match(src[p:]):
                p += m.end()
            elif m := PACK_OP.match(src[p:]):
                p += m.end()
                shape = tuple(map(int, m.group().lstrip("@").split(",")))
                stack.append(np.array(take(math.prod(shape))).reshape(shape))
            elif m := NUMBER.match(src[p:]):
                p += m.end()
                stack.append(parse_number(m.group()))
            else:
                m = WORD.match(src[p:])
                assert m
                p += m.end()
                word = m.group()
                if word in stdlib:
                    stdlib[word].dispatch(stack)
                else:
                    raise ValueError(f"Unknown token @ {p} ({word})")

    return stack


def pretty_stack(stack: Iterable[Obj]) -> str:
    res = ""
    for i in stack:
        if isinstance(i, np.ndarray):
            for row in i.reshape((-1, i.shape[-1])):
                for x in row:
                    res += unparse_number(x) + " "
                res = res.strip() + ";"
            res = res.strip(";")
            res += " @" + ",".join(str(x) for x in i.shape)
        else:
            res += unparse_number(i)
        res += "; "
    return res.strip("; ") or ";"
