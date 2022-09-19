# ham

tiny stack-based maybe-a-little-esoteric multi-precision reverse-polish-notation calculator

## features

1. objectively cooler than infix notation
2. arbitrary precision with both integers and fractions where possible
3. complex number support for all functions (automatic domain)
4. automatic type promotion (and demotion): the most appropriate type is always
  used. dividing two `int`s will result in a `Fraction`, xoring a `Fraction`
  will result in a `int`.
5. safe<sup>[disputed]</sup>

## spec

(subject to changes!!)

expressions are simply sequences of constants and operators. constants push
themselves into the stack, and operators pop from the top of the stack and push
back to it. that is, instead of `1 + 2` or `abs(-1)`, you write `1 2 +` and
`-1 abs`.

`pi`, `tau` and `e` are valid constants

numbers may be formed from the following:
- simple decimals: `143`, `1.23`, `.33333`
- prefixed bases: `0b10001111`, `0xfa50a0`, `0o777`, `0x1.abc`
- exponents: `1.337e3`, `0x5.39p8` (ieee 754 hex exponent)
- arbitrary padding: `0b1010_0101_1010`, `1_2_3_4`, `1_______3`

the following operators are supported:
- `+`, `-`, `*`, `/`: basic arithmetic
- `%`: modulo
- `**`: exponent
- `^`, `&`, `|`: bitwise operators
- `<<`, `>>`: left/right bit shift
- `~`: unary not
- `abs`: absolute value
- `sqrt`: square root
- `ln`: natural log
- `sin`, `cos`, `tan`: trigonometric functions
- `rad`, `deg`: radians/degrees conversion
- `phase`: phase (aka argument) of a complex number
- `swap`: swap the top two values of the stack
- `dup`: duplicate the top value of the stack
- `pop`: discard the top value of the stack
- `splat`: unpack an array to the stack

the special operator `@`, suffixed with a shape (e.g. `@4,4`) will pop values
from the stack and pack them into a ndarray:
- `1 2 3 4 @4` → `[1, 2, 3, 4]`
- `1 2 3 1 2 3 @2,3` → `[[1, 2, 3], [1, 2, 3]]`

ndarrays are supported by all operators, and work element-wise:
`0 1 2 3 @4 2 **` → `0 1 4 9 @4`
