from __future__ import annotations
from typing import Type
import sys


class LoF_Number:
    """Base numeric type: subclasses must implement __float__()."""

    def __add__(self, y: "LoF_Number"):
        return float(self) + float(y)

    def __mul__(self, y: "LoF_Number"):
        return float(self) * float(y)

    def __sub__(self, y: "LoF_Number"):
        return float(self) - float(y)

    def __truediv__(self, y: "LoF_Number"):
        return float(self) / float(y)


def LoF_FloatFactory(bitwidth: int, num_mantissa: int, exp_bias: int) -> Type[LoF_Number]:
    """
    Returns a new LoF_Float class with static (class-level) fields:
      - bitwidth
      - num_mantissa
      - exp_bias
    """

    class LoF_Float(LoF_Number):
        def __init__(self, value=0.0):
            self._value = float(value)

        def __float__(self) -> float:
            return self._value

        def __repr__(self) -> str:
            return (
                f"LoF_Float<{self.bitwidth}, m={self.num_mantissa}, bias={self.exp_bias}>"
                f"({self._value})"
            )

    # ✅ assign static fields AFTER class creation
    LoF_Float.bitwidth = bitwidth
    LoF_Float.num_mantissa = num_mantissa
    LoF_Float.exp_bias = exp_bias

    return LoF_Float


# --- Example usage ---
if __name__ == "__main__":
    FP10 = LoF_FloatFactory(bitwidth=10, num_mantissa=5, exp_bias=7)
    FP8 = LoF_FloatFactory(bitwidth=8, num_mantissa=5, exp_bias=0)

    a = FP10(5.0)
    b = FP10(2.0)

    c = FP8(1.0)

    print("Static fields:", FP10.bitwidth, FP10.num_mantissa, FP10.exp_bias)
    print("Static fields:", FP8.bitwidth, FP8.num_mantissa, FP8.exp_bias)
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("a + b =", a + b)
    print("a * b =", a * b)
    print("a - b =", a - b)
    print("a / b =", a / b)
    print("sizeof(a)", sys.getsizeof(a))
