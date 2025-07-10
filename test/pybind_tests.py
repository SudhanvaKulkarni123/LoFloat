##@author Sudhanva Kulkarni
##test file for checking pybinds
import sys
import os
import torch


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
print(f"Loading module from: {module_path}")
sys.path.insert(0, module_path)


import LoFloat as lo_float

def main():
    # Ensure type exists
    if not hasattr(lo_float, "binary8p5se"):
        raise RuntimeError("Type binary8p4sf not found in module!")

    Float8 = lo_float.binary8p5se

    # Create instances
    a = Float8(1.5)
    b = Float8(2.0)

    # Print instances
    print(f"a = {a}")
    print(f"b = {b}")

    # Test arithmetic
    c = a + b
    print(f"a + b = {c}")

    d = a * b
    print(f"a * b = {d}")

    # Test comparisons
    print(f"a == b? {a == b}")
    print(f"a < b? {a < b}")

    # Convert to float
    print(f"float(a) = {float(a)}")

    #tst pytorch compatibilty
    print(torch.ones((2,4), dtype=torch.bfloat16))

if __name__ == "__main__":
    main()
