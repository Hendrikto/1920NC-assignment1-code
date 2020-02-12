from math import (
    cos,
    exp,
    log,
    sin,
)
from operator import (
    add,
    mul,
    sub,
    truediv,
)

import numpy as np
from deap import gp

x_values = np.linspace(-1, 1, num=21)
y_values = np.array((
    0,
    -0.1629,
    -0.2624,
    -0.3129,
    -0.3264,
    -0.3125,
    -0.2784,
    -0.2289,
    -0.1664,
    -0.0909,
    0,
    0.1111,
    0.2496,
    0.4251,
    0.6496,
    0.9375,
    1.3056,
    1.7731,
    2.3616,
    3.0951,
    4,
))

primitive_set = gp.PrimitiveSet('exercise8', 1)
primitive_set.renameArguments(ARG0='x')

primitive_set.addPrimitive(add, 2)
primitive_set.addPrimitive(cos, 1)
primitive_set.addPrimitive(exp, 1)
primitive_set.addPrimitive(log, 1)
primitive_set.addPrimitive(mul, 2)
primitive_set.addPrimitive(sin, 1)
primitive_set.addPrimitive(sub, 2)
primitive_set.addPrimitive(truediv, 2)
