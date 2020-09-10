import numpy as np

import numba
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.typed import Dict, List

import numba
from numba import generated_jit, njit


# @generated_jit(nopython=True)
# def fn(key, value):
#     dict_type = numba.types.DictType(numba.types.unicode_type, value)
#     value_type = value

#     def impl(key, value):
#         d = numba.typed.Dict.empty(numba.types.unicode_type, value_type)
#         d[key] = value
#         return d


#     return impl

@numba.njit
def impl(x, keys, values):
    for i, item in x:  #enumerate(numba.literal_unroll(x)):
        item[keys[i]]  = values[i]
    return x

if __name__=="__main__":
    
    nlevels = 3
    # x = nlevels*(numba.typed.Dict.empty(numba.types.unicode_type, numba.types.SliceType),)
    x = nlevels*(numba.typed.Dict(),)
    keys = List(['dd', 'ad', 'da'])
    
    values = List([numba.types.UniTuple(dtype=array.dtype, count=tuple_size)(1, 2), (5, 8), (8,10)])
    # # # values = List([np.random.randn(5, 5), np.random.randn(4, 4), np.random.randn(3, 3)])
    x = impl(x, keys, values)

    # print(fn('key', slice(0,10)))