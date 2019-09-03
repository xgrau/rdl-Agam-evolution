# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


cimport numpy as cnp
import numpy as np


def count_gametes(cnp.int8_t[:, :] h):
    """Count the number of each gametic type observed for each pair of variants.
    Observation of all four gametic types for any pair of variants is evidence for
    recombination."""

    cdef:
        Py_ssize_t n, m, i, j, k
        cnp.uint8_t[:, :] d
        cnp.uint32_t[:, :, :, :] count

    n = h.shape[0]
    m = h.shape[1]
    count = np.zeros((n, n, 2, 2), dtype='u4')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                count[i, j, h[i, k], h[j, k]] += 1

    return np.asarray(count)