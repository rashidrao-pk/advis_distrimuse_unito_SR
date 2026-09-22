# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Compiled TAAS spatial-neighbourhood distance calculation."""

from libc.math cimport sqrtf
import numpy as np
cimport numpy as cnp


def distance_offset(
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] image_a,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] image_b,
    int offset,
):
    """Return minimum RGB L2 distance within a square spatial neighbourhood."""
    cdef:
        Py_ssize_t height = image_a.shape[0]
        Py_ssize_t width = image_a.shape[1]
        Py_ssize_t channels = image_a.shape[2]
        Py_ssize_t row, col, other_row, other_col, channel
        Py_ssize_t row_start, row_stop, col_start, col_stop
        float delta, squared, best
        cnp.ndarray[cnp.float32_t, ndim=2] output

    if image_b.shape[0] != height or image_b.shape[1] != width:
        raise ValueError("TAAS input image shapes differ")
    if image_b.shape[2] != channels:
        raise ValueError("TAAS input channel counts differ")
    if channels < 1:
        raise ValueError("TAAS input has no channels")
    if offset < 0:
        raise ValueError("TAAS offset must be non-negative")

    output = np.empty((height, width), dtype=np.float32)
    with nogil:
        for row in range(height):
            row_start = row - offset
            if row_start < 0:
                row_start = 0
            row_stop = row + offset + 1
            if row_stop > height:
                row_stop = height
            for col in range(width):
                col_start = col - offset
                if col_start < 0:
                    col_start = 0
                col_stop = col + offset + 1
                if col_stop > width:
                    col_stop = width
                best = 3.402823466e+38
                for other_row in range(row_start, row_stop):
                    for other_col in range(col_start, col_stop):
                        squared = 0.0
                        for channel in range(channels):
                            delta = (
                                image_a[row, col, channel]
                                - image_b[other_row, other_col, channel]
                            )
                            squared += delta * delta
                        if squared < best:
                            best = squared
                output[row, col] = sqrtf(best)
    return output
