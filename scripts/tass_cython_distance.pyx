# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

"""
Fast local-distance operations for float32 images/matrices.

Functions
---------
compute_minimization_offset(distanceMat, offset)
    Square-window minimum filter over a 2-D float32 matrix.

compute_distance_offset(imgA, imgB, offset)
    For every RGB pixel in imgA, find the minimum Euclidean RGB distance
    to a pixel of imgB inside the square window of radius ``offset``.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrtf, INFINITY

cnp.import_array()


def compute_minimization_offset(
        const cnp.float32_t[:, ::1] distanceMat,
        int offset):
    """
    Compute the minimum value in a square neighborhood around each element.

    Parameters
    ----------
    distanceMat : 2-D float32 array
        Input matrix. The last dimension must be contiguous.
    offset : int
        Search radius. The window size is (2*offset + 1)^2, clipped at
        the matrix boundaries.

    Returns
    -------
    numpy.ndarray, dtype=float32
        Matrix of local minima with the same height and width as distanceMat.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    cdef Py_ssize_t H = distanceMat.shape[0]
    cdef Py_ssize_t W = distanceMat.shape[1]

    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = \
        np.empty((H, W), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] distanceMin = result

    cdef Py_ssize_t i, j, ki, kj
    cdef Py_ssize_t i0, i1, j0, j1
    cdef float min_dist, dist

    for i in range(H):
        i0 = i - offset
        if i0 < 0:
            i0 = 0

        i1 = i + offset + 1
        if i1 > H:
            i1 = H

        for j in range(W):
            j0 = j - offset
            if j0 < 0:
                j0 = 0

            j1 = j + offset + 1
            if j1 > W:
                j1 = W

            min_dist = INFINITY

            for ki in range(i0, i1):
                for kj in range(j0, j1):
                    dist = distanceMat[ki, kj]
                    if dist < min_dist:
                        min_dist = dist

            distanceMin[i, j] = min_dist

    return result


def compute_distance_offset(
        const cnp.float32_t[:, :, ::1] imgA,
        const cnp.float32_t[:, :, ::1] imgB,
        int offset):
    """
    For each pixel in imgA, compute the minimum RGB Euclidean distance
    to pixels in imgB within a square search neighborhood.

    The squared RGB distance is minimized in the inner loop and sqrtf()
    is evaluated only once for each output pixel.

    Parameters
    ----------
    imgA, imgB : 3-D float32 arrays
        Arrays of shape (H, W, 3). The last dimension must be contiguous.
    offset : int
        Search radius. The window size is (2*offset + 1)^2, clipped at
        the image boundaries.

    Returns
    -------
    numpy.ndarray, dtype=float32
        2-D array of shape (H, W).
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    if (imgA.shape[0] != imgB.shape[0] or
        imgA.shape[1] != imgB.shape[1] or
        imgA.shape[2] != imgB.shape[2]):
        raise ValueError("imgA and imgB must have the same shape")

    if imgA.shape[2] != 3:
        raise ValueError("input images must have exactly 3 channels")

    cdef Py_ssize_t H = imgA.shape[0]
    cdef Py_ssize_t W = imgA.shape[1]

    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = \
        np.empty((H, W), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] distance = result

    cdef Py_ssize_t i, j, ki, kj
    cdef Py_ssize_t i0, i1, j0, j1

    cdef float a0, a1, a2
    cdef float d0, d1, d2
    cdef float dist, min_dist

    for i in range(H):
        i0 = i - offset
        if i0 < 0:
            i0 = 0

        i1 = i + offset + 1
        if i1 > H:
            i1 = H

        for j in range(W):
            j0 = j - offset
            if j0 < 0:
                j0 = 0

            j1 = j + offset + 1
            if j1 > W:
                j1 = W

            # imgA[i, j] is invariant over the complete search window.
            a0 = imgA[i, j, 0]
            a1 = imgA[i, j, 1]
            a2 = imgA[i, j, 2]

            min_dist = INFINITY

            for ki in range(i0, i1):
                for kj in range(j0, j1):
                    d0 = a0 - imgB[ki, kj, 0]
                    d1 = a1 - imgB[ki, kj, 1]
                    d2 = a2 - imgB[ki, kj, 2]

                    dist = d0 * d0 + d1 * d1 + d2 * d2

                    if dist < min_dist:
                        min_dist = dist

            distance[i, j] = sqrtf(min_dist)

    return result
