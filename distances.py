import numpy as np

__all__ = ['norm', 'dst_sphere', 'dst_cube', 'intersect', 'union', 'substraction', 'unit']


def norm(v):
    return np.sqrt(np.einsum('...k,...k->...', v, v))


def unit(v):
    return v / norm(v)[..., None]


def dst_sphere(p: np.array, center: np.array, radius: float):
    return norm(p - center) - radius


def dst_cube(p: np.array, center, size):
    offset = abs(p - center) - size
    return norm(np.maximum(offset, 0)) + np.max(np.minimum(offset, 0))


def intersect(d1, d2):
    return np.maximum(d1, d2)


def union(d1, d2):
    return np.minimum(d1, d2)


def substraction(d1, d2):
    return np.maximum(d1, -d2)
