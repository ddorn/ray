import numpy as np
from pexpect import searcher_re

from distances import dst_sphere, dst_cube, substraction, union, intersect


class Object:
    def dst(self, p):
        return NotImplemented

    def __add__(self, other):
        return Combi(self, other, union)

    def __sub__(self, other):
        if isinstance(other, (int, float)) or callable(other):
            return Rounded(self, other)
        return Combi(self, other, substraction)

    def __and__(self, other):
        return Combi(self, other, intersect)

class Sphere(Object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def dst(self, p):
        return dst_sphere(p, self.center, self.radius)

class Cube(Object):
    def __init__(self, center, size):
        self.center = center
        self.size = size
    def dst(self, p):
        return dst_cube(p, self.center, self.size)

class Combi(Object):
    def __init__(self, a: Object, b: Object, blend_func):
        self.a = a
        self.b = b
        self.blend_func = blend_func

    def dst(self, p):
        return self.blend_func(self.a.dst(p), self.b.dst(p))

class Rounded(Object):
    def __init__(self, a, iso):
        self.a = a
        self.iso = iso

    def dst(self, p):
        if callable(self.iso):
            return self.a.dst(p) - self.iso(p)
        return self.a.dst(p) - self.iso