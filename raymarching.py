#!/usr/bin/env python3
import profile

import pygame
import numpy as np

ALL_NORMALS = False

def pd(*v):
    print(*v)
    return v


def vec(x, y, z):
    return np.array((x, y, z), dtype=np.float)


def norm(v):
    return np.sqrt(np.einsum('...k,...k->...', v, v))


def dst_sphere(p: np.array, center: np.array, radius: float):
    return norm(p - center) - radius


def dst_cube(p: np.array, center, size):
    offset = abs(p - center) - size
    return norm(np.maximum(offset, 0)) + np.max(np.minimum(offset, 0))


class Camera:
    bottomleft: vec
    horizontal: vec
    vertical: vec
    origin: vec

    def __init__(self):
        self.bottomleft = vec(-2, -1, -1)
        self.horizontal = vec(4, 0, 0)
        self.vertical = vec(0, 2, 0)
        self.origin = vec(0, 0, 0)

    def genRays(self, reso):
        x = np.linspace(0, 1, reso[0])
        y = np.linspace(1, 0, reso[1])
        ys, xs = np.meshgrid(y, x)
        a = self.horizontal * np.array(xs)[:, :, None]
        b = self.vertical * np.array(ys)[:, :, None]
        directions = a + b + (self.bottomleft - self.origin)
        origin = self.origin[None, :].repeat(reso[0] * reso[1], axis=0).reshape(reso + (3,))

        directions /= norm(directions[:, :, None])

        assert origin.shape == directions.shape == (reso + (3,))
        return origin, directions


def color(points, directions, dist_func):
    assert points.shape == directions.shape

    EPS = 0.0001
    HIT = 0.001
    shape = points.shape

    running = np.ones(shape[:-1], dtype=np.bool)

    dists = dist_func(points)
    steps = np.zeros(shape[:-1], dtype=np.int)

    i = 0
    while np.any(running):

        points[running] += dists[running, None] * directions[running]
        dists[running] = dist_func(points[running])

        running = (HIT < dists) & (dists < 3)
        steps += running

        print(i, np.sum(running))
        i += 1


    if not ALL_NORMALS:
        hits = dists < HIT
        normals = vec(0, 1 - 0.647, 1)[None, :].repeat(shape[0] * shape[1], axis=0).reshape(shape)
        normals[hits] = dists[hits, None]
        normals[hits, 0] -= dist_func(points[hits] + vec(EPS, 0, 0))
        normals[hits, 1] -= dist_func(points[hits] + vec(0, EPS, 0))
        normals[hits, 2] -= dist_func(points[hits] + vec(0, 0, EPS))
        normals[hits] /= norm(normals[hits])[:, None]


        img = (255 * (0.5 + 0.5 * normals + steps[..., None] / steps.max())).astype(int)

    else:
        normals = np.zeros(shape)
        normals[:] = dists[:, :, None]
        normals[:, :, 0] -= dist_func(points + vec(EPS, 0, 0))
        normals[:, :, 1] -= dist_func(points + vec(0, EPS, 0))
        normals[:, :, 2] -= dist_func(points + vec(0, 0, EPS))
        normals /= norm(normals)[:, :, None]

        img = (255 * (0.5 + 0.5 * normals)).astype(int)

    return np.minimum(img, 255)

def intersect(d1, d2):
    return np.maximum(d1, d2)

def union(d1, d2):
    return np.minimum(d1, d2)

def substraction(d1, d2):
    return np.maximum(d1, -d2)

def dst_function(p):
    # return dst_cube(p, vec(0.7, 0.7, -2), 0.5)
    return substraction(
        intersect(
            dst_sphere(p, vec(0, 0, -2), 0.5),
            dst_cube(p, vec(0, 0, -2), 0.4)
        ),
        dst_sphere(p, vec(0, 0, -2), 0.45)
    )


def main():
    SIZE = 800, 400

    screen: pygame.SurfaceType = pygame.display.set_mode(SIZE)
    camera = Camera()

    point, direction = camera.genRays(SIZE)
    img = color(point, direction, dst_function)

    # img **= 3
    # img[img < 0.3] = 0
    # print(img.max())
    # img = img * 255 / img.max()
    # img = np.repeat(img.astype(np.int), 3).reshape(SIZE + (3,))
    pygame.surfarray.pixels3d(screen)[::] = img

    stop = False
    while not stop:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                stop = True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    stop = True

        pygame.display.update()


if __name__ == '__main__':
    main()
