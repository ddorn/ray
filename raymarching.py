#!/usr/bin/env python3

import pygame
import numpy as np


def pd(*v):
    print(*v)
    return v


def vec(x, y, z):
    return np.array((x, y, z), dtype=np.float)


def norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def dst_sphere(p: np.array, center: np.array, radius: float):
    return norm(p - center) - radius


def dst_cube(p: np.array, center, size):
    offset = abs(p - center) - size
    return norm(np.max(offset, 0)) + np.max(np.min(offset, 0))


class Camera:
    bottomleft: vec
    horizontal: vec
    vertical: vec
    origin: vec

    def __init__(self):
        self.bottomleft = vec(-2, -1, -1)
        self.horizontal = vec(4, 0, 0)
        self.vertical = vec(0, 2, 0)
        self.origin = vec(0, 1, 0)

    def genRays(self, reso):
        x = np.linspace(0, 1, reso[0])
        y = np.linspace(0, 1, reso[1])
        ys, xs = np.meshgrid(y, x)
        a = self.horizontal * np.array(xs)[:, :, None]
        b = self.vertical * np.array(ys)[:, :, None]
        directions = a + b + (self.bottomleft - self.origin)
        origin = self.origin[None, :].repeat(reso[0] * reso[1], axis=0).reshape(reso + (3,))

        directions /= norm(directions[:, :, None])

        assert origin.shape == directions.shape == (reso + (3,))
        return origin, directions


def color(points, directions, dist_func):
    img = np.zeros(points.shape)

    # running = np.ones(points.shape)

    dists = dist_func(points)
    print(points.shape, dists.shape, directions.shape)
    while np.any((0.01 < dists) & (dists < 12)):
        points += dists[:, :, None] * directions
        dists = dist_func(points)

    img[dists < 0.01] = (255, 169, 0)
    return img


def dst_function(p):
    return np.minimum(
        np.minimum(
            dst_sphere(p, vec(-1, 0, -2), 0.5),
            dst_sphere(p, vec(0, 0, -2), 0.5)
        ),
        dst_sphere(p, vec(1, 0, -2), 0.5),
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
