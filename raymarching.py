#!/usr/bin/env python3
import profile

import pygame
import numpy as np


from camera import Camera
from distances import *
from objects import Sphere, Cube

ALL_NORMALS = False

pygame.init()
# noinspection PyArgumentList
pygame.key.set_repeat(100, 30)

def vec(x, y, z):
    return np.array((x, y, z), dtype=np.float)


def march(points, directions, dist_func):
    assert points.shape == directions.shape

    EPS = 0.00001
    HIT = 0.001
    shape = points.shape

    running = np.ones(shape[:-1], dtype=np.bool)

    dists = dist_func(points)
    steps = np.zeros(shape[:-1], dtype=np.float)

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
        # normals = vec(0, 1 - 0.647, 1)[None, :].repeat(shape[0] * shape[1], axis=0).reshape(shape)
        normals = np.zeros(shape)
        normals[hits] = dists[hits, None]
        normals[hits, 0] = dist_func(points[hits] - vec(EPS, 0, 0)) - dist_func(points[hits] + vec(EPS, 0, 0))
        normals[hits, 1] = dist_func(points[hits] - vec(0, EPS, 0)) - dist_func(points[hits] + vec(0, EPS, 0))
        normals[hits, 2] = dist_func(points[hits] - vec(0, 0, EPS)) - dist_func(points[hits] + vec(0, 0, EPS))
        normals[hits] /= norm(normals[hits])[:, None]

        # steps /= steps.max()
        # steps **= 3
        # steps[steps < 0.3] = 0
        # img = (255 * (0.5 + 0.5 * normals)).astype(int)

    else:
        normals = np.zeros(shape)
        normals[:] = dists[:, :, None]
        normals[:, :, 0] -= dist_func(points + vec(EPS, 0, 0))
        normals[:, :, 1] -= dist_func(points + vec(0, EPS, 0))
        normals[:, :, 2] -= dist_func(points + vec(0, 0, EPS))
        normals /= norm(normals)[:, :, None]

        # img = (255 * (0.5 + 0.5 * normals)).astype(int)

    return points, normals, steps


def dst_function(p):
    disp = lambda p: 0.25 * np.sin(5 * p[..., 0]) * np.sin(p[...,1] * 5) * np.sin(p[...,2] * 5) #+ 0.1 * np.sin(p[...,2])

    scene = Sphere((0, 0, -4), 1) + Cube((0, 0, -4), 0.8) - 0.2
    # scene = Cube((0, 0, -4), 0.8) - 0.2 - Sphere((0, 0, -4), 1.2)
    return scene.dst(p)

    return dst_sphere(p, vec(0, 0, -2), 0.5)
    return substraction(
        intersect(
            dst_sphere(p, vec(0, 0, -4), 1),
            dst_cube(p, vec(0, 0, -4), 0.8)
        ),
        dst_sphere(p, vec(0, 0, -4), 0.95)
    )


def draw_image(camera, scene, dst):
    point, direction = camera.genRays(dst.get_size())
    points, normals, steps = march(point, direction, scene)
    hits = norm(normals) > 0

    # find intensity of light
    ambiant = 0.1
    light_pos = vec(1, 1, 1) * 10

    light_dir = unit(points - light_pos)

    diffuse = np.maximum(np.einsum('ijk,ijk->ij', normals, light_dir), 0)

    color = (diffuse)[..., None] * vec(1, 0, 0)
    color[hits] += vec(0.1, 0, 0)
    img = (255 * color).astype(int)
    img = np.minimum(img, 255)

    pygame.surfarray.pixels3d(dst)[:] = img


def main():
    SIZE = 1920, 1080
    FACTOR = 5

    SMALL_STEP = 0.1

    screen: pygame.SurfaceType = pygame.display.set_mode(SIZE)
    surf = pygame.Surface((SIZE[0] // FACTOR, SIZE[1] // FACTOR))
    camera = Camera(50, vec(0, 0, 0), vec(0, 0, -4), vec(0, 1, 0), SIZE[1]/SIZE[0])

    draw_image(camera, dst_function, surf)
    pygame.transform.scale(surf, SIZE, screen)

    stop = False
    while not stop:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                stop = True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return
                elif e.key == pygame.K_d:
                    camera.move((SMALL_STEP, 0, 0))
                elif e.key == pygame.K_a:
                    camera.move((-SMALL_STEP, 0, 0))
                elif e.key == pygame.K_w:
                    camera.move((0, SMALL_STEP, 0))
                elif e.key == pygame.K_s:
                    camera.move((0, -SMALL_STEP, 0))
                elif e.key == pygame.K_f:
                    camera.move((0, 0, SMALL_STEP))
                elif e.key == pygame.K_r:
                    camera.move((0, 0, -SMALL_STEP))
                elif e.key == pygame.K_SPACE:
                    camera = Camera(50, vec(0, 0, 0), vec(0, 0, -2), vec(0, 1, 0), 0.5)
                elif e.unicode.isdigit():
                    FACTOR = int(e.unicode) + 1
                    surf = pygame.Surface((SIZE[0] // FACTOR, SIZE[1] // FACTOR))
                elif e.key == pygame.K_p:
                    pygame.image.save(screen, 'screenshot.png')
                    continue

                draw_image(camera, dst_function, surf)
                pygame.transform.scale(surf, SIZE, screen)

        pygame.display.update()


if __name__ == '__main__':
    main()
