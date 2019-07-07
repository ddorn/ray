from colorsys import hsv_to_rgb
from time import time

import numpy as np
import pygame


class Camera:
    def __init__(self, size):
        self.center = 0j
        self.height = 2
        self.size = size

    def genPoints(self):
        x = np.linspace(self.center.real - self.width / 2, self.center.real + self.width / 2, self.size[0])
        y = np.linspace(self.center.imag + self.height / 2, self.center.imag - self.height / 2, self.size[1])

        return x[:, None] + 1j * y

    def complex_at(self, pixel):
        cpixel =   (pixel[0] / self.size[0] - 0.5) * self.width \
                 + (pixel[1] / self.size[1] * 1j - 0.5j) * -self.height + self.center
        return cpixel

    def pixel_at(self, complex):
        complex -= self.center
        pixel = (
            (complex.real / self.width + 0.5) * self.size[0],
            (complex.imag / -self.height + 0.5) * self.size[1]
        )
        return round(pixel[0]), round(pixel[1])

    def test_conversion(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                print(x, y, self.complex_at((x, y)), self.pixel_at(self.complex_at((x, y))))
                assert self.pixel_at(self.complex_at((x, y))) == (x, y)

    @property
    def width(self):
        return self.height * self.size[0] / self.size[1]



def compute(points: np.ndarray, maxi=50):
    zs = points
    ps = points
    steps: np.ndarray = np.zeros(points.size)
    idx = np.arange(points.size)
    for i in range(maxi):
        print(zs.shape)
        zs = zs ** 2 + ps

        keep = np.abs(zs) < 2
        idx = idx[keep]
        zs = zs[keep]
        ps = ps[keep]

        steps[idx] += 1

    return steps / steps.max()


def compute_one(c: complex, maxi=50):
    z = c
    yield z
    for i in range(maxi):
        z = z*z + c
        yield z
        if abs(z) > 2:
            return


def colorize(steps):

    # img = np.zeros(dest.get_size() + (3,))

    color = hsv_to_rgb(time() / 4 % 1, 1, 1)
    img = steps[..., None] * color
    img[steps == 1] = (0, 0, 0)
    # img[..., 0] = (np.sin(time()) + 1) + steps
    # img[..., 1] = (np.sin(2*time()) + 1) + steps
    # img[..., 2] = np.sin(time() + 2) * steps


    # img /= img.max()

    return (img * 255).astype(int)

def draw_image(camera, dest, steps=None):

    if steps is None:
        shape = dest.get_size()
        points = camera.genPoints()
        assert points.shape == shape, (points.shape, shape)

        steps = compute(points.flatten()).reshape(shape)


    pygame.surfarray.pixels3d(dest)[:] = colorize(steps)
    return steps


def main():
    SIZE = 1920, 1080
    FACTOR = 5

    SMALL_STEP = 0.1

    screen: pygame.SurfaceType = pygame.display.set_mode(SIZE)
    surf = pygame.Surface((SIZE[0] // FACTOR, SIZE[1] // FACTOR))
    camera = Camera(surf.get_size())
    steps = None

    draw_image(camera, surf, steps)
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
                    camera.center += camera.width / 15
                elif e.key == pygame.K_a:
                    camera.center -= camera.width / 15
                elif e.key == pygame.K_w:
                    camera.center += camera.height / 15 * 1j
                elif e.key == pygame.K_s:
                    camera.center -= camera.height / 15 * 1j
                elif e.key == pygame.K_r:
                    camera.height /= 1.5
                elif e.key == pygame.K_f:
                    camera.height *= 1.5
                elif e.key == pygame.K_SPACE:
                    camera = Camera(surf.get_size())
                elif e.unicode.isdigit():
                    FACTOR = int(e.unicode) + 1
                    surf = pygame.Surface((SIZE[0] // FACTOR, SIZE[1] // FACTOR))
                    camera.size = surf.get_size()
                elif e.key == pygame.K_p:
                    pygame.image.save(screen, 'screenshot.png')
                    continue
                steps = None

        steps = draw_image(camera, surf, steps)

        mouse = pygame.mouse.get_pos()
        mouse = mouse[0] / FACTOR, mouse[1] / FACTOR
        c = camera.complex_at(mouse)
        pygame.draw.aalines(surf, (255, 169, 0), False, [camera.pixel_at(p) for p in compute_one(c)])

        pygame.transform.scale(surf, SIZE, screen)

        pygame.display.update()


if __name__ == '__main__':
    main()
