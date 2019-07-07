from math import pi, tan

import numpy as np

from distances import norm, unit


class Camera:
    bottomleft: np.array
    horizontal: np.array
    vertical: np.array
    origin: np.array

    def __init__(self, vfov, position, lookat, up, vh_ratio):
        """
        A camera to generate rays
        :param vfov: Vertical field of view in degrees
        :param position: origin of rays
        :param direction: direction of the center of the screen
        :param up: up direction
        :param vh_ratio: vertical/horizontal ratio
        """

        self.vfov = vfov
        self.position = position
        self.lookat = lookat
        self.up = up
        self.vh_ratio = vh_ratio

        self.update()


    def update(self):
        theta = self.vfov * pi / 180

        self.z = z = (self.position - self.lookat) / norm(self.position - self.lookat)
        self.x = x = unit(np.cross(self.up, z))
        self.y = y = np.cross(z, x)

        half_height = tan(theta / 2)
        half_width = half_height / self.vh_ratio

        self.bottomleft = self.position - half_width * x - half_height * y - z
        self.horizontal = 2 * half_width * x
        self.vertical = 2 * half_height * y
        self.origin = self.position


    def move(self, translation):
        self.position += self.x * translation[0] + self.y * translation[1] + self.z * translation[2]
        self.lookat += self.x * translation[0] + self.y * translation[1] + self.z * translation[2]
        self.update()

    def __str__(self):
        return f'<Camera({self.origin}, BL={self.bottomleft}, H={self.horizontal}, V={self.vertical}'

    def genRays(self, reso):
        x = np.linspace(0, 1, reso[0])
        y = np.linspace(1, 0, reso[1])
        ys, xs = np.meshgrid(y, x)
        # ys += np.random.random(ys.shape) / reso[0]
        # xs += np.random.random(xs.shape) / reso[0]
        a = self.horizontal * np.array(xs)[:, :, None]
        b = self.vertical * np.array(ys)[:, :, None]
        directions = a + b + (self.bottomleft - self.origin)
        origin = self.origin[None, :].repeat(reso[0] * reso[1], axis=0).reshape(reso + (3,))

        directions /= norm(directions[:, :, None])

        assert origin.shape == directions.shape == (reso + (3,))
        return origin, directions
