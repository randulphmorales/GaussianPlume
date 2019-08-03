#!/usr/bin/env python
# python

import numpy as np


class receptorGrid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(y, z, x)


class pointSource:
    def __init__(self, x, y, rate, H):
        self.x = x
        self.y = y
        self.rate = rate
        self.H = H
        self.sourceType = "point"


class areaSource:
    def __init__(self, x0, dx, nx, y0, dy, ny, z, rate, H):
        self.x = np.linspace(x0, nx * dx, nx + 1)
        self.y = np.linspace(y0, ny * dy, ny + 1)
        self.z = z
        self.sourceType = 'area'
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(self.y, z, self.x)
        self.H = H
        self.rate = rate
        self.dx = dx
        self.dy = dy


##### COMPUTE SIGMAS #####

def coriolis(latitude):
    fcor = 2 * 7.29 * 0.00001 * np.sin(np.deg2rad(latitude))

    return fcor


def bl_height(u_star, L, fcor):
    if L < 0:
        first_term = 0.4 * np.sqrt(u_star * 1000 / fcor)
        blh = np.min([first_term, 800]) + 300 * ((2.72) ** (L * 0.01))
    elif L >= 0:
        first_term = 0.4 * np.sqrt(u_star * 1000 / fcor)
        blh = np.min([first_term, 800])

    return blh


class sigma_y:
    def __init__(self, z, u_star, L, blh, fcor):
        self.z = z
        self.u_star = u_star
        self.L = L
        self.blh = blh
        self.fcor = fcor

        def sigma_v(z, u_star, blh, fcor):
            if L < 0:
                sv = u_star * (12 - (0.5 * blh / L)) ** (1 / 3)
            elif L == 0:
                sv = 1.3 * u_star * np.exp(-2 * (fcor * z / u_star))
            elif L > 0:
                sv = np.max([(1.3 * u_star * (1 - (z / blh))), 0.2])

            return sv

        def timescale_y(z, sv, u_star, blh, fcor):
            if L < 0:
                tsy = 0.15 * (blh / sv)
            elif L == 0:
                tsy = (0.5 * (z / sv)) / (1 + 15 * (fcor * z / u_star))
            elif L > 0:
                tsy = 0.07 * (blh / sv) * np.sqrt(z / blh)

            return tsy

        def fy(t, tsy):
            return (1 + 0.5 * (t / tsy)) ** (-0.5)

        def sy(sv, t, fy):
            return sv * t * fy

        self.sigma_v = sigma_v
        self.timescale_y = timescale_y
        self.fy = fy
        self.sy = sy


class sigma_z:
    def __init__(self, z, u_star, L, blh, fcor):
        self.z = z
        self.u_star = u_star
        self.L = L
        self.blh = blh
        self.fcor = fcor

        # compute for sigma_w, timescalez, fz
        def sigma_w(z, u_star, blh):
            if L < 0:
                sw = 0.6 * 0.2
            elif L >= 0:
                sw = 1.3 * u_star * (1 - (z / blh)**0.75)
            return sw

        def timescale_z(z, sw, u_star, blh, fcor):
            if L < 0:
                tsz = 0.15 * blh * (1 - np.exp(-5 * (z / blh))) / (sw)
            elif L == 0:
                tsz = (0.5 * (z / sw)) / (1 + 15 * (fcor * z / u_star))
            elif L > 0:
                tsz = 0.10 * (blh / sw) * (z / blh)**0.8

            return tsz

        def fz(z, t, tsz):
            if L < 0:
                f_z = (1 + 0.5 * (t / tsz))**(-1 / 2)
            elif L >= 0:
                # if z < 50:
                f_z = (1 + 0.9 * (t / 50))**-1
                # elif z >= 50:
                #     f_z = (1 + 0.945 * ((0.1 * t) ** 0.806)) ** -1
            return f_z

        # compute sigma_z

        def sz(sw, t, f_z):
            return sw * t * f_z

        self.sigma_w = sigma_w
        self.timescale_z = timescale_z
        self.fz = fz
        self.sz = sz


class gaussianPlume:
    def __init__(self, source, grid, sigma_y, sigma_z, blh, fcor, U):
        self.source = source
        self.grid = grid
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.blh = blh
        self.fcor = fcor
        self.U = U

    def calculateConcentration(self):
        # calculate sigma_y
        sv = self.sigma_y.sigma_v(
            self.grid.zMesh, self.sigma_y.u_star, self.blh, self.fcor)
        tsy = self.sigma_y.timescale_y(
            self.grid.zMesh, sv, self.sigma_y.u_star, self.blh, self.fcor)
        fy = self.sigma_y.fy((self.grid.xMesh - self.source.x) / self.U, tsy)

        sy = self.sigma_y.sy(
            sv, (self.grid.xMesh - self.source.x) / self.U, fy)

        # calculate sigma_z
        sw = self.sigma_z.sigma_w(
            self.grid.zMesh, self.sigma_z.u_star, self.blh)
        tsz = self.sigma_z.timescale_z(
            self.grid.zMesh, sw, self.sigma_z.u_star, self.blh, self.fcor)
        f_z = self.sigma_z.fz(
            self.grid.zMesh, (self.grid.xMesh - self.source.x), tsz)
        sz = self.sigma_z.sz(
            sw, (self.grid.xMesh - self.source.x) / self.U, f_z)

        conc = np.zeros_like(self.grid.xMesh, dtype=float)

        # if self.source.sourceType == "area":
        #     for x in self.source.x:
        #         for y in self.source.y:
        #             a = self.source.rate * self.source.dx * self.source.dy / \
        #                 (2 * np.pi * self.U * sigma_y * sigma_z)
        #             b = np.exp(-(self.grid.yMesh - y)**2 / (2 * sigma_y))
        #             c = np.exp(-(self.grid.zMesh - self.source.H) ** 2 / (2 * sigma_z**2)) + \
        #                 np.exp(-(self.grid.zMesh + self.source.H)
        #                        ** 2 / (2 * sigma_z**2))
        #             conc += a * b * c

        if self.source.sourceType == "point":
            y = self.source.y
            a = self.source.rate / (2 * np.pi * self.U * sy * sz)
            b = np.exp(-(self.grid.yMesh - y)**2 / (2 * sy ** 2))
            c = np.exp(-(self.grid.zMesh - self.source.H)**2 / (2 * sz**2)) + \
                np.exp(-(self.grid.zMesh + self.source.H) ** 2 / (2 * sz**2))

            conc += a * b * c

        return conc
