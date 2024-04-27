import numpy as np
import matplotlib.pyplot as plt


class Checker:

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

        if resolution % (2 * tile_size) != 0:
            raise f'Resolution that are evenly dividable by 2 tile size are accepted.'

    def draw(self):
        num_tiles = self.resolution // self.tile_size
        pattern_tile = np.indices((num_tiles, num_tiles)).sum(axis=0) % 2
        self.output = np.kron(pattern_tile, np.ones((self.tile_size, self.tile_size)))

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title('Checkerboard')
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)

        x_center, y_center = self.position
        dis_from_center = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

        self.output = (dis_from_center <= self.radius).astype(int)

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title('Circle')
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        spectrum = np.linspace(0, 1, self.resolution)

        red = np.tile(spectrum, (self.resolution, 1))
        green = np.tile(spectrum, (self.resolution, 1)).T
        blue = 1 - red

        self.output = np.dstack((red, green, blue))

        return  self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.title('RGB')
        plt.show()




