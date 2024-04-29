from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():
    checker = Checker(resolution=100, tile_size=10)
    checker.draw()
    checker.show()

    circle = Circle(resolution=100, radius=20, position=(50, 50))
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=100)
    spectrum.draw()
    spectrum.show()

    generator = ImageGenerator(
        file_path='C:\\Users\\Amin\\PycharmProjects\\deepLearning\\exercise_data',
        label_path='C:\\Users\\Amin\\PycharmProjects\\deepLearning\\labels.json',
        batch_size=50,
        image_size=[32, 32, 3],
        rotation=True,
        mirroring=True,
        shuffle_data=True
    )
    generator.show()


if __name__ == '__main__':
    main()
