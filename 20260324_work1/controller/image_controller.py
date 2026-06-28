from service.image_service import ImageService


class ImageController:

    def __init__(self):

        self.service = ImageService()

    def open_image(self, path):

        return self.service.load_image(path)

    def reset(self):

        return self.service.reset()

    def gray(self):

        return self.service.gray()

    def binary(self):

        return self.service.binary()

    def inverse(self):

        return self.service.inverse()

    def gamma(self):

        return self.service.gamma()

    def log_transform(self):

        return self.service.log_transform()

    def exp_transform(self):

        return self.service.exp_transform()

    def resize_half(self):

        return self.service.resize_half()

    def get_info(self):

        return self.service.get_info()

    def glass(self):

        return self.service.glass()

    def relief(self):

        return self.service.relief()

    def oil(self):

        return self.service.oil()

    def mask(self):

        return self.service.mask()

    def sketch(self):

        return self.service.sketch()

    def old(self):

        return self.service.old()

    def lighting(self):

        return self.service.lighting()

    def cartoonize(self):

        return self.service.cartoonize()

    def hist_equalize(self):

        return self.service.hist_equalize()

    def zoom(self, factor):
        return self.service.zoom(factor)

    def rotate(self, angle):
        return self.service.rotate(angle)

    def translate(self, tx, ty):
        return self.service.translate(tx, ty)

    def flip_h(self):
        return self.service.flip_h()

    def flip_v(self):
        return self.service.flip_v()

    def shear(self, factor):
        return self.service.shear(factor)

    def perspective(self):
        return self.service.perspective()

    def wave(self):
        return self.service.wave()

    def stitch(self, img2):
        return self.service.stitch(img2)