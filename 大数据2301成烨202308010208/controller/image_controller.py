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