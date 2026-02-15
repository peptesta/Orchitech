from PIL import Image
import torchvision.transforms as T

def rotate_image(image):
    return image.rotate(90, expand=True)

def is_landscape(image):
    width, height = image.size
    return width > height

def horizontal_flip(img):
    return T.functional.hflip(img)

def vertical_flip(img):
    return T.functional.vflip(img)

def horizontal_and_vertical_flip(img):
    return T.functional.hflip(T.functional.vflip(img))

def identity(img):
    return img

class RotateIfLandscapeTransform:
    def __call__(self, image):
        if is_landscape(image):
            image = rotate_image(image)
        return image
