import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_image(image, target_width, target_height):
    width, height = image.size
    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = width / height

    if current_aspect_ratio > target_aspect_ratio:
        # Image is too wide, crop the sides
        new_width = int(height * target_aspect_ratio)
        left = (width - new_width) / 2
        top = 0
        right = (width + new_width) / 2
        bottom = height
    else:
        # Image is too tall, crop the top and bottom
        new_height = int(width / target_aspect_ratio)
        left = 0
        top = (height - new_height) / 2
        right = width
        bottom = (height + new_height) / 2

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def resize_image(image, target_width, target_height):
    cropped_image = crop_image(image, target_width, target_height)
    resized_image = cropped_image.resize((target_width, target_height))
    return resized_image

def is_correct_image_size(image, target_width, target_height):
    width, height = image.size
    return width == target_width and height == target_height

class ResizeImageTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        logging.debug(f'Applying resize transform to image of size {image.size}')
        if not is_correct_image_size(image, self.width, self.height):
            image = resize_image(image, self.width, self.height)
        return image