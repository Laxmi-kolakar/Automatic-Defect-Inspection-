from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def load_rgb_image(source):
    """Load an uploaded file/path/PIL image and return a safe RGB PIL image."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            image = Image.open(source)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file is not a readable image. Upload a JPG, JPEG, or PNG file.") from exc

def image_to_array(image, size=128):
    image = load_rgb_image(image).resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0

def image_dimensions(image):
    image = load_rgb_image(image)
    return image.width, image.height
