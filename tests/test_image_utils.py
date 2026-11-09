from io import BytesIO
from PIL import Image
import pytest
from core.image_utils import load_rgb_image, image_dimensions

def test_load_rgb_converts_mode():
    img=Image.new("L",(20,10),128); out=load_rgb_image(img); assert out.mode=="RGB" and out.size==(20,10)
def test_dimensions(): assert image_dimensions(Image.new("RGB",(15,12)))==(15,12)
def test_invalid_image():
    with pytest.raises(ValueError): load_rgb_image(BytesIO(b"not an image"))
