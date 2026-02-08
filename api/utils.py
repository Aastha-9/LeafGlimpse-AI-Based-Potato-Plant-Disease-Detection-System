import numpy as np
from PIL import Image
import io

def read_file_as_image(data):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
