from PIL import Image, ImageDraw
import numpy as np

image = Image.new("RGBA", (800, 800), color=(159, 161, 110))
draw = ImageDraw.Draw(image)

vertices1 = [
    (np.random.randint(0, 800), np.random.randint(0, 800)),
    (np.random.randint(0, 800), np.random.randint(0, 800)),
    (np.random.randint(0, 800), np.random.randint(0, 800))
]

vertices2 = [
    (np.random.randint(0, 800), np.random.randint(0, 800)),
    (np.random.randint(0, 800), np.random.randint(0, 800)),
    (np.random.randint(0, 800), np.random.randint(0, 800))
]

draw.polygon(vertices1, fill=(255, 255, 255, 128))
draw.polygon(vertices2, fill=(255, 255, 255, 255))

image.save("yeet.png")