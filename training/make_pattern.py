"""Builds the tiled paw pattern that sits behind the conversation.

WhatsApp's doodle background works because it is almost invisible: one flat grey, no
colour, and low enough contrast that text over it reads normally. This does the same to the
paw mark. The output is a seamless tile written twice, once dark-on-light and once
light-on-dark, so each theme gets a pattern that disappears into its own background.

    python training/make_pattern.py <source.png>
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "static", "cats")

TILE = 360          # the repeat, in CSS pixels
PAW = 92            # how big one paw sits inside it
ROTATIONS = (-18, 12, -6, 22)


def flatten(source):
    """Greyscale silhouette on transparency, with the colour thrown away."""
    image = Image.open(source).convert("RGBA")
    image = image.crop(image.getbbox())
    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            # anything close to white is page, not drawing
            ink = 255 - min(r, g, b)
            pixels[x, y] = (0, 0, 0, min(a, ink))
    return image


def build(paw, colour, opacity, name):
    """A 2x2 offset grid of rotated paws, which tiles without a visible seam."""
    tile = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    spots = [(TILE * 0.18, TILE * 0.20), (TILE * 0.68, TILE * 0.34),
             (TILE * 0.38, TILE * 0.70), (TILE * 0.86, TILE * 0.82)]

    for (x, y), angle in zip(spots, ROTATIONS):
        stamp = paw.copy()
        stamp.thumbnail((PAW, PAW), Image.LANCZOS)
        stamp = stamp.rotate(angle, expand=True, resample=Image.BICUBIC)

        tinted = Image.new("RGBA", stamp.size, colour + (0,))
        alpha = stamp.getchannel("A").point(lambda v: int(v * opacity))
        tinted.putalpha(alpha)

        left, top = int(x - stamp.width / 2), int(y - stamp.height / 2)
        # paste every wrap-around copy so a paw crossing an edge continues on the other side
        for dx in (-TILE, 0, TILE):
            for dy in (-TILE, 0, TILE):
                tile.alpha_composite(tinted, (left + dx, top + dy))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    tile.save(path, optimize=True)
    print(f"{name}  {tile.size}  {os.path.getsize(path) / 1024:.1f}KB")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    paw = flatten(sys.argv[1])
    build(paw, (0, 0, 0), 0.055, "paw-light.png")     # dark ink on a light page
    build(paw, (255, 255, 255), 0.075, "paw-dark.png")  # light ink on a dark one


if __name__ == "__main__":
    main()
