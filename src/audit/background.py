"""Background-bias probe: features taken only from the image border.

A classifier trained on these should score near chance (1/38) unless the
background carries label information.
"""


BORDER_POSITIONS = [(0, 0), (16, 0), (31, 0), (0, 16), (31, 16), (0, 31), (16, 31), (31, 31)]


def border_features(path, size=32, positions=BORDER_POSITIONS):
    """RGB values at ``positions`` on a size x size thumbnail."""
    from PIL import Image

    image = Image.open(path)
    image.draft("RGB", (size, size))
    pixels = image.convert("RGB").resize((size, size)).load()
    return [channel for xy in positions for channel in pixels[xy]]
