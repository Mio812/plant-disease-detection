"""Dataset-bias diagnostics for PlantVillage.

Two probes quantify how much of the headline accuracy comes from capture bias
rather than leaf pathology:

* ``border_features`` — samples a handful of background pixels from the image
  border. A classifier trained on these alone should be near chance (1/38) if
  the background carries no label information.
* ``segmented_path``  — maps a colour image to its background-removed twin in
  the ``segmented`` variant, so trained models can be re-scored without the
  background.
"""


BORDER_POSITIONS = [(0, 0), (16, 0), (31, 0), (0, 16), (31, 16), (0, 31), (16, 31), (31, 31)]


def border_features(path, size=32, positions=BORDER_POSITIONS):
    """Return the RGB values of ``positions`` on a ``size``x``size`` thumbnail."""
    from PIL import Image

    image = Image.open(path)
    image.draft("RGB", (size, size))
    pixels = image.convert("RGB").resize((size, size)).load()
    return [channel for xy in positions for channel in pixels[xy]]
