from colorito.utils.io import PaletteReader
from colorito import DEFAULT_PALETTE


class SmartPalette(object):

    def __init__(self, palette=DEFAULT_PALETTE):
        self.palette = None
        self.load_palette(
            palette
        )

    def load_palette(self, palette):
        self.palette = PaletteReader.read_palette(palette)

    def get_similar_colors(self, color, as_rgb=False):
        pass
