from colorito.exceptions import InvalidColorFormatException
from colorito.utils.logging import logger
from colorito.utils.convert import ColorConverter


class PaletteReader(object):

    def __init__(self):
        pass

    @staticmethod
    def read_palette(palette):
        # TODO: implement logic to infer separator plus
        #       file format automatically when possible
        return PaletteReader.read_palette_from_csv(palette)

    @staticmethod
    def read_palette_from_csv(palette_f, sep=","):
        """
        Returns a text-to-rgb dictionary for the palette.
        :return: Dict<String, Tuple<Int, Int, Int>>.
        """
        palette = {}
        with open(palette_f, 'r') as f:
            for line in filter(
                    lambda x: x,
                    f.read().split('\n')
            ):
                color, code = line.lower().strip().split(sep)
                try:
                    palette[color] = ColorConverter.color_code_to_rgb(code)
                except InvalidColorFormatException as e:
                    logger.warning(
                        f" {e}; color '{color}' will be skipped."
                    )

        return palette
