from colorito.utils.logging import logger
from colorito.exceptions import InvalidColorFormatException
import os
import re


class PaletteReader(object):

    # patterns for hexadecimal and rgb color codes

    HEX_CC = re.compile(r"^#?[0-9abcdef]{6}$")
    RGB_CC = re.compile(
        r"^\(?{1-3}[,;\s]\d{1-3}[,;\s]\d{1-3}\)?$"
    )

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
                    palette[color] = PaletteReader.color_code_to_rgb(code)
                except InvalidColorFormatException as e:
                    logger.warning(
                        f" {e}; color '{color}' will be skipped."
                    )

        return palette

    @staticmethod
    def color_code_to_rgb(color_code):
        if re.match(
            PaletteReader.HEX_CC,
            color_code
        ):
            color_code = color_code.lstrip('#')
            return tuple(int(
                color_code[i:i+2], 16
            ) for i in (0, 2, 4))

        elif re.match(
                PaletteReader.RGB_CC,
                color_code
        ):
            rgb = []
            for primary_color in re.finditer(
                r"\d{1-3}[^\d]", color_code
            ):
                rgb.append(
                    primary_color.group(0)
                )

            rgb = (
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2])
            )

            PaletteReader.check_rgb_code(rgb)

            return rgb

        else:
            raise InvalidColorFormatException(
                "color code {} is of an unidentified format"
                " - only hex and rgb codes allowed".format(
                    color_code
                )
            )

    @staticmethod
    def check_rgb_code(rgb_code):
        for component in rgb_code:
            if 0 > component > 255:
                raise InvalidColorFormatException(
                    f'invalid rgb code: {rgb_code}'
                )
