from colorito.exceptions import InvalidColorFormatException
import re


class ColorConverter(object):

    # patterns for hexadecimal and rgb color codes

    HEX_CC = re.compile(r"^#?[0-9abcdef]{6}$")
    RGB_CC = re.compile(
        r"^\(?[0-2]?[0-5][0-5][,;\s][0-2]?[0"
        r"-5][0-5][,;\s][0-2]?[0-5][0-5]\)?$"
    )

    def __init__(self):
        pass

    @staticmethod
    def color_code_to_rgb(color_code):
        color_code  = str(color_code)
        if re.match(
                ColorConverter.HEX_CC,
                color_code
        ):
            return tuple(int(
                color_code.lstrip(
                    '#')[i:i + 2],
                16

            ) for i in (0, 2, 4))

        elif re.match(
                ColorConverter.RGB_CC,
                color_code
        ):
            rgb = []
            for primary_color in re.finditer(
                    r"\d{1-3}[^\d]",
                    color_code
            ):
                rgb.append(
                    primary_color.group(0)
                )

            rgb = (
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2])
            )

            ColorConverter.check_rgb_code(rgb)

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
        rgb_code = str(rgb_code)
        if not re.match(ColorConverter.RGB_CC, rgb_code):
            raise InvalidColorFormatException(
                f'invalid rgb code: {rgb_code}'
            )
