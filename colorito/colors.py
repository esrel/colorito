from colormath.color_objects import XYZColor, sRGBColor, LabColor
from colormath.color_conversions import convert_color
import pylab as plt

import re


class Color(object):
    """
    Class modelling a color. Associates the
    color name to its hexadecimal value and
    provides method to build/convert the co-
    lor to the RGB or Lab space.
    """

    L_RANGE = [0, 100]
    A_RANGE = [-128, 127]
    B_RANGE = [-128, 127]
    RGB_RANGE = [0, 255]

    def __init__(self, name, hexc):
        """
        Build a color from a name
        and its hexadecimal value.

        :param name:
        :param hexc:
        """
        hexc = str(hexc).lower()
        if not re.match(r'#?[\w\d]{6}', hexc):
            raise ValueError(
                f'[!] {hexc} is not a valid hex'
                f'-value for a color {name}... '
            )

        self.name = name
        self.hexc = hexc

    def __str__(self):
        return f'{self.name} ({self.hexc})'

    def render(self):
        """
        Displays the color in a patch.

        :return:
        """
        data = [[self.rgb]]
        plt.figure(figsize=(2, 2))
        plt.imshow(data, interpolation='nearest')
        plt.show()

    @classmethod
    def from_rgb(cls, name, r, g, b, unscale=False):
        """
        Builds a color with the given name and the
        given r, g, b values. If the r, g, b coord-
        inates are defined in the [0, 1] range, th-
        en set `unscale` to True.

        :param name:
        :param r:
        :param g:
        :param b:
        :param unscale:
        :return:
        """
        r, g, b = float(r), float(g), float(b)
        if unscale:
            # bring rgb from [0, 1] to [0, 255]
            r = cls._unscale(r, *cls.RGB_RANGE)
            g = cls._unscale(g, *cls.RGB_RANGE)
            b = cls._unscale(b, *cls.RGB_RANGE)

        r, g, b = round(r), round(g), round(b)

        def _to_hexc(c):
            return f'0{hex(c)[2:]}'[-2:]

        r, g, b = (
            _to_hexc(r),
            _to_hexc(g),
            _to_hexc(b)
        )

        hexc = f'#{r}{g}{b}'

        return cls(name, hexc)

    @classmethod
    def from_lab(cls, name, l, a, b, unscale=False):
        """
        Builds a color with the given name and the
        given l, a, b values. If the l, a, b coord-
        inates are defined in the [0, 1] range, th-
        en set `unscale` to True.

        :param name:
        :param r:
        :param g:
        :param b:
        :param unscale:
        :return:
        """
        l, a, b = float(l), float(a), float(b)
        if unscale:
            # bring lab from [0, 1] to range
            l = cls._unscale(l, *cls.L_RANGE)
            a = cls._unscale(a, *cls.A_RANGE)
            b = cls._unscale(b, *cls.B_RANGE)

        lab = LabColor(l, a, b, illuminant='d65')
        rgb = convert_color(lab, sRGBColor)

        r, g, b = rgb.get_upscaled_value_tuple()

        return cls.from_rgb(name, r, g, b, unscale=False)

    @property
    def rgb(self):
        """
        Returns the RGB value of the
        colors as an integer triple.

        :return:
        """
        r, g, b = re.finditer(r'[\w\d]{2}', self.hexc)

        hex_to_decimal = self._hex_digit_to_decimal()

        r, g, b = r.group(0), g.group(0), b.group(0)

        r = hex_to_decimal[r[1]] + hex_to_decimal[r[0]] * 16
        g = hex_to_decimal[g[1]] + hex_to_decimal[g[0]] * 16
        b = hex_to_decimal[b[1]] + hex_to_decimal[b[0]] * 16

        return r, g, b

    @property
    def lab(self):
        """
        Returns the Lab value of the
        colors as a floating triple.

        :return:
        """
        r, g, b = self.rgb
        rgb = sRGBColor(
            rgb_r=self._rescale(r, *self.RGB_RANGE),
            rgb_g=self._rescale(g, *self.RGB_RANGE),
            rgb_b=self._rescale(b, *self.RGB_RANGE)
        )
        lab = convert_color(rgb, LabColor)

        return lab.lab_l, lab.lab_a, lab.lab_b

    @property
    def rescaled_rgb(self):
        """
        Returns the RGB value, but with
        each coordinate being mapped to
        [0, 1] range.

        :return:
        """
        r, g, b = self.rgb
        r = self._rescale(r, *self.RGB_RANGE)
        g = self._rescale(g, *self.RGB_RANGE)
        b = self._rescale(b, *self.RGB_RANGE)

        return r, g, b

    @property
    def rescaled_lab(self):
        """
        Returns the Lab value, but with
        each coordinate being mapped to
        [0, 1] range.

        :return:
        """
        l, a, b = self.lab
        l = self._rescale(l, self.L_RANGE[0], self.L_RANGE[1])
        a = self._rescale(a, self.A_RANGE[0], self.A_RANGE[1])
        b = self._rescale(b, self.B_RANGE[0], self.B_RANGE[1])

        return l, a, b

    @staticmethod
    def _rescale(v, v_min, v_max):
        return (v - v_min) / (v_max - v_min)

    @staticmethod
    def _unscale(v, v_min, v_max):
        return (v * (v_max - v_min)) + v_min

    @staticmethod
    def _hex_digit_to_decimal():
        return {
            v: i for i, v in enumerate(
                [f'{_}' for _ in range(10)] +
                ['a', 'b', 'c', 'd', 'e', 'f']
            )
        }
