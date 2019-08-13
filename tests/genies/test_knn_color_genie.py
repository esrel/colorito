from colorito import HTML_PALETTE
from colorito.genies.knn import KNNColorGenie
from colorito.palette import SmartPalette

"""

Tests for the KNNColorGenie functions

"""


def create_smart_palette():
    colorgenie = KNNColorGenie()
    my_palette = SmartPalette(colorgenie)
    my_palette.update_palette(HTML_PALETTE)
    my_palette.prepare()
    return my_palette


def test_get_color_rgb():
    pass


def test_get_main_tint():
    palette = create_smart_palette()
    tests = {
        'salmon': 'pink',
        'mustard yellow': 'yellow',
        'olive': 'green',
        'chocolate': 'brown',
        'smoke': 'white'
    }

    for color, main_tint in tests.items():
        assert palette.get_main_tint(color) == main_tint
