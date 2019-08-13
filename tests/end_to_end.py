from colorito.palette import SmartPalette
from colorito.genies.knn import KNNColorGenie
from colorito import HTML_PALETTE


def print_color_rgb(color, rgb):
    print(
        '{} rgb is: {}'
        ''.format(
            color,
            rgb
        )
    )


def load_palette(from_, genie_cls):
    return SmartPalette.load(
        genie_cls, from_
    )


def persist_palette(palette, to, name='palette'):
    palette.persist(to, name)


if __name__ == '__main__':

    g = KNNColorGenie()
    p = SmartPalette(g)
    p.update_palette(HTML_PALETTE)

    print_color_rgb(
        'blue',
        p.get_color_rgb(
                 'blue')
    )

    p.prepare()

    print_color_rgb(
        'river blue',
        p.get_color_rgb(
          ' river blue')
    )

    print(p.get_shades_of(
           'grass green'))

    # persist_palette(p, '.')
