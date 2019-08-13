from colorito import HTML_PALETTE, DEFAULT_PALETTE
from colorito.genies.knn import KNNColorGenie
from colorito.palette import SmartPalette

"""

Tests for the SmartPalette functions

"""

RED, ORANGE, ORANGE_RED = 'red', 'orange', 'orange red'

CLUSTERS = {
    'red': {
        RED,
        'indian red',
        'fire brick',
        'dark red',
        'crimson'
    },
    'orange': {
        'orange red'
    }
}


def create_smart_palette():
    colorgenie = KNNColorGenie()
    my_palette = SmartPalette(colorgenie)
    my_palette.update_palette(HTML_PALETTE)
    return my_palette


def test_color_cluster_creation():
    # create the cluster of red colors.
    my_palette = create_smart_palette()
    my_palette.create_color_cluster(
        CLUSTERS[RED]
    )
    # assert the cluster is correctly
    # returned by the palette when as-
    # king for shades of red.
    assert my_palette.get_shades_of(RED) == CLUSTERS[RED]
    red_cluster_id = my_palette.get_color_cluster_of(RED)
    my_palette.mov_color_into_cluster(
        'tomato', red_cluster_id
    )
    # assert that the new color has
    # been added to the cluster.
    assert 'tomato' in my_palette.get_shades_of(RED)


def test_color_cluster_update():
    # create red cluster.
    my_palette = create_smart_palette()
    my_palette.create_color_cluster(
        CLUSTERS[RED]
    )
    # create pink cluster.
    my_palette = create_smart_palette()
    my_palette.create_color_cluster(
        CLUSTERS[ORANGE]
    )
    # assert orange red is only
    # in orange cluster and not
    # in red.
    assert (
        ORANGE_RED in my_palette.get_shades_of(ORANGE) and
        ORANGE_RED not in my_palette.get_shades_of(RED)
    )
    # move orange to red cluster
    # and assert if the move was
    # a success.
    my_palette.mov_color_into_cluster(
        ORANGE_RED,
        my_palette.get_color_cluster_of(RED)
    )
    assert (
        ORANGE_RED in my_palette.get_shades_of(RED) and
        ORANGE_RED not in my_palette.get_shades_of(ORANGE)
    )
    # assert that red cluster's
    # the only one that is left.
    assert (
        len(my_palette.color_clusters) == 1 and
        my_palette.get_color_cluster_of(RED) > 0
    )
