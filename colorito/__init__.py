import os

RESOURCE_FOLDER = os.path.split(
    os.path.abspath(__file__))[0]

DEFAULT_PALETTE = os.path.join(
    RESOURCE_FOLDER,
    'resources/genies.csv'
)
HTML_PALETTE = os.path.join(
    RESOURCE_FOLDER,
    'resources/html_colors.csv'
)