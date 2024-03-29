import os


COLORITO = os.path.split(os.path.abspath(__file__))[0]
COLORS = os.path.join(COLORITO, 'data/colors')
MODELS = os.path.join(COLORITO, 'nnet/models')
DEVICE = 'cpu'

DEFAULT_PALETTE = os.path.join(
     COLORS, 'html-colors.csv')

DEFAULT_NETWORK = os.path.join(
     MODELS, 'color-generators', 'base')

LITE_NETWORK = os.path.join(
     MODELS, 'color-generators', 'lite')
