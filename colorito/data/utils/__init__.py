from colorito.utils.logs import setup_logger

from torch.nn.utils.rnn import pad_sequence

import re
import spacy
import numpy as np
import unicodedata

logger = setup_logger('data:utils')
spacy_ = spacy.load('en')


def clean(strings):
    logger.info(' cleaning strings...')

    def _apply(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', r' ', s)
        s = re.sub(
            r'[^\s\d]*[\d]+[^\s\d]*',  # removing any token with digits in it.
            r' ', s)
        s = re.sub(r'[\s]+', r' ', s)

        return s.strip()

    separat = '   '
    strings = separat.join([
        _apply(string) for
        string in strings
    ])

    cleaned = ''.join([
        token.text_with_ws for
        token in spacy_(strings)
    ])

    # removing accents:

    for string in cleaned.split(separat):
        string = unicodedata.normalize('NFD', string)
        string = string.encode('ascii', 'ignore').decode(
                                                 'utf-8')
        yield string


def encode(strings, vectorizer):
    strings = np.array([
        strings
    ]).reshape(-1).tolist( )
    strings = clean(strings)
    tensors = vectorizer.torch_transform(strings)
    tensors = pad_sequence(
        list(tensors),
        batch_first=1.
    )

    return tensors
