from tqdm import tqdm

import string
import torch
import numpy as np


class NgramVectorizer(object):

    PAD = '#'
    UNK = '?'

    LEXICON = f'#{string.ascii_lowercase} '

    def __init__(self, order, bound=False):
        """
        Builds the vectorizer to transform
        text to bag-of-ngrams up to `order`
        ngrams.

        If bound=True, will use word boun-
        daries when extracting n-grams.

        :param order:
        :param bound:
        """
        self.word_bound = bound
        self.ngramorder = order
        self.inverted = []
        self.lexicons = [
            {char: i for i, char in enumerate(self.LEXICON)}
        ] + [
            {'#': 0} for _ in range( self.ngramorder  -  1 )
        ]

        if self.ngramorder < 1:
            raise ValueError(
                f'Got an invalid n-gram order'
                f' ({order}) - must be > 0...'
            )

    def _ngrams(self, text, order):
        """
        Extracts n-grams of the specified
        order from `text`. If `text` does
        not reach the length of the order
        instead of returning no n-grams,
        it pads the text.

        :param text:
        :param order:
        :return:
        """
        if len(text) < order:
            left = order - len(text)
            text += "".join([
                '#' for _ in range(left)
            ])

        at = 0
        while at + order <= len(text):
            yield text[at: at + order]
            at += 1

    def n_grams(self, phrase, order):
        """
        Extracts n-grams of the specified
        order from `phrase`. If the vecto-
        rizer was initialized with parame-
        ter bound=True, word boundaries w-
        ill be added.

        :param phrase:
        :param order:
        :return:
        """
        if not self.word_bound:
            words = [phrase]
        else:
            words = [
                f'<{word}>' for word
                in phrase.split()
            ]

        for word in words:
            for ngram in self._ngrams(word, order):
                yield ngram

    def fit(self, strings):
        """
        Builds lexicons for n-grams up to
        the order specified at initializa-
        tion.

        :param strings:
        :return:
        """
        for string in tqdm(strings):

            # build 1-gram, 2-gram, 3-gram...
            # up to self.order-gram lexicons.

            max_order = self.ngramorder + 1
            for order in range(1, max_order):
                lexicon = self.lexicons[order - 1]
                for ngram in self.n_grams(string,
                                          order):
                    if ngram not in lexicon:
                        lexicon[ngram] = len(lexicon)
        # add `UNK`
        for lexicon in self.lexicons:
            lexicon.update({'?': len(lexicon)})

        self.inverted = [{
            ix: ngram for ngram ,
            ix in lexicon.items()
        } for lexicon in
            self.lexicons]

    def transform(self, strings, **kwargs):
        """
        Transforms the passed list of strings into
        a list of matrices, where each matrix repr-
        esents one of the strings. Each matrix con-
        tains the indexes of all the n-grams in th-
        e string, up to `self.ngramorder` order.

        To clarify, given a string `hello`, we tra-
        nsform this into a matrix where:

            * 1st row: index of `h`, `he`, `hel`, ...
            * 2nd row: index of `e`, `el`, `ell`, ...
            * ...

        So row zero contains the indexes of all n-
        grams from character zero of the string;
        row one the same but n-grams start from t-
        he first character, row two from the seco-
        nd, and so on.

        :param strings:
        :param max_order:
        :return:
        """

        max_order = kwargs.get('max_order', self.ngramorder)

        if not 0 < max_order <= self.ngramorder:
            raise ValueError(
                f'Provided max_order is invalid. It has '
                f'to be between 1 and {self.ngramorder}.'
            )

        max_order += 1

        vectors = []
        for string in strings:
            # collect n-grams for each order
            # first row contains string's 1-
            # grams, second row 2-grams, th-
            # ird of 3-grams, and so on.
            n_grams_by_order = [
                list(self.n_grams(string, order))
                for order in range(1, max_order)
            ]
            # map each ngram to its lexicon's
            # index and transpose - now first
            # row contains indexes of all the
            # n-grams up to self.ngramorder f-
            # or the first string's character,
            # the second row it's the same bu-
            # t for the second character:
            #   e.g. `hello`
            #     1st row: index of `h`, `he`, `hel`, ...
            #     2nd row: index of `e`, `el`, `ell`, ...
            #   where the length of each row is max_order
            n_grams = np.array([
                np.array([
                    self._find(ngram, i + 1) for ngram in ngrams
                ])  for i, ngrams in enumerate(n_grams_by_order)
            ])

            vectors.append(self._t(n_grams))

        return vectors

    def translate(self, vectors):
        """
        Rebuilds the strings from vectors.

        :param vectors:
        :return:
        """
        strings = []
        for vector in vectors:
            strings.append(
                ''.join([
                    f'{self._inverse_find(v[0], 1)}' for v in vector
                ])
            )

        return strings

    def inverse_transform(self, vectors, **kwargs):
        """
        Given a list of vectorized strings, it retur-
        ns, for each vector, the list of n-grams that
        the indexes in the vectors represent.

        :param vectors:
        :param max_order:
        :return:
        """

        max_order = kwargs.get('max_order', self.ngramorder)

        if not 0 < max_order <= self.ngramorder:
            raise ValueError(
                f'Provided max_order is invalid. It has'
                f'to be between 1 and {self.ngramorder}'
            )

        max_order += 1

        ngrams = []
        for vector in vectors:
            next = []
            for eleme in vector:
                next.append([
                    self._inverse_find(v, i + 1)
                    for i, v in enumerate(eleme)
                ])

            ngrams.append(next)

        return ngrams

    def torch_transform(self, strings, **kwargs):
        """
        Transforms strings in tensors, rather than
        vectors (uses the same algorithm as `trans-
        form` method. Returns an array of tensors.

        :param strings:
        :param max_order:
        :return:
        """
        return [
            torch.tensor(arr) for arr in self.transform(
                                      strings, **kwargs)
        ]

    def torch_translate(self, tensors):
        """
        Same as `translate`, but for torch.Tensors.

        :param tensors:
        :return:
        """
        return self.translate(tensors.numpy())

    @staticmethod
    def _t(lists):
        """
        Transposes a list of lists of unev-
        en lengths, and pads it with zeros.

        :param lists:
        :return:
        """
        max_length = max([len(l) for l in lists])
        filler = np.zeros((
            len(lists),
            max_length
        ))

        for ix, li in enumerate(lists):
            filler[ix][:len(li)] = li

        return np.transpose(filler)

    def _find(self, ngram, order):
        lexicon = self.lexicons[order - 1]
        return lexicon.get(ngram,
                           lexicon['?'])

    def _inverse_find(self, index, order):
        order -= 1
        inverse = self.inverted[order]
        return inverse.get(index, '?')
