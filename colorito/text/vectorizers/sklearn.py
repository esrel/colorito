from colorito.text.vectorizers import ColorNameVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class ColorNameSklearnVectorizer(ColorNameVectorizer):

    def __init__(self, sklearn_vectorizer):
        super().__init__()
        self.sklearn_vect = sklearn_vectorizer

    def fit(self, palette):
        self.sklearn_vect.fit(palette)

    def transform(self, color_name):
        if not isinstance(color_name, list):
            color_name = [color_name]

        return self.sklearn_vect.transform(color_name)[0]


class ColorNameCountVectorizer(ColorNameSklearnVectorizer):

    def __init__(self, **kwargs):
        super().__init__(CountVectorizer(**kwargs))


class ColorNameTfidfVectorizer(ColorNameSklearnVectorizer):

    def __init__(self, **kwargs):
        super().__init__(TfidfVectorizer(**kwargs))