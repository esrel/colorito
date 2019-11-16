class SaveError(Exception):

    def __init__(self, cls, err):
        super(SaveError, self).__init__(
         f'Could not save {cls}: {err}')


class LoadError(Exception):

    def __init__(self, cls, err):
        super(LoadError, self).__init__(
         f'Could not load {cls}: {err}')
