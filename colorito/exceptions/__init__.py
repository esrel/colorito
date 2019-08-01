class InvalidColorFormatException(Exception):

    def __init__(self, message):
        super().__init__(message)


class GenieDoesntKnowException(Exception):

    def __init__(self, message):
        super().__init__(message)


class GenieLoadException(Exception):

    def __init__(self, message):
        super().__init__(message)


class GeniePersistException(Exception):

    def __init__(self, message):
        super().__init__(message)
