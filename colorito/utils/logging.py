import logging

COLORS = {
    'critical': "\x1b[;41m",
    'warning': "\x1b[33;1m",
    'error': "\x1b[31;1m",
    'info': "\x1b[32;1m",
    'debug': "\x1b[35;1m"
}


class ColoredLogger(object):

    def __init__(self):
        self.logger = logging.getLogger("colorito")

    def _colored_log(self, message, level):
        message = COLORS[level] + message + "\x1b[0m"
        {
            'info': self.logger.info,
            'error': self.logger.error,
            'debug': self.logger.debug,
            'warning':  self.logger.warning,
            'critical': self.logger.critical

        }[level](message)

    def critical(self, message):
        self._colored_log(message, 'critical')

    def warning(self, message):
        self._colored_log(message, 'warning')

    def debug(self, message):
        self._colored_log(message, 'debug')

    def error(self, message):
        self._colored_log(message, 'error')

    def info(self, message):
        self._colored_log(message, 'info')


logger = ColoredLogger()
