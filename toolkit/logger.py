import logging
import logging.config
import colorlog
from pythonjsonlogger import jsonlogger
from pathlib import Path
import sys
import gc


class Logger:
    def __init__(self, logger_name):
        self.name = logger_name

    def __del__(self):
        gc.collect()

    def get_logger(self):
        """Add logging function to project.

        It creates a console log and a file log.

        Args:
            logger_name (str): Name of the logger, recommended to use __name__ to trace the module structure

        Returns:
            Logger object: Configured logger class
        """

        try:
            self.logger = logging.getLogger(self.name)

            textlog = Path("./logs/log.txt")
            jsonlog = Path("./logs/log.json")

            console = colorlog.StreamHandler(stream=sys.stderr)
            text_logger = logging.handlers.RotatingFileHandler(
                filename=textlog, mode="a", maxBytes=1048576, backupCount=5
            )
            json_logger = logging.handlers.RotatingFileHandler(
                filename=jsonlog, mode="a", maxBytes=1048576, backupCount=5
            )

            fmt_console = colorlog.ColoredFormatter(
                "[ {log_color}{levelname:^10s}{reset} ] {message:s}", style="{"
            )
            fmt_text_logger = logging.Formatter(
                "%(asctime)s |"
                "%(levelname)-10s |"
                "%(name)-40s:%(lineno)5s:%(funcName)-25s |"
                "%(message)s"
            )
            fmt_json_logger = jsonlogger.JsonFormatter(
                "%(asctime)s"
                "%(levelname)s"
                "%(name)s"
                "%(lineno)s"
                "%(funcName)s"
                "%(message)s"
            )

            console.addFilter(lambda record: record.levelno >= logging.INFO)

            console.setFormatter(fmt_console)
            text_logger.setFormatter(fmt_text_logger)
            json_logger.setFormatter(fmt_json_logger)

            self.logger.addHandler(console)
            self.logger.addHandler(text_logger)
            self.logger.addHandler(json_logger)

            self.logger.setLevel(logging.DEBUG)
            logging.captureWarnings(True)

            self.logger.debug("INIT LOGGER")
            return self.logger

        except Exception as e:
            print("Logger not initialized!")
            raise e
