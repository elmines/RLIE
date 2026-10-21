import logging
import colorlog
from logging import Logger

class LoggerConfig:
    level = None

    @staticmethod
    def setup_logger(
        level=logging.DEBUG,
        log_file_path=None,
    ):
        LoggerConfig.level = level

        # Colored formatter for console
        console_formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s] %(purple)s%(name)s: %(blue)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Plain formatter for file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handlers: list[logging.Handler] = []

        if log_file_path is not None:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        logging.basicConfig(
            handlers=handlers,
        )


    @staticmethod
    def get_logger(name: str) -> Logger:
        if LoggerConfig.level is None:
            LoggerConfig.setup_logger()
        logger = logging.getLogger(name)
        logger.setLevel(LoggerConfig.level)
        return logger
