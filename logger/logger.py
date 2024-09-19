import logging
import os
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.BLUE
    }

    def format(self, record):
        log_message = super().format(record)
        if logging.StreamHandler in [type(h) for h in logging.getLogger().handlers]:
            return f"{self.COLORS.get(record.levelname, '')}{log_message}{Style.RESET_ALL}"
        return log_message

class Logger:
    def __init__(self, name, log_dir='logs', log_file='api.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create file handler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file), maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

# Create a global logger instance
logger = Logger('api_logger')
