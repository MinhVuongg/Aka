import logging
import os
from src.config.config import LOGGER_OUTPUT
import colorama

# Khởi tạo colorama để hỗ trợ màu trên tất cả nền tảng
colorama.init()

# Định nghĩa các màu ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ColoredLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def info(self, msg, *args, color=None, **kwargs):
        if color:
            msg = f"{color}{msg}{RESET}"
        super().info(msg, *args, **kwargs)

    def error(self, msg, *args, color=RED, **kwargs):
        if color:
            msg = f"{color}{msg}{RESET}"
        super().error(msg, *args, **kwargs)

    def warning(self, msg, *args, color=YELLOW, **kwargs):
        if color:
            msg = f"{color}{msg}{RESET}"
        super().warning(msg, *args, **kwargs)

logging.setLoggerClass(ColoredLogger)

os.makedirs(os.path.dirname(LOGGER_OUTPUT), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # Handler để in log ra console
        logging.StreamHandler(),
        # Handler để ghi log ra file
        logging.FileHandler(LOGGER_OUTPUT)
    ]
)
logger = logging.getLogger(__name__)