import logging
import os
from src.config.config import LOGGER_OUTPUT

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