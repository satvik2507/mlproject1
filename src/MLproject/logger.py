import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

logging.getLogger().addHandler(logging.StreamHandler())