import os
from datum.utility.logging.logger import Logger

file_path = "./outputs/datum.log"

if os.path.exists(file_path):
    os.remove(file_path)

logger = Logger("./outputs/datum.log", True)
