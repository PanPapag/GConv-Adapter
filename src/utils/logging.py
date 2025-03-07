import os
import logging


def setup_logging(log_folder: str) -> logging.Logger:
    """
    Set up logging to a file in the specified folder.

    Args:
        log_folder (str): Folder to save the log file.

    Returns:
        logging.Logger: The configured logger.
    """
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "pipeline.log")

    # Get the root logger
    logger = logging.getLogger()

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and set it for both handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
