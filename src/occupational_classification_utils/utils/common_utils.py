"""Provide data access for key files.

This module contains utility functions to load and process data.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text_from_config(file_path: str) -> str:
    """Loads text content from a configuration file.

    This function reads the content of a text file specified by the given
    configuration section and returns it as a string.

    Args:
        file_path (str): A filename of the configuration file.

    Returns:
        str: The content of the configuration file as a string.

    """
    logger.info("Loading text from %s", file_path)

    with open(file_path, encoding="utf-8") as f:
        return f.read()
