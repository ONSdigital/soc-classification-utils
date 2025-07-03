"""Provides data access for key files.

This module contains utility functions to load and process data from
SOC-related Excel files. The filepaths for these files are defined in
the configuration function in `embedding.py`.
"""

import logging
from importlib.resources import files

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_soc_index(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC index from an Excel file.

    The SOC index provides a list of activities and their associated SOC codes.

    Args:
        resource_ref (tuple): A tuple containing the package name and filename
            of the Excel file containing the SOC index.

    Returns:
        pd.DataFrame: A DataFrame containing the SOC index with columns
        `code` and `title`.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.info("Loading SOC index from %s", file_path)

    soc_index_df = pd.read_excel(
        file_path,
        sheet_name="Alphabetical Index",
        skiprows=2,
        usecols=["SOC 2020", "Title"],
        dtype=str,
    )

    soc_index_df.columns = ["code", "title"]

    return soc_index_df


def load_soc_structure(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC structure from an Excel file.

    This function loads a worksheet containing all the levels and names
    of the UK SOC 2020 hierarchy.

    Args:
        resource_ref (tuple): A tuple containing the package name and filename
            of the Excel file containing the SOC structure.

    Returns:
        pd.DataFrame: A DataFrame containing the SOC structure.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.info("Loading SOC structure from %s", file_path)

    soc_df = pd.read_excel(
        file_path,
        sheet_name="SOC2020 descriptions",
        dtype=str,
    )

    # Clean up column names to match what the SOC meta library expects
    soc_df.columns = soc_df.columns.str.replace('\n', ' ').str.lower().str.replace(' ', '_')
    # Handle specific case where 'soc_2020_unit_group' should remain as 'soc_2020_unit_group'
    soc_df.columns = soc_df.columns.str.replace('soc_2020_', 'soc2020_').str.replace('soc2020_unit_group', 'soc_2020_unit_group')

    return soc_df


def load_text_from_config(config_section: tuple[str, str]) -> str:
    """Loads text content from a configuration file.

    This function reads the content of a text file specified by the given
    configuration section and returns it as a string.

    Args:
        config_section (tuple[str, str]): A tuple containing the package name
            and the filename of the configuration file.

    Returns:
        str: The content of the configuration file as a string.

    """
    pkg, filename = config_section
    file_path = files(pkg).joinpath(filename)

    logger.info("Loading text from %s", file_path)

    with file_path.open(encoding="utf-8") as f:
        return f.read() 