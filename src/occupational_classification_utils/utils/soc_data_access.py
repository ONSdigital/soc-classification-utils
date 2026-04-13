"""Provides data access for key files.

This module contains utility functions to load and process data from
SOC-related Excel files. The filepaths for these files are defined in
the configuration function in `embedding.py`.
"""

import logging
from importlib.resources import files

import pandas as pd
from occupational_classification.hierarchy.soc_hierarchy import SOC, load_hierarchy
from occupational_classification.meta.soc_meta import SocMeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_soc_index(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC index from an Excel file.

    Args:
        resource_ref (tuple): The path to the Excel file containing the SOC index.

    Returns:
        pd.DataFrame: A DataFrame containing index data with columns
        `code` and `title`.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)
    logger.debug("Loading SOC index from %s", file_path)

    soc_index_df = pd.read_excel(file_path, dtype=str)

    columns = {col.lower().strip(): col for col in soc_index_df.columns}
    code_column = next(
        (
            columns[name]
            for name in ("soc 2020", "soc2020", "soc 2020 code", "soc code", "code")
            if name in columns
        ),
        None,
    )
    title_column = next(
        (
            columns[name]
            for name in (
                "index entry",
                "group title",
                "description",
                "activity",
                "title",
            )
            if name in columns
        ),
        None,
    )

    if code_column is None or title_column is None:
        raise ValueError(
            "SOC index workbook must contain code and title columns "
            "(for example 'SOC 2020' and 'Index entry')."
        )

    soc_index_df = soc_index_df[[code_column, title_column]].copy()
    soc_index_df.columns = ["code", "title"]
    soc_index_df = soc_index_df.dropna(subset=["code", "title"])
    soc_index_df["code"] = soc_index_df["code"].astype(str).str.strip()
    soc_index_df["title"] = soc_index_df["title"].astype(str).str.strip()
    soc_index_df = soc_index_df[soc_index_df["code"].str.fullmatch(r"\d+")]
    return soc_index_df


def load_soc_structure(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC structure from an Excel file.

    Args:
        resource_ref (tuple): The path to the Excel file containing the SOC structure.

    Returns:
        pd.DataFrame: A DataFrame containing SOC hierarchy codes.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)
    logger.debug("Loading SOC structure from %s", file_path)

    soc_df = pd.read_excel(file_path, dtype=str)

    columns = {col.lower().strip(): col for col in soc_df.columns}
    code_column = next(
        (
            columns[name]
            for name in ("soc 2020", "soc2020", "soc code", "code", "label")
            if name in columns
        ),
        None,
    )
    if code_column is None:
        raise ValueError(
            "SOC structure workbook must contain a SOC code column "
            "(for example 'SOC 2020')."
        )

    soc_df = soc_df[[code_column]].copy()
    soc_df.columns = ["code"]
    soc_df = soc_df.dropna(subset=["code"])
    soc_df["code"] = soc_df["code"].astype(str).str.strip()
    soc_df = soc_df[soc_df["code"].str.fullmatch(r"\d+")]

    codes: set[str] = set()
    for code in soc_df["code"]:
        for i in range(1, len(code) + 1):
            codes.add(code[:i])
    return pd.DataFrame({"code": sorted(codes, key=lambda c: (len(c), c))})


def load_soc_hierarchy(
    index_ref: tuple[str, str], structure_ref: tuple[str, str]
) -> SOC:
    """Loads the SOC hierarchy from configured index and structure resources."""
    soc_index_df = load_soc_index(index_ref)
    soc_df = load_soc_structure(structure_ref)
    return load_hierarchy(soc_df, soc_index_df)


def get_soc_meta(structure_ref: tuple[str, str]):
    """Returns in-library SOC metadata (``SocMeta.soc_meta``).

    ``structure_ref`` is unused; retained for config shape parity with SIC callers.

    Args:
        structure_ref: Tuple ``(package_name, path)`` (unused).

    Returns:
        The ``soc_meta`` mapping from ``SocMeta()``.
    """
    _ = structure_ref
    return SocMeta().soc_meta


def load_text_from_config(config_section: tuple[str, str]) -> str:
    """Loads text content from a configuration file.

    Args:
        config_section: Tuple containing the package name and the filename.

    Returns:
        The file content as a string.
    """
    pkg, filename = config_section
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading text from %s", file_path)

    with file_path.open(encoding="utf-8") as f:
        return f.read()
