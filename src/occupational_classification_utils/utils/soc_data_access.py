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


def _combine_soc_index_job_title(row: pd.Series) -> str:
    job_title = ""
    if pd.notna(row["add"]):
        job_title += f"{row['add']} "
    if pd.notna(row["indexocc"]):
        job_title += str(row["indexocc"])
    if pd.notna(row["ind"]):
        job_title += f" ({row['ind']})"
    return job_title.strip()


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

    soc_index_df = pd.read_excel(
        file_path,
        sheet_name="SOC2020 coding index",
        usecols=["SOC_2020", "INDEXOCC", "ADD", "IND"],
        dtype=str,
    )
    soc_index_df.columns = [col.lower() for col in soc_index_df.columns]
    soc_index_df = soc_index_df.rename(columns={"soc_2020": "code"})
    soc_index_df["title"] = soc_index_df.apply(_combine_soc_index_job_title, axis=1)
    soc_index_df = soc_index_df.dropna(subset=["code", "title"])
    soc_index_df = soc_index_df[["code", "title"]]
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

    soc_df = pd.read_excel(
        file_path,
        sheet_name="SOC2020 descriptions",
        usecols=[
            "SOC\n2020 Major Group",
            "SOC\n2020 Sub-Major Group",
            "SOC\n2020 Minor Group",
            "SOC 2020 Unit Group",
        ],
        dtype=str,
    )

    codes: set[str] = set()
    for col in soc_df.columns:
        for raw in soc_df[col].dropna():
            code = str(raw).strip()
            if code.isdigit():
                codes.add(code)
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
