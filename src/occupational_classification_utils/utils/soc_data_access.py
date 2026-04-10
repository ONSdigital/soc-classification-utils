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

_SOC_INDEX_SHEET = "SOC2020 coding index"
_SOC_STRUCTURE_SHEET = "SOC2020 descriptions"
_MAX_SOC_CODE_LENGTH = 4

_STRUCTURE_CODE_COLS = [
    "SOC\n2020 Major Group",
    "SOC\n2020 Sub-Major Group",
    "SOC\n2020 Minor Group",
    "SOC 2020 Unit Group",
]


def _combine_soc_index_job_title(row: pd.Series) -> str:
    job_title = ""
    if pd.notna(row["add"]):
        job_title += f"{row['add']} "
    if pd.notna(row["natural_word"]):
        job_title += str(row["natural_word"])
    if pd.notna(row["ind"]):
        job_title += f" ({row['ind']})"
    return job_title.strip()


def _all_prefix_codes_from_unit_codes(series: pd.Series) -> set[str]:
    out: set[str] = set()
    for raw in series.dropna():
        label = str(raw).strip()
        if not label.isdigit():
            continue
        cap = min(len(label), _MAX_SOC_CODE_LENGTH)
        for i in range(1, cap + 1):
            out.add(label[:i])
    return out


def _merge_structure_with_index_codes(
    struct_df: pd.DataFrame, index_df: pd.DataFrame
) -> pd.DataFrame:
    """Ensure every coding-index unit code (and its prefixes) has a hierarchy row."""
    codes = set(
        struct_df["code"].astype(str).str.strip()
    ) | _all_prefix_codes_from_unit_codes(index_df["code"])
    sorted_codes = sorted(codes, key=lambda c: (len(c), c))
    return pd.DataFrame({"code": sorted_codes})


def load_soc_index(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC coding index from an Excel file.

    The workbook lists index occupations and their SOC 2020 unit codes.

    Args:
        resource_ref: ``(package_name, filename)`` for the Volume 2 index workbook.

    Returns:
        DataFrame with columns ``code`` and ``title``.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading SOC index from %s", file_path)

    soc_index_df = pd.read_excel(
        file_path,
        sheet_name=_SOC_INDEX_SHEET,
        usecols=["SOC_2020", "INDEXOCC_-_natural_word_order", "ADD", "IND"],
        dtype=str,
    )
    soc_index_df.columns = [col.lower() for col in soc_index_df.columns]
    soc_index_df = soc_index_df.rename(
        columns={
            "indexocc_-_natural_word_order": "natural_word",
            "soc_2020": "code",
        }
    )
    soc_index_df = soc_index_df[soc_index_df["code"] != "}}}}"]
    soc_index_df["title"] = soc_index_df.apply(_combine_soc_index_job_title, axis=1)
    soc_index_df = soc_index_df.dropna(subset=["code", "title"])
    soc_index_df = soc_index_df[["code", "title"]]
    soc_index_df["code"] = soc_index_df["code"].astype(str).str.strip()
    soc_index_df["title"] = (
        soc_index_df["title"].astype(str).str.strip().str.capitalize()
    )
    soc_index_df = soc_index_df[soc_index_df["code"].str.fullmatch(r"\d+")]
    return soc_index_df.reset_index(drop=True)


def load_soc_structure(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SOC structure from an Excel file.

    Reads Volume 1 unit-group columns and returns a single ``code`` column
    suitable for ``load_hierarchy`` (mirrors how SIC structure feeds ``load_hierarchy``).

    Args:
        resource_ref: ``(package_name, filename)`` for the Volume 1 structure workbook.

    Returns:
        DataFrame with column ``code`` for each hierarchy node (1-4 digits).
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading SOC structure from %s", file_path)

    soc_df = pd.read_excel(
        file_path,
        sheet_name=_SOC_STRUCTURE_SHEET,
        usecols=_STRUCTURE_CODE_COLS,
        dtype=str,
    )
    codes: set[str] = set()
    for col in soc_df.columns:
        for raw in soc_df[col].dropna():
            v = str(raw).strip()
            if v.isdigit() and 1 <= len(v) <= _MAX_SOC_CODE_LENGTH:
                codes.add(v)
    sorted_codes = sorted(codes, key=lambda c: (len(c), c))
    return pd.DataFrame({"code": sorted_codes})


def load_soc_hierarchy(
    index_ref: tuple[str, str], structure_ref: tuple[str, str]
) -> SOC:
    """Loads hierarchy via ``load_soc_index``, ``load_soc_structure``, and ``load_hierarchy``."""
    soc_index_df = load_soc_index(index_ref)
    soc_df = load_soc_structure(structure_ref)
    soc_df = _merge_structure_with_index_codes(soc_df, soc_index_df)
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
