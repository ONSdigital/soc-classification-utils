"""Provide data access for key files.

Filepaths are defined in config.
"""

import pandas as pd


def combine_job_title(row):
    """Produces full job title wih IND and ADD qualifiers."""
    job_title = row["natural_word"]
    if pd.notna(row["add"]):
        job_title = f"{row['add']} " + job_title
    if pd.notna(row["ind"]):
        job_title += f" ({row['ind']})"
    return job_title


def load_soc_index(filepath: str) -> pd.DataFrame:
    """Load SOC index.

    Provides a list of over 32,000 titles associated with employment.
    """
    soc_index_df = pd.read_excel(
        filepath,
        sheet_name="SOC2020 coding index",
        usecols=["SOC_2020", "INDEXOCC_-_natural_word_order", "ADD", "IND"],
        dtype=str,
    )

    soc_index_df.columns = [col.lower() for col in soc_index_df.columns]

    soc_index_df = soc_index_df.rename(
        columns={"indexocc_-_natural_word_order": "natural_word", "soc_2020": "code"}
    )

    soc_index_df = soc_index_df[soc_index_df["code"] != "}}}}"]
    soc_index_df["title"] = soc_index_df.apply(combine_job_title, axis=1)
    soc_index_df = soc_index_df.dropna(subset=["code", "natural_word"])
    soc_index_df = soc_index_df[["code", "title"]]
    soc_index_df["title"] = soc_index_df["title"].str.capitalize()
    return soc_index_df


def load_soc_structure(filepath: str) -> pd.DataFrame:
    """Load SOC structure.

    Provides structure with all levels and names of the SOC 2020.

    Returns:
        DataFrame with group code, group title, group description and list of tasks.
    """
    soc_df = pd.read_excel(
        filepath,
        sheet_name="SOC2020 descriptions",
        usecols=[
            "SOC\n2020 Major Group",
            "SOC\n2020 Sub-Major Group",
            "SOC\n2020 Minor Group",
            "SOC 2020 Unit Group",
            "SOC\n2020 \nGroup Title",
            "Typical Entry Routes And Associated Qualifications",
            "Group  Description",
            "Tasks",
        ],
        dtype=str,
    )
    soc_df.columns = [
        col.lower().replace(" ", "_").replace("__", "_").replace("\n", "")
        for col in soc_df.columns
    ]
    soc_df = soc_df.rename(
        columns={"typical_entry_routes_and_associated_qualifications": "qualifications"}
    )

    for col in soc_df.columns:
        soc_df[col] = soc_df[col].str.strip()

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