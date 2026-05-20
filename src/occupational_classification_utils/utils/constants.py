"""Module for common constant definitions.

This module contains constants used across the occupational classification utilities.
"""

from occupational_classification_utils.models.config_model import FullConfig

MAX_ALT_CANDIDATES = 10
DEFAULT_TRUNCATE_LEN = 8


def truncate_identifier(value: str | None, max_len: int = DEFAULT_TRUNCATE_LEN) -> str:
    """Return a truncated string safely, handling None and short values.

    Used for logging to preserve privacy while providing enough context.
    Mirrors industrial_classification_utils.utils.constants.truncate_identifier (SIC).

    Args:
        value (str | None): The string to truncate.
        max_len (int): Maximum length before truncation. Defaults to 8.

    Returns:
        str: Empty string if value is None/empty, otherwise truncated string
            with "..." suffix if longer than max_len.
    """
    if not value:
        return ""
    return value if len(value) <= max_len else value[:max_len] + "..."


def get_default_config() -> FullConfig:
    """Return SOC defaults in the same structural shape as sic-classification-utils.

    Top-level keys are ``embedding``, ``llm``, and ``lookups`` (see ``FullConfig``).

    Returns:
        FullConfig: Embedding model path, generative LLM defaults, and SOC lookup paths.
    """
    return {
        "embedding": {
            "embedding_model_name": "all-MiniLM-L6-v2",
            "db_dir": "src/occupational_classification_utils/data/vector_store",
            "k_matches": 20,
        },
        "llm": {
            "llm_model_name": "gemini-2.5-flash",
            "model_location": "europe-west2",
            "code_digits": 4,
            "candidates_limit": 10,
        },
        "lookups": {
            "soc_index": (
                "occupational_classification_utils.data.soc_index",
                "soc2020volume2thecodingindexexcel16102024.xlsx",
            ),
            "soc_structure": (
                "occupational_classification_utils.data.soc_index",
                "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
            ),
        },
    }
