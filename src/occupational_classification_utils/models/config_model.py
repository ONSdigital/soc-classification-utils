"""This module defines configuration models for the occupational classification utilities.

The models are implemented using Python's `TypedDict` and are used to represent
configuration settings for various components of the system, such as language
models and lookup tables.

Classes:
    LLMConfig: Configuration for language and embedding models.
    LookupsConfig: Configuration for SOC-related lookup tables.
"""

from typing import TypedDict


class LLMConfig(TypedDict):
    """Configuration for language and embedding models and location of
    the vector store.

    Attributes:
        embedding_model_name (str): Name of the embedding model.
        db_dir (str): Directory for the database.
    """

    embedding_model_name: str
    db_dir: str


class LookupsConfig(TypedDict):
    """Configuration for SOC-related lookup tables.

    Attributes:
        soc_index (tuple[str, str]): Path to the SOC index file.
        soc_structure (tuple[str, str]): Path to the SOC structure file.
    """

    soc_index: tuple[str, str]
    soc_structure: tuple[str, str]


class FullConfig(TypedDict):
    """Full configuration model for the SOC classification.

    Attributes:
        llm (LLMConfig): Configuration for language and embedding models.
        lookups (LookupsConfig): Configuration for SOC-related lookup tables.
    """

    llm: LLMConfig
    lookups: LookupsConfig
