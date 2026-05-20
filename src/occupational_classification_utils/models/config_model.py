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
        llm_model_name (str): Name of the generative LLM (Survey Assist classify path).
        model_location (str): GCP region for Vertex AI.
        code_digits (int): Number of digits in the SOC unit group code (four for SOC 2020).
        candidates_limit (int): Maximum shortlist size passed to RAG / unambiguous prompts.
    """

    embedding_model_name: str
    db_dir: str
    llm_model_name: str
    model_location: str
    code_digits: int
    candidates_limit: int


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
