"""This module defines configuration models for the occupational classification utilities.

The models are implemented using Python's `TypedDict` and are used to represent
configuration settings for various components of the system, such as language
models and lookup tables.

Classes:
    EmbeddingConfig: Configuration for embedding model and vector store.
    LLMConfig: Configuration for generative LLM classify defaults.
    LookupsConfig: Configuration for SOC-related lookup tables.
"""

from typing import TypedDict


class EmbeddingConfig(TypedDict):
    """Configuration for embedding model and vector store.

    Attributes:
        embedding_model_name (str): Name of the embedding model.
        db_dir (str): Directory for the database.
        k_matches (int): Number of matches to return in similarity search.
    """

    embedding_model_name: str
    db_dir: str
    k_matches: int


class LLMConfig(TypedDict):
    """Configuration for generative LLM classification (Survey Assist style).

    Attributes:
        llm_model_name (str): Name of the language model.
        model_location (str): GCP region for Vertex AI.
        code_digits (int): Number of digits in the SOC unit group code (four for SOC 2020).
        candidates_limit (int): Maximum shortlist size passed to RAG / unambiguous prompts.
    """

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
        embedding (EmbeddingConfig): Configuration for embedding model and vector store.
        llm (LLMConfig): Configuration for generative LLM defaults.
        lookups (LookupsConfig): Configuration for SOC-related lookup tables.
    """

    embedding: EmbeddingConfig
    llm: LLMConfig
    lookups: LookupsConfig
