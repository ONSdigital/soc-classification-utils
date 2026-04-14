"""This module contains tests for the EmbeddingHandler class, focusing on embedding
and searching functionalities.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from occupational_classification_utils.embed.embedding import EmbeddingHandler

# pylint: disable=redefined-outer-name


# %%
@pytest.fixture
def embedding_handler():
    """Fixture to initialise an EmbeddingHandler instance with a toy index.

    Returns:
        EmbeddingHandler: An instance of EmbeddingHandler with a toy index embedded.
    """
    embedding_handler = EmbeddingHandler(db_dir=None)
    file_path = "src/occupational_classification_utils/data/example/toy_index.txt"
    with open(file_path, encoding="utf-8") as file_object:
        embedding_handler.embed_index(from_empty=True, file_object=file_object)
    return embedding_handler


@pytest.mark.embed
def test_embed_index_with_file_object(embedding_handler):
    """Test embedding an index from a file object.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The number of entries in the index is as expected.
    """
    assert (
        embedding_handler._index_size  # pylint: disable=protected-access
        == 4  # noqa: PLR2004
    )


@pytest.mark.embed
def test_search_index(embedding_handler):
    """Test searching the index with a single query.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The top result matches the expected code.
    """
    results = embedding_handler.search_index("primary school teacher")
    assert len(results) >= 1
    assert results[0]["code"] == "2314"


@pytest.mark.embed
def test_search_index_multi(embedding_handler):
    """Test searching the index with multiple queries.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The total number of results matches the expected count.
    """
    queries = ["primary teacher", "head teacher"]
    results = embedding_handler.search_index_multi(queries)
    assert len(results) >= 1


@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("textembedding-004", "CustomVertexAIEmbeddings"),
        ("text-embedding-004", "CustomVertexAIEmbeddings"),
        ("all-MiniLM-L6-v2", "HuggingFaceEmbeddings"),
    ],
)
@pytest.mark.embed
def test_embedding_handler_initialisation(model_name, expected_class):
    """Test embedding handles initialisation.

    Args:
        model_name (str): name of embedding to be used in English.
        expected_class (str): name of embedding.
    """
    with patch(
        "occupational_classification_utils.embed.embedding.CustomVertexAIEmbeddings"
    ) as mock_vertex, patch(
        "occupational_classification_utils.embed.embedding.HuggingFaceEmbeddings"
    ) as mock_hf:
        EmbeddingHandler(embedding_model_name=model_name, db_dir=None)

        if expected_class == "HuggingFaceEmbeddings":
            mock_hf.assert_called_once_with(model_name=model_name)
            mock_vertex.assert_not_called()
        else:
            mock_vertex.assert_called_once_with(model=model_name)
            mock_hf.assert_not_called()


@pytest.mark.embed
def test_docs_ids_from_hierarchy_use_code_and_text_for_unique_ids():
    """IDs remain unique when two SOC codes share the same text."""
    handler = EmbeddingHandler(db_dir=None)
    mock_soc = MagicMock()
    mock_soc.all_leaf_text.return_value = pd.DataFrame(
        [
            {"code": "1111", "text": "Shared description"},
            {"code": "2222", "text": "Shared description"},
        ]
    )

    docs, ids, _, _ = (
        handler._docs_ids_from_hierarchy(  # pylint: disable=protected-access
            soc=mock_soc,
            soc_index_file=("pkg", "index.xlsx"),
            soc_structure_file=("pkg", "structure.xlsx"),
        )
    )

    assert len(docs) == 2  # noqa: PLR2004
    assert len(ids) == 2  # noqa: PLR2004
    assert len(set(ids)) == 2  # noqa: PLR2004
