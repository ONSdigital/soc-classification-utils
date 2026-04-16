"""This module provides utilities for embedding and searching occupational classification data
using Chroma vector stores and language models.

It includes functionality for embedding SOC hierarchy data, managing vector stores,
and performing similarity searches.
"""

# Optional but doesn't hurt
import logging
import os
import sqlite3  # noqa: F401 # pylint: disable=unused-import

# Docker Image may have old sqlite3 version for ChromaDB
# Top of your module (before any langchain or chroma import)
import uuid
from typing import Any, Optional, Union

from autocorrect import Speller
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from occupational_classification.hierarchy.soc_hierarchy import SOC

from occupational_classification_utils.models.config_model import (
    FullConfig,
)
from occupational_classification_utils.utils.soc_data_access import (
    load_soc_hierarchy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Share configuration with other modules
embedding_config = {
    "embedding_model_name": "unknown",
    "db_dir": "unknown",
    "soc_index": "unknown",
    "soc_structure": "unknown",
    "matches": 0,
    "index_size": 0,
}


def get_config() -> FullConfig:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "llm": {
            "embedding_model_name": "all-MiniLM-L6-v2",  # text-embedding-004
            "db_dir": "src/occupational_classification_utils/data/vector_store",
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


config = get_config()
MAX_BATCH_SIZE = 5400


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    """Custom VertexAIEmbeddings to specify task type for embeddings."""

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 0,
        *,
        embeddings_task_type="SEMANTIC_SIMILARITY",
    ) -> list[list[float]]:
        """Embeds a list of documents using the specified task type."""
        return super().embed_documents(
            texts,
            batch_size=batch_size,
            embeddings_task_type=embeddings_task_type,
        )

    def embed_query(
        self,
        text: str,
        *,
        embeddings_task_type="SEMANTIC_SIMILARITY",
    ) -> list[float]:
        """Embeds a single query using the specified task type."""
        return super().embed_query(text, embeddings_task_type=embeddings_task_type)


class EmbeddingHandler:
    """Handles embedding operations for the Chroma vector store.

    Attributes:
        embeddings (Any): The embedding model used for vectorization.
        db_dir (str): Directory where the vector store database is located.
        vector_store (Chroma): The Chroma vector store instance.
        k_matches (int): Number of nearest matches to retrieve during search.
        spell (Speller): Autocorrect spell checker instance.
        _index_size (int): Number of entries in the vector store.
    """

    def __init__(
        self,
        embedding_model_name: str = config["llm"]["embedding_model_name"],
        db_dir: str = config["llm"]["db_dir"],
        k_matches: int = 20,
    ):
        """Initializes the EmbeddingHandler.

        Args:
            embedding_model_name (str, optional): Name of the embedding model to use.
                Defaults to the value in the configuration file.
            db_dir (str, optional): Directory for the vector store database.
                Defaults to the value in the configuration file.
            k_matches (int, optional): Number of nearest matches to retrieve.
                Defaults to 20.
        """
        self.embeddings: Any  # Use Any if no common base type exists
        if embedding_model_name.startswith(("textembedding-", "text-embedding-")):
            self.embeddings = CustomVertexAIEmbeddings(model=embedding_model_name)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        logger.info("Using embedding model: %s", embedding_model_name)

        self.db_dir = db_dir
        self.vector_store = self._create_vector_store()
        self.k_matches = k_matches
        self.spell = Speller()
        self._index_size = self.vector_store._client.get_collection("langchain").count()

        logger.info(
            "Vector store created in: %s containing %s entries.",
            self.db_dir,
            self._index_size,
        )

        # 🔄 Update shared config
        embedding_config["embedding_model_name"] = embedding_model_name
        embedding_config["db_dir"] = db_dir
        embedding_config["matches"] = self.k_matches
        embedding_config["index_size"] = self._index_size
        logger.debug("EmbeddingHandler initialised with config: %s", embedding_config)

    def _create_vector_store(self) -> Chroma:
        """Initializes the Chroma vector store.

        Returns:
            Chroma: The LangChain vector store object for Chroma.
        """
        if self.db_dir is None:
            logger.warning("No db_dir provided; using in-memory vector store.")
            return Chroma(  # pylint: disable=not-callable
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "l2"},
            )
        # else

        if not os.path.exists(self.db_dir):
            logger.warning("Persist directory does not exist: %s", self.db_dir)
        else:
            logger.debug("Persist directory exists: %s", self.db_dir)
            logger.debug("Readable: %s", os.access(self.db_dir, os.R_OK))
            logger.debug("Writable: %s", os.access(self.db_dir, os.W_OK))

        try:
            chroma = Chroma(  # pylint: disable=not-callable
                embedding_function=self.embeddings,
                persist_directory=self.db_dir,
                collection_metadata={"hnsw:space": "l2"},
            )
            logger.info("Vector store created successfully.")
            return chroma
        except Exception as e:
            logger.exception("Failed to create vector store: %s", e)
            raise

    def _docs_ids_from_file_object(
        self, file_object: Any
    ) -> tuple[list[Document], list[str]]:
        """Build (docs, ids) from a line-oriented file object (code: description)."""
        docs: list[Document] = []
        ids: list[str] = []
        for line in file_object:
            if line:
                bits = line.split(":", 1)
                docs.append(
                    Document(
                        page_content=bits[1],
                        metadata={
                            "code": bits[0],
                            "four_digit_code": bits[0][0:4],
                            "two_digit_code": bits[0][0:2],
                        },
                    )
                )
                ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))
        return docs, ids

    def _docs_ids_from_hierarchy(
        self,
        soc: Optional[SOC],
        soc_index_file: Any,
        soc_structure_file: Any,
    ) -> tuple[list[Document], list[str], Any, Any]:
        """Build (docs, ids) from SOC hierarchy; load from files if soc is None.

        Returns effective index/structure paths for config.
        """
        if soc_index_file is None:
            soc_index_file = config["lookups"]["soc_index"]
        if soc_structure_file is None:
            soc_structure_file = config["lookups"]["soc_structure"]
        if soc is None:
            logger.info(
                "Loading SOC hierarchy from files: %s, %s",
                soc_index_file,
                soc_structure_file,
            )
            soc = load_soc_hierarchy(soc_index_file, soc_structure_file)
        docs: list[Document] = []
        ids: list[str] = []
        for _, row in soc.all_leaf_text().iterrows():  # type: ignore[union-attr]
            code = str(row["code"]).strip()
            docs.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "code": code,
                        "four_digit_code": code[:4],
                        "two_digit_code": code[:2],
                    },
                )
            )
            # Keep deterministic IDs while preventing collisions when
            # different SOC codes share the same text description.
            id_seed = f"{code}:{row['text']}"
            ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, id_seed)))
        return docs, ids, soc_index_file, soc_structure_file

    def embed_index(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        self,
        from_empty: bool = True,
        soc: Optional[SOC] = None,
        file_object=None,
        soc_index_file=None,
        soc_structure_file=None,
    ):
        """Embeds the index entries into the vector store.

        Args:
            from_empty (bool, optional): Whether to drop the current vector store
                content and start fresh. Defaults to True.
            soc (Optional[SOC], optional): The SOC hierarchy object. If None, the hierarchy
                is loaded from files specified in the config. Defaults to None.
            file_object (StringIO object, optional): The index file as a StringIO object.
                If provided, the file will be read line by line and embedded.
                Each line should have the format **code**: **description**.
            soc_index_file (optional): Config-style tuple (package, path) to override
                default SOC index source. Must be tuple for data-access parity with SIC.
            soc_structure_file (optional): Config-style tuple (package, path) to override
                default SOC structure source. Must be tuple for data-access parity with SIC.
        """
        # Log parameters
        logger.info(
            "Embedding index: from_empty=%s, soc=%s, file_object=%s, "
            "soc_index_file=%s, soc_structure_file=%s",
            from_empty,
            soc,
            file_object,
            soc_index_file,
            soc_structure_file,
        )
        if from_empty:
            logger.info("Dropping existing vector store content.")
            self.vector_store._client.delete_collection(  # pylint: disable=protected-access
                "langchain"
            )
            self.vector_store = self._create_vector_store()

        if file_object is not None:
            docs, ids = self._docs_ids_from_file_object(file_object)
            effective_index_file = soc_index_file or config["lookups"]["soc_index"]
            effective_structure_file = (
                soc_structure_file or config["lookups"]["soc_structure"]
            )
        else:
            docs, ids, effective_index_file, effective_structure_file = (
                self._docs_ids_from_hierarchy(soc, soc_index_file, soc_structure_file)
            )

        def split_into_batches(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]

        for batch_docs, batch_ids in zip(
            split_into_batches(docs, MAX_BATCH_SIZE),
            split_into_batches(ids, MAX_BATCH_SIZE),
        ):
            self.vector_store.add_documents(batch_docs, ids=batch_ids)
        self._index_size = self.vector_store._client.get_collection(  # pylint: disable=protected-access
            "langchain"
        ).count()

        logger.debug(
            "Inserted %s entries into vector embedding database.", f"{len(docs):,}"
        )

        # Update shared config
        embedding_config["index_size"] = self._index_size
        embedding_config["soc_index"] = effective_index_file
        embedding_config["soc_structure"] = effective_structure_file
        embedding_config["matches"] = self.k_matches
        embedding_config["db_dir"] = self.db_dir
        embedding_config["embedding_model_name"] = self.embeddings.model_name
        logger.info("Embedding config updated: %s", embedding_config)

    def search_index(
        self, query: str, return_dicts: bool = True
    ) -> Union[list[dict], list[tuple[Document, float]]]:
        """Returns k document chunks with the highest relevance to the query.

        Args:
            query (str): Query string for which the most relevant index entries
                will be returned.
            return_dicts (bool, optional): If True, returns data as a list of
                dictionaries. Otherwise, returns document tuples. Defaults to True.

        Returns:
            Union[list[dict], list[tuple[Document, float]]]: List of top k index entries
            by relevance.
        """
        top_matches = self.vector_store.similarity_search_with_score(
            query=query, k=self.k_matches
        )

        if return_dicts:
            return [
                {"distance": float(doc[1])}
                | {"title": doc[0].page_content}
                | doc[0].metadata
                for doc in top_matches
            ]
        return top_matches

    def search_index_multi(self, query: list[str]) -> list[dict]:
        """Returns k document chunks with the highest relevance to a list of query fields.

        Args:
            query (list[str]): List of query fields (in priority order) for which
                the most relevant index entries will be returned.
                Example: [industry_descr, job_title, job_descr].

        Returns:
            list[dict]: List of top k index entries by relevance.
        """
        query = [x for x in query if x is not None]
        search_terms_list = set()
        for i in range(len(query)):
            x = " ".join(query[: (i + 1)])
            search_terms_list.add(x)
            search_terms_list.add(self.spell(x))
        short_list = [y for x in search_terms_list for y in self.search_index(query=x)]
        return sorted(short_list, key=lambda x: x["distance"])  # type: ignore

    def get_embed_config(self) -> dict:
        """Returns the current embedding configuration as a dictionary."""
        return {
            "embedding_model_name": str(embedding_config["embedding_model_name"]),
            "db_dir": str(embedding_config["db_dir"]),
            "soc_index": str(embedding_config["soc_index"]),
            "soc_structure": str(embedding_config["soc_structure"]),
            "matches": embedding_config["matches"],
            "index_size": embedding_config["index_size"],
        }
