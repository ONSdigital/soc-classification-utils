"""This module provides utilities for embedding and searching occupational classification data
using Chroma vector stores and language models.

It includes functionality for embedding SOC hierarchy data, managing vector stores,
and performing similarity searches.
"""

import logging
import uuid
from typing import Any, Optional, Union

from autocorrect import Speller
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma  # pylint: disable=no-name-in-module
from occupational_classification.hierarchy.soc_hierarchy import SOC

from occupational_classification_utils.utils.soc_data_access import (
    load_soc_index,
)

logger = logging.getLogger(__name__)


# Share configuration with other modules
embedding_config = {
    "embedding_model_name": "unknown",
    "llm_model_name": "unknown",
    "db_dir": "unknown",
    "soc_index": "unknown",
    "soc_structure": "unknown",
    "soc_condensed": "unknown",
    "matches": 0,
    "index_size": 0,
}


def get_config() -> dict[str, dict[str, str]]:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "llm": {
            "llm_model_name": "gemini-1.5-flash",
            "embedding_model_name": "all-MiniLM-L6-v2",
            "db_dir": "src/occupational_classification_utils/data/vector_store",
        },
        "lookups": {
            "soc_index": (
                "src/occupational_classification_utils/data/soc_index/"
                "soc2020volume2thecodingindexexcel16102024.xlsx"
            ),
            "soc_structure": (
                "src/occupational_classification_utils/data/soc_index/"
                "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx"
            ),
            "soc_condensed": (
                "src/occupational_classification_utils/data/example/"
                "soc_4d_condensed.txt"
            ),
        },
    }


config = get_config()


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
        if embedding_model_name.startswith(
            "textembedding-"
        ) or embedding_model_name.startswith("text-embedding-"):
            self.embeddings = VertexAIEmbeddings(model_name=embedding_model_name)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.db_dir = db_dir
        self.vector_store = self._create_vector_store()
        self.k_matches = k_matches
        self.spell = Speller()
        self._index_size = self.vector_store._client.get_collection("langchain").count()

        # 🔄 Update shared config
        embedding_config["embedding_model_name"] = embedding_model_name
        embedding_config["llm_model_name"] = config["llm"].get(
            "llm_model_name", "unknown"
        )
        embedding_config["db_dir"] = db_dir
        embedding_config["matches"] = self.k_matches
        embedding_config["index_size"] = self._index_size

    def _create_vector_store(self) -> Chroma:
        """Initializes the Chroma vector store.

        Returns:
            Chroma: The LangChain vector store object for Chroma.
        """
        if self.db_dir is None:
            return Chroma(  # pylint: disable=not-callable
                embedding_function=self.embeddings
            )
        # else
        return Chroma(  # pylint: disable=not-callable
            embedding_function=self.embeddings, persist_directory=self.db_dir
        )

    def embed_index(  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
            soc_index_file (optional): Custom path or file-like object to override
                default SOC index source.
            soc_structure_file (optional): Custom path or file-like object to override
                default SOC structure source.
        """
        if from_empty:
            self.vector_store._client.delete_collection(  # pylint: disable=protected-access
                "langchain"
            )
            self.vector_store = self._create_vector_store()

        docs = []
        ids = []
        if file_object is not None:
            for line in file_object:
                if line:
                    bits = line.split(":", 1)
                    docs.append(
                        Document(
                            page_content=bits[1],
                            metadata={
                                "code": bits[0],
                            },
                        )
                    )
                    ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, line)))

        elif soc is None:
            soc_index_df = load_soc_index(
                soc_index_file or config["lookups"]["soc_index"]
            )
            logger.debug(
                "Loaded %s entries from soc_index_df.", f"{len(soc_index_df):,}"
            )
            # soc = load_hierarchy(soc_df, soc_index_df)
            for _, row in soc_index_df.iterrows():
                docs.append(
                    Document(
                        page_content=row["title"],
                        metadata={"code": row["code"]},
                    )
                )
                unique_id = str(uuid.uuid4())
                ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, unique_id)))
                # soc = load_hierarchy(soc_df, soc_index_df)
                # logger.debug("Loading entries from SOC hierarchy for embedding.")
                # for _, row in soc.all_leaf_text().iterrows():
                #     code = (row["code"])
                #     docs.append(
                #         Document(
                #             page_content=row["text"],
                #             metadata = {
                #                 "code": code
                #             },
                #         )
                #     )
                #     ids.append(str(uuid.uuid3(uuid.NAMESPACE_URL, row["text"])))

        self.vector_store.add_documents(docs, ids=ids)
        self._index_size = self.vector_store._client.get_collection(  # pylint: disable=protected-access
            "langchain"
        ).count()
        logger.debug(
            "Inserted %s entries into vector embedding database.", f"{len(docs):,}"
        )

        # Update shared config
        embedding_config["index_size"] = self._index_size
        embedding_config["soc_index"] = soc_index_file or config["lookups"]["soc_index"]
        embedding_config["soc_structure"] = (
            soc_structure_file or config["lookups"]["soc_structure"]
        )
        embedding_config["soc_condensed"] = (
            soc_index_file or config["lookups"]["soc_condensed"]
        )
        embedding_config["matches"] = self.k_matches
        embedding_config["db_dir"] = self.db_dir
        embedding_config["embedding_model_name"] = self.embeddings.model_name
        embedding_config["llm_model_name"] = config["llm"].get(
            "llm_model_name", "unknown"
        )

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
            "llm_model_name": str(embedding_config["llm_model_name"]),
            "db_dir": str(embedding_config["db_dir"]),
            "soc_index": str(embedding_config["soc_index"]),
            "soc_structure": str(embedding_config["soc_structure"]),
            "soc_condensed": str(embedding_config["soc_condensed"]),
            "matches": embedding_config["matches"],
            "index_size": embedding_config["index_size"],
        }
