"""This module initialises and updates an embeddings index for occupational classification
and then performs an llm lookup.

The module uses the `EmbeddingHandler` class from the
`occupational_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

The example then uses a `ClassificationLLM` object from
the `occupational_classification_utils.llm.llm` package to perform a lookup using
the embeddings index, followed by a call to the llm model.
"""

import asyncio

from occupational_classification_utils.embed.embedding import EmbeddingHandler
from occupational_classification_utils.llm.llm import ClassificationLLM

embed = EmbeddingHandler()

EXAMPLE_QUERY = "school teacher primary education"
LLM_MODEL = "gemini-1.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"
CANDIDATE_LIMIT = 100

try:
    embed.embed_index(from_empty=False)
except Exception:
    embed.embed_index(from_empty=True)

# The vector store is not yet decoupled from the LLM.
# The expectation is that the embedding search
# takes place within LLM
gemini_llm = ClassificationLLM(model_name=LLM_MODEL)


async def run_sa_rag() -> None:
    response, short_list, prompt = await gemini_llm.sa_rag_soc_code(
        ORG_DESCRIPTION,
        JOB_TITLE,
        JOB_DESCRIPTION,
        candidates_limit=CANDIDATE_LIMIT,
    )
    print(response)


asyncio.run(run_sa_rag())
