"""Demonstration module mirroring sic-classification-utils/demos/llm/llm_embedding_example.py.

Vector search is decoupled from the LLM: the shortlist is loaded from JSON (mock embed
results), then passed to Survey Assist-style RAG and two-step classify methods.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from occupational_classification_utils.llm.llm import ClassificationLLM

DATA_DIR = Path(__file__).resolve().parent / "data"

LLM_MODEL = "gemini-2.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"
CANDIDATE_LIMIT = 100

with (DATA_DIR / "school_embed_short_list_soc.json").open(encoding="utf-8") as handle:
    EXAMPLE_EMBED_SHORT_LIST = json.load(handle)

gemini_llm = ClassificationLLM(model_name=LLM_MODEL)


async def main() -> None:
    """Run one-shot, single-step RAG, and unambiguous SOC examples."""
    response_soc = await gemini_llm.get_soc_code(
        JOB_TITLE,
        JOB_DESCRIPTION,
        "degree",
        False,
        ORG_DESCRIPTION,
    )
    print(response_soc.model_dump_json(indent=2))

    response, _short_list, _prompt = await gemini_llm.sa_rag_soc_code(
        ORG_DESCRIPTION,
        JOB_TITLE,
        JOB_DESCRIPTION,
        candidates_limit=CANDIDATE_LIMIT,
        short_list=EXAMPLE_EMBED_SHORT_LIST,
    )
    print(response.model_dump_json(indent=2))

    query_response, _call_dict = await gemini_llm.unambiguous_soc_code(
        industry_descr=ORG_DESCRIPTION,
        semantic_search_results=EXAMPLE_EMBED_SHORT_LIST,
        job_title=JOB_TITLE,
        job_description=JOB_DESCRIPTION,
    )
    print(query_response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
