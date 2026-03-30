# pylint: disable=logging-not-lazy,logging-fstring-interpolation,too-many-lines
"""This module provides utilities for leveraging Large Language Models (LLMs)
to classify respondent data into Standard Occupational Classification (SOC) codes.

The `ClassificationLLM` class encapsulates the logic for using LLMs to perform
classification tasks, including direct generative methods and Retrieval Augmented
Generation (RAG). It supports various prompts and configurations for different
classification scenarios, such as unambiguous classification, reranking, and
general-purpose classification.

Classes:
    ClassificationLLM: A wrapper for LLM-based SOC classification logic.

Functions:
    (None at the module level)
"""

from collections import defaultdict
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from occupational_classification.hierarchy.soc_hierarchy import SOC
from pydantic import SecretStr
from survey_assist_utils.logging import get_logger

from occupational_classification_utils.embed.embedding import get_config
from occupational_classification_utils.llm.prompt import (
    FIX_PARSING_PROMPT,
    SA_SOC_PROMPT_RAG,
    SOC_PROMPT_PYDANTIC,
)
from occupational_classification_utils.models.response_model import SocResponse
from occupational_classification_utils.utils.soc_data_access import (
    get_soc_meta,
    load_soc_hierarchy,
)

logger = get_logger(__name__)
config = get_config()


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
class ClassificationLLM:
    """Wraps the logic for using an LLM to classify respondent's data
    based on provided index. Includes direct (one-shot) generative llm
    method and Retrieval Augmented Generation (RAG).

    Args:
        model_name (str): Name of the model. Defaults to the value in the `config` file.
            Used if no LLM object is passed.
        llm (LLM): LLM to use. Optional.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 1600.
        temperature (float): Temperature of the LLM model. Defaults to 0.0.
        verbose (bool): Whether to print verbose output. Defaults to False.
        openai_api_key (str): OpenAI API key. Optional, but needed for OpenAI models.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = config["llm"]["llm_model_name"],
        llm: Optional[Union[ChatVertexAI, ChatOpenAI]] = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        verbose: bool = True,
        openai_api_key: Optional[SecretStr] = None,
    ):
        """Initialises the ClassificationLLM object."""
        logger.info(
            f"Init LLM {llm} model: {model_name} max_tokens: {max_tokens} temp: {temperature}"
        )
        if llm is not None:
            self.llm = llm
        elif model_name.startswith("text-") or model_name.startswith("gemini"):
            # Mirror SIC: ChatVertexAI, europe-west1, thinking_budget=0
            self.llm = ChatVertexAI(
                model_name=model_name,
                max_output_tokens=max_tokens,
                temperature=temperature,
                location="europe-west1",
                model_kwargs={"thinking_budget": 0},  # Reduce latency
            )
        elif model_name.startswith("gpt"):
            if openai_api_key is None:
                raise NotImplementedError("Need to provide an OpenAI API key")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
            )
        else:
            raise NotImplementedError("Unsupported model family")

        self.soc_meta = get_soc_meta(config["lookups"]["soc_structure"])
        self.soc_prompt = SOC_PROMPT_PYDANTIC
        self.sa_soc_prompt_rag = SA_SOC_PROMPT_RAG
        self.soc: Optional[SOC] = None
        self.verbose = verbose

    @lru_cache  # noqa: B019
    async def get_soc_code(
        self,
        job_title: str,
    ) -> SocResponse:
        """Generates a SOC classification based on respondent's data
        using the full SOC index embedded in the query (mirror SIC one-shot).

        Args:
            job_title (str): The title of the job.

        Returns:
            SocResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error parsing the response from the LLM model.

        """
        chain = self.soc_prompt | self.llm
        response = await chain.ainvoke(
            {
                "job_title": job_title,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug(f"LLM response: {response}")
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SocResponse,
        )

        try:
            chain = FIX_PARSING_PROMPT | self.llm
            response = await chain.ainvoke(
                {
                    "llm_output": str(response.content),
                    "format_instructions": parser.get_format_instructions(),
                },
                return_only_outputs=True,
            )
            validated_answer_sr = parser.parse(str(response.content))
            logger.debug("Successfully parsed reformatted response.")
        except (ValueError, AttributeError) as parse_error2:
            logger.error(
                f"Failed to parse response again: {parse_error2}",
                error=str(parse_error2),
            )
            logger.warning(
                "Failed to parse response again",
                response_content=str(response.content),
            )
            reasoning = (
                f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
            )
            validated_answer_sr = SocResponse(
                codable=False, soc_candidates=[], reasoning=reasoning
            )

        return validated_answer_sr

    def _prompt_candidate(
        self,
        code: str,
        job_titles: list[str],
        include_all: bool = False,
    ) -> str:
        """Reformat the candidate activities for the prompt.

        Args:
            code (str): The code for the item.
            job_titles (list[str]): The list of example job titles.
            include_all (bool, optional): Whether to include all the soc metadata.

        Returns:
            str: A formatted string containing the code, title, and example activities.
        """
        if self.soc is None:
            self.soc = load_soc_hierarchy(
                config["lookups"]["soc_index"],
                config["lookups"]["soc_structure"],
            )

        item = self.soc[code]
        txt = "{" + f"Code: {item.soc_code}, Title: {item.group_title}"
        txt += f", Example job_titles: {', '.join(job_titles)}"

        if include_all:
            pass  # Full metadata optional; structure matches SIC _prompt_candidate
        return txt + "}"

    def _prompt_candidate_list(
        self,
        short_list: Union[list[dict], list[tuple[Document, float]]],  # list[dict],
        chars_limit: int = 14000,
        candidates_limit: int = 5,
        titles_limit: int = 3,
        code_digits: int = 4,
    ) -> str:
        """Create candidate list for the prompt based on the given parameters.

        This method takes a structured list of candidates and generates a short
        string list based on the provided parameters. It filters the candidates
        based on the code digits and activities limit, and shortens the list to
        fit the character limit.

        Args:
            short_list (list[dict]): A list of candidate dictionaries.
            chars_limit (int, optional): The character limit for the generated
                prompt. Defaults to 14000.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            titles_limit (int, optional): The maximum number of job titles
                to include for each code. Defaults to 3.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 4.

        Returns:
            str: The generated candidate list for the prompt.
        """
        a: defaultdict[Any, list] = defaultdict(list)

        logger.debug(
            f"Chars Lmt: {chars_limit} Candidate Lmt: {candidates_limit} "
            f"Titles Lmt: {titles_limit} Short List Len: {len(short_list)} "
            f"Code Digits: {code_digits}"
        )

        for item in short_list:
            if (
                isinstance(item, dict)
                and item["title"] not in a[item["code"][:code_digits]]
            ):
                a[item["code"][:code_digits]].append(item["title"])

        soc_candidates = [
            self._prompt_candidate(code, job_titles[:titles_limit])
            for code, job_titles in a.items()
        ][:candidates_limit]

        if chars_limit:
            chars_count = np.cumsum([len(x) for x in soc_candidates])
            nn = sum(x <= chars_limit for x in chars_count)
            if nn < len(soc_candidates):
                logger.warning(
                    f"Shortening list of candidates to fit token limit from "
                    f"{len(soc_candidates)} to {nn}"
                )
                soc_candidates = soc_candidates[:nn]

        return "\n".join(soc_candidates)

    async def sa_rag_soc_code(  # noqa: PLR0913
        self,
        job_title: Optional[str] = None,
        expand_search_terms: bool = True,
        code_digits: int = 4,
        candidates_limit: int = 5,
        short_list: Optional[list[dict[Any, Any]]] = None,
    ) -> tuple[SocResponse, Optional[list[dict[Any, Any]]], Optional[Any]]:
        """Generates a SOC classification based on respondent's data using RAG approach.

        Caller must provide short_list (e.g. from vector store API). Mirrors
        sic-classification-utils sa_rag_sic_code (raise when short_list is None;
        use SocResponse throughout, align with SIC).

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            expand_search_terms (bool, optional): Kept for API compatibility;
                unused (short_list is required from caller). Defaults to True.
            code_digits (int, optional): The number of digits in the generated
                SOC code. Defaults to 4.
            candidates_limit (int, optional): The maximum number of SOC code candidates
                to consider. Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from
                embedding or vector store search (e.g. from soc-classification-vector-store).
                Each dict should have "code" and "title" keys. When provided, the
                embedding handler is not used.

        Returns:
            SocResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If short_list is None.

        """
        _ = expand_search_terms  # API compatibility; unused when short_list required

        def prep_call_dict(job_title, soc_codes):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            call_dict = {
                "job_title": job_title,
                "soc_index": soc_codes,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        soc_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            job_title=job_title,
            soc_codes=soc_codes,
        )

        if self.verbose:
            final_prompt = self.sa_soc_prompt_rag.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sa_soc_prompt_rag | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = SocResponse(
                codable=False,
                followup="Follow-up question not available due to error.",
                soc_candidates=[],
                reasoning="Error from chain, exit early",
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug(f"LLM response: {response}")

        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SocResponse,
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = SocResponse(
                    codable=False,
                    followup="Follow-up question not available due to error.",
                    soc_candidates=[],
                    reasoning=reasoning,
                )

        return validated_answer, short_list, call_dict
