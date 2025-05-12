"""This module provides utilities for leveraging Large Language Models (LLMs)
to classify respondent data into Standard Occupational Classification (SOC) codes.

The `ClassificationLLM` class encapsulates the logic for using LLMs to perform
classification tasks, including direct generative methods and Retrieval Augmented
Generation (RAG). It supports various prompts and configurations for different
classification scenarios, such as unambiguous classification, reranking, and
general-purpose classification.

Classes:
    ClassificationLLM: A wrapper for LLM-based SIC classification logic.

Functions:
    (None at the module level)
"""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from occupational_classification.hierarchy.soc_hierarchy import load_hierarchy
from occupational_classification.meta.soc_meta import soc_meta
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import VertexAI
from langchain_openai import ChatOpenAI

from occupational_classification_utils.embed.embedding import get_config
from occupational_classification_utils.llm.prompt import (
    GENERAL_PROMPT_RAG,
    SA_SOC_PROMPT_RAG,
    SOC_PROMPT_PYDANTIC,
)
from occupational_classification_utils.models.response_model import (
    SocResponse,
    SurveyAssistSocResponse,
)
from occupational_classification_utils.utils.soc_data_access import (
    load_sic_index,
    load_sic_structure,
)

logger = logging.getLogger(__name__)
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

    def __init__(
        self,
        model_name: str = config["llm"]["llm_model_name"],
        llm: Optional[Union[VertexAI, ChatOpenAI]] = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        verbose: bool = True,
        openai_api_key: Optional[str] = None,
    ):
        """Initialises the ClassificationLLM object."""
        print(f"model_name: {model_name}")
        if llm is not None:
            self.llm = llm
        elif model_name.startswith("text-") or model_name.startswith("gemini"):
            self.llm = VertexAI(
                model_name=model_name,
                max_output_tokens=max_tokens,
                temperature=temperature,
                location="europe-west2",
            )
        elif model_name.startswith("gpt"):
            if openai_api_key is None:
                raise NotImplementedError("Need to provide an OpenAI API key")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise NotImplementedError("Unsupported model family")

        self.soc_prompt = SOC_PROMPT_PYDANTIC
        self.soc_meta = soc_meta
        self.sa_soc_prompt_rag = SA_SOC_PROMPT_RAG
        self.soc = None
        self.verbose = verbose

    @lru_cache
    def get_soc_code(
        self,
        job_title: str,
        job_description: str,
        level_of_education: str,
        manage_others: bool,
        industry_descr: str,
    ) -> SocResponse:
        """
        Generates a SOC classification based on respondent's data
        using a whole condensed index embedded in the query.

        Args:
            job_title (str): The title of the job.
            job_description (str): The description of the job.
            level_of_education (str): The level of education required for the job.
            manage_others (bool): Indicates whether the job involves managing others.
            industry_descr (str): The description of the industry.

        Returns:
            SocResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error parsing the response from the LLM model.

        """
        chain = LLMChain(llm=self.llm, prompt=self.soc_prompt)
        response = chain.invoke(
            {
                "job_title": job_title,
                "job_description": job_description,
                "level_of_education": level_of_education,
                "manage_others": manage_others,
                "industry_descr": industry_descr,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug(f"{response=}")
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(pydantic_object=SocResponse)
        try:
            validated_answer = parser.parse(response["text"])
        except Exception as parse_error:
            logger.error(f"Unable to parse llm response: {str(parse_error)}")
            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = SocResponse(
                codable=False, soc_candidates=[], reasoning=reasoning
            )

        return validated_answer

    def _prompt_candidate(
        self, code: str, job_titles: list[str], include_all: bool = False
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
            soc_index_df = load_soc_index(config["lookups"]["soc_index"])
            soc_df_input = load_soc_structure(config["lookups"]["soc_structure"])
            soc_df = socDB.create_soc_dataframe(soc_df_input)    
            self.soc = load_hierarchy(soc_df, soc_index_df)
        if self.sic is None:
            sic_index_df = load_sic_index(config["lookups"]["sic_index"])
            sic_df = load_sic_structure(config["lookups"]["sic_structure"])
            self.sic = load_hierarchy(sic_df, sic_index_df)

        item = self.soc[code]
        txt = "{" + f"Code: {item.soc_code}, Title: {item.group_title}"
        txt += f", Example activities: {', '.join(activities)}"
        # if include_all:
        #     if item.soc_meta.group_description:
        #         txt += f", Description: {item.soc_meta.detail}"
        #     if item.soc_meta.qualifications:
        #         txt += f", Qualifications: {', '.join(item.sic_meta.includes)}"
        return txt + "}"

    def _prompt_candidate_list(
        self,
        short_list: list[dict],
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
                the code for filtering candidates. Defaults to 5.

        Returns:
            str: The generated candidate list for the prompt.
        """
        a: defaultdict[Any, list] = defaultdict(list)

        logger.debug(
            "Chars Lmt: %d Candidate Lmt: %d Titles Lmt: %d Short List Len: %d Code Digits: %d",
            chars_limit,
            candidates_limit,
            titles_limit,
            len(short_list),
            code_digits,
        )

        for item in short_list:
            if item["title"] not in a[item["code"][:code_digits]]:
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
                    "Shortening list of candidates to fit token limit from %d to %d",
                    len(soc_candidates),
                    nn,
                )
                soc_candidates = soc_candidates[:nn]

        return "\n".join(soc_candidates)

        def sa_rag_soc_code(
            self,
            industry_descr: str,
            job_title: str = None,
            job_description: str = None,
            expand_search_terms: bool = True,
            code_digits: int = 4,
            candidates_limit: int = 5,
        ) -> SurveyAssistSocResponse:
            """
            Generates a SOC classification based on respondent's data using RAG approach.

            Args:
                industry_descr (str): The description of the industry.
                job_title (str, optional): The job title. Defaults to None.
                job_description (str, optional): The job description. Defaults to None.
                expand_search_terms (bool, optional): Whether to expand the search terms
                    to include job title and description. Defaults to True.
                code_digits (int, optional): The number of digits in the generated
                    SOC code. Defaults to 4.
                candidates_limit (int, optional): The maximum number of SOC code candidates
                    to consider. Defaults to 5.

            Returns:
                SurveyAssistSocResponse: The generated response to the query.

            Raises:
                ValueError: If there is an error during the parsing of the response.
                ValueError: If the default embedding handler is required but
                    not loaded correctly.

            """

            def prep_call_dict(industry_descr, job_title, job_description, soc_codes):
                # Helper function to prepare the call dictionary
                is_job_title_present = job_title is None or job_title in {"", " "}
                job_title = "Unknown" if is_job_title_present else job_title

                is_job_description_present = job_description is None or job_description in {
                    "",
                    " ",
                }
                job_description = (
                    "Unknown" if is_job_description_present else job_description
                )

                call_dict = {
                    "industry_descr": industry_descr,
                    "job_title": job_title,
                    "job_description": job_description,
                    "soc_index": soc_codes,
                }
                return call_dict

            # Retrieve relevant SOC codes and format them for prompt
            if expand_search_terms:
                short_list = self.embed.search_index_multi(
                    query=[industry_descr, job_title, job_description]
                )
            else:
                short_list = self.embed.search_index(query=job_title)

            soc_codes = self._prompt_candidate_list(
                short_list, code_digits=code_digits, candidates_limit=candidates_limit
            )

            call_dict = prep_call_dict(
                industry_descr=industry_descr,
                job_title=job_title,
                job_description=job_description,
                soc_codes=soc_codes,
            )

            if self.verbose:
                final_prompt = self.sa_rag_soc_prompt.format(**call_dict)
                logger.debug(final_prompt)

            chain = LLMChain(llm=self.llm, prompt=self.sa_rag_soc_prompt)

            try:
                response = chain.invoke(call_dict, return_only_outputs=True)
            except ValueError as err:
                logger.exception(err)
                logger.warning("Error from LLMChain, exit early")
                validated_answer = SurveyAssistSocResponse(
                    soc_candidates=[],
                    reasoning="Error from LLMChain, exit early",
                )
                return validated_answer, short_list, call_dict

            if self.verbose:
                logger.debug(f"{response=}")

            # Parse the output to the desired format
            parser = PydanticOutputParser(pydantic_object=SurveyAssistSocResponse)
            try:
                validated_answer = parser.parse(response["text"])
            except ValueError as parse_error:
                logger.exception(parse_error)
                logger.warning(f"Failed to parse response:\n{response['text']}")

                reasoning = (
                    f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
                )
                validated_answer = SurveyAssistSocResponse(
                    soc_candidates=[],
                    reasoning=reasoning,
                )

            return validated_answer, short_list, call_dict