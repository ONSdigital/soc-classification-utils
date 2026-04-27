"""Module for generating prompt templates for SOC classification tasks.

This module provides various prompt templates for tasks related to the classification
of respondent data into UK SOC (Standard Occupational Classification) codes. The prompts
are designed to work with the LangChain library and include configurations for
different use cases, such as determining SOC codes, re-ranking SOC codes, and handling
ambiguous classifications.

The module includes:
- Core prompt templates for SOC classification tasks.
- Support for partial variables and format instructions.
- Integration with Pydantic models for structured output parsing.

Attributes:
    SOC_PROMPT_PYDANTIC (PromptTemplate): Template for determining SOC codes based on
        respondent data.
    SA_SOC_PROMPT_RAG (PromptTemplate): Template for determining a list of most likely
        SOC codes with confidence scores.
    GENERAL_PROMPT_RAG (PromptTemplate): Template for determining custom classification
        codes with a relevant subset of codes provided.
"""

# pylint: disable=invalid-name # Need to clean up the code to remove this

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from occupational_classification_utils.embed.embedding import get_config
from occupational_classification_utils.models.response_model import (
    RagResponse,
    SocResponse,
)
from occupational_classification_utils.utils.soc_data_access import (
    load_soc_index,
)

config = get_config()

_core_prompt = """You are a conscientious classification assistant of respondent data
for the use in the UK official statistics. Respondent data may be in English or Welsh,
but you always respond in British English."""

_soc_template = """"Given the respondent job title your task is to determine
the UK SOC (Standard Occupational Classification) code for this job.
If the code cannot be determined, identify the additional information
needed to determine it. Make sure to use the provided 2020 SOC index.

===Respondent Data===
- Job Title: {job_title}


===Output Format===
{format_instructions}

===Relevant subset of UK SOC 2020===
{soc_index}
"""

# Load the full SOC index from the configuration (mirror SIC: full index into one-shot prompt)
soc_index = load_soc_index(config["lookups"]["soc_index"])

parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=SocResponse
)

SOC_PROMPT_PYDANTIC = PromptTemplate.from_template(
    template=_core_prompt + _soc_template,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "soc_index": soc_index,
    },
)


_sa_soc_template_rag = """"Given the respondent's job title, your task is to determine
a list of the most likely UK SOC (Standard Occupational Classification) codes for this individual.

The following will be provided to make your decision and send appropriate output:
Respondent Data
Relevant subset of UK SOC 2020 (you must only use this list to classify)
Output Format (the output format MUST be valid JSON)

Only use the subset of UK SOC 2020 provided to determine if you can match the most
likely soc codes, provide a confidence score between 0 and 1 where 0.1 is least
likely and 0.9 is most likely.

You must return the subset list of possible soc codes (UK SOC 2020 codes provided)
that might match with a confidence score for each.

You must provide a follow up question that would help identify the exact coding based
on the list you respond with.

===Respondent Data===
- Job Title: {job_title}

===Relevant subset of UK SOC 2020===
{soc_index}

===Output Format===
{format_instructions}

===Output===
"""

parser = PydanticOutputParser(
    pydantic_object=SocResponse  # type: ignore # Suspect langchain ver bug
)

SA_SOC_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _sa_soc_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "soc_index": soc_index,
    },
)


_general_template_rag = """"Given the respondent's data, your task is to determine
the classification code. Make sure to use the provided Relevant subset of
classification index and select codes from this list only.
If the code cannot be determined (or not included in the provided subset),
do not provide final code, instead identify the additional information needed
to determine the correct code and suggest few most likely codes.

===Respondent Data===
{respondent_data}

===Relevant subset of classification index===
{classification_index}

===Output Format===
{format_instructions}

===Output===
"""
parser = PydanticOutputParser(
    pydantic_object=RagResponse  # type: ignore # Suspect langchain ver bug
)

GENERAL_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _general_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

FIX_PARSING_PROMPT = PromptTemplate.from_template(
    """You are a meticulous assistant tasked with ensuring that
the output from a language model adheres strictly to the required JSON format.

Your task is to review the output and make any necessary adjustments to ensure it is valid JSON.
If the output is not valid JSON, you must fix it without altering the intended meaning.

====Output from LLM====
{llm_output}

===Output Format===
{format_instructions}
"""
)


CORRECT_SPELLING_PROMPT = PromptTemplate.from_template(
    """Correct spelling of this job title: {job_title}

    Accepted abbreviations and their meaning in a dictionary format: {abbreviations}
    Do not replace abbreviations. Use the dictionary only to correct the misspelled job titles.
    If the original job title contains an abbreviation, check if the meaning is correct,
    but do not replace the abbreviation.

    If there is no need to correct the spelling, return the original job title.
    """
)
