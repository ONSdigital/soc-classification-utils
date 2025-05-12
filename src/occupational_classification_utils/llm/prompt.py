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
    SOC_PROMPT_PYDANTIC (PromptTemplate): Template for determining SIC codes based on
        respondent data.
    SA_SOC_PROMPT_RAG (PromptTemplate): Template for determining a list of most likely
        SIC codes with confidence scores.
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
    SurveyAssistSocResponse,
)
from occupational_classification_utils.utils.soc_data_access import (
    load_text_from_config,
)

config = get_config()

_core_prompt = """You are a conscientious classification assistant of respondent data
for the use in the UK official statistics. Respondent data may be in English or Welsh,
but you always respond in British English."""

_soc_template = """"Given the respondent data (that may include all or some of
job title, job description, level of education, line management responsibilities,
and company's main activity) your task is to determine
the UK SOC (Standard Occupational Classification) code for this job if it can be
determined. If the code cannot be determined, identify the additional information
needed to determine it. Make sure to use the provided 2020 SOC index.

===Respondent Data===
- Job Title: {job_title}
- Job Description: {job_description}
- Level of Education: {level_of_education}
- Line Management Responsibilities: {manage_others}
- Company's main activity: {industry_descr}

===Output Format===
{format_instructions}

===2020 SOC Index===
{soc_index}
"""

# Load the SOC index from the configuration and convert to file path string
soc_index = load_text_from_config(config["lookups"]["soc_condensed"])

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


_sa_soc_template_rag = """"Given the respondent's description of the main activity their
company does, their job title and job description (which may be different to the
main company activity), your task is to determine a list of the most likely UK SOC
(Standard Occupational Classification) codes for this individual.

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
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Relevant subset of UK SOC 2020===
{soc_index}

===Output Format===
{format_instructions}

===Output===
"""

parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=SurveyAssistSocResponse
)

# Was sic_template_rag
SA_SOC_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _sa_soc_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
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
parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=RagResponse
)

GENERAL_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _general_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)
