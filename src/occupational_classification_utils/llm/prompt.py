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
from occupational_classification.data_access.soc_data_access import load_soc_index

from occupational_classification_utils.embed.embedding import get_config
from occupational_classification_utils.models.response_model import (
    OpenFollowUp,
    RagResponse,
    SocResponse,
    UnambiguousResponse,
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
on the list you respond with when the coding is ambiguous.
When one SOC code is clearly the most likely match, return that final SOC code and
leave followup empty.

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

parser = PydanticOutputParser(
    pydantic_object=SocResponse  # type: ignore # Suspect langchain ver bug
)

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
parser = PydanticOutputParser(
    pydantic_object=RagResponse  # type: ignore # Suspect langchain ver bug
)

GENERAL_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _general_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)


_soc_template_unambiguous = """"You are an expert in ocucpational classifications.
You are tasked with determining whether a survey response can be assigned to a
single 4-digit UK Standard Occupational Classification (SOC) code based on initial respondent data alone.

Key objective:  Determine if the response can be coded unambiguously to a single 4-digit SOC code.

Assignment logic:
1. Code as unambiguous when response can be coded to a single 4-digit SOC code with 99
per cent confidence based on available evidence.
2. Code as uncodable to 4-digit when multiple candidates are plausible and
additional information is needed to distinguish between them.

===Analysis steps===
Follow these steps in order:
1. Review each candidate from the shortlist of relevant SOC codes against the respondent data.
2. Assess alignment - Consider:
   - Semantic similarity between respondent descriptions and SOC code descriptions
   - Job role compatibility with typical activities in each SOC code
   - Industry context alignment
   - Matches with specific examples listed under each code.
3. Assign confidence scores - Rate each candidate from 0.1 (least likely) to 0.9 (most likely).
4. Decide if response can be codeded unambiguously to a single 4-digit SOC code with 99 per cent confidence.
5. Provide reasoning for your decision.

===Respondent Data===
- Industry description: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}
- Level of Education: {level_of_education}

===Shortlist===
{soc_candidates}

===Output Format===
{format_instructions}
"""

parser_unambiguous = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=UnambiguousResponse
)

SOC_PROMPT_UNAMBIGUOUS = PromptTemplate.from_template(
    template=_core_prompt + _soc_template_unambiguous,
    partial_variables={
        "format_instructions": parser_unambiguous.get_format_instructions(),
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


_open_follow_up = """"You are an expert survey methodologist specialising in
    UK industrial classification (UK SOC). Generate one open-ended follow-up question
    to help assign the most relevant UK SOC code.

Objective
- Produce exactly one question that elicits the key information needed to distinguish
    between the shortlisted SOC candidates, focusing on the employer's main business activity.

Inputs
- Respondent data:
- Company's main activity: {industry_descr}
- Job title: {job_title}
- Job description: {job_description}
- Level of Education: {level_of_education}
- Shortlist from previous model: {llm_output}
- Note: These are candidate SOC categories; do not mention codes or "SOC" to the respondent.

How to decide what to ask
- Identify the smallest, most informative difference among the candidates and target that with a single question.
- Prioritise discriminators in this order:
1) Stage in the value chain (e.g., manufacture/processing vs wholesale vs retail vs repair/installation vs
    rental/leasing vs publishing/software vs consultancy/training).
2) Main product or service category (what goods/services the employer mainly provides).
3) Main customer type (households vs businesses vs government/health/education).
4) Delivery mode or setting (on-site vs online; physical goods vs digital; own-brand vs third-party).
- Ask about only one discriminator—the one most likely to resolve the ambiguity.

Quality standards
- Language and clarity:
    - Use plain British English; avoid or define jargon and abbreviations.
    - Keep the single question concise (max 25 words), grammatically correct, and neutral.
    - Use "employer" for for-profit; use "organisation" for non-profits, charities, public bodies, and education.
        Default to "employer", if ambiguous.
    - Refer to the present situation (e.g., "currently", "main").
    - Do not mention SOC or any code numbers.
    - Do not ask for company names, client names, or other personal/sensitive data.
- Question structure:
    - Start with "What", "How", "Which", or "Where".
    - Focus on the employer's main business activities, products, or services—not the respondent's personal tasks.
    - One issue per question; no A/B or either/or phrasing; avoid binary questions.
    - Limit to one sentence ending with a question mark.
    - You may add one additional sentence with broad, non-leading examples covering a wide range of options;
        omit examples if they would be leading.
- Respondent considerations:
    - Make it easy to answer in a few words.
    - Ask only what a typical employee would reasonably know.
    - Avoid requiring calculations or percentages.

Edge cases
- If the shortlist is empty or clearly points to one category, ask a general clarifying question about
    the main product/service or value-chain stage to confirm classification.
- Do not output explanations or reasoning; only the formatted result.

Output format
- Return output that strictly follows:
{format_instructions}
"""
parser_followup_open = PydanticOutputParser(pydantic_object=OpenFollowUp)

SOC_PROMPT_OPENFOLLOWUP = PromptTemplate.from_template(
    template=_core_prompt + _open_follow_up,
    partial_variables={
        "format_instructions": parser_followup_open.get_format_instructions(),
    },
)
