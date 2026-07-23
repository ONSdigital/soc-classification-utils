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

from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from occupational_classification.data_access.soc_data_access import load_soc_index

from occupational_classification_utils.embed.embedding import get_config
from occupational_classification_utils.models.response_model import (
    OpenFollowUp,
    RagResponse,
    SocResponse,
    TopOneResponse,
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
on the list you respond with.

Always provide reasoning for your decision.


===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}
- Level of Education: {level_of_education}

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

_soc_template_unambiguous = """"You are an expert in occupational classifications.
You are tasked with determining whether a survey response can be assigned to a
single four-digit UK Standard Occupational Classification (SOC 2020) unit group
based on initial respondent data alone.

Key objective: Determine if the response can be coded unambiguously to a single
four-digit SOC code.

Assignment logic:
1. Code as unambiguous when the response can be coded to a single four-digit SOC
code with 99 per cent confidence based on available evidence.
2. Code as uncodable when multiple candidates are plausible and additional
information is needed to distinguish between them.

===Analysis steps===
Follow these steps in order:
1. Review each candidate from the shortlist of relevant SOC codes against the respondent data.
2. Assess alignment - job role, tasks, industry context, and example activities.
3. Assign confidence scores from 0.1 (least likely) to 0.9 (most likely).
4. Decide if a single four-digit SOC code can be assigned with 99 per cent confidence.
5. Provide reasoning for your decision.

===Respondent Data===
{respondent_data}

===Shortlist===
{soc_candidates}

===Output Format===
{format_instructions}
"""
parser_unambiguous = PydanticOutputParser(
    pydantic_object=UnambiguousResponse  # type: ignore[arg-type]
)

SOC_PROMPT_UNAMBIGUOUS = PromptTemplate.from_template(
    template=_core_prompt + _soc_template_unambiguous,
    partial_variables={
        "format_instructions": parser_unambiguous.get_format_instructions(),
    },
)

_soc_template_top_one_only = """"You are an expert in occupational classifications.
You are tasked with selecting the single most likely four-digit UK Standard
Occupational Classification (SOC 2020) unit group from a shortlist of relevant
candidates.

Key objective:
- Select exactly one four-digit SOC code from the shortlist.
- The selected code must come from the shortlist only.
- Always return the best available match, even when the evidence is imperfect.

Assignment logic:
1. Review each candidate from the shortlist against the respondent data.
2. Assess alignment of the respondent's job role (main tasks, industry context, skill level or seniority) with the candidate SOC code details.
3. Focus on the respondent main job role, not only the employer's main activity.
4. Choose the single four-digit SOC code from the shortlist that best matches the respondent's main job role and context.
5. Derive the likelihood score from two things together:
    - direct fit: how strongly the chosen code matches the respondent data on its own merits
    - separation: how clearly the chosen code outranks the next-best shortlisted alternatives
6. Use only these likelihood values: 0.2, 0.4, 0.6, 0.8, or 0.9.
7. Use this calibration for likelihood scores:
    - 0.9: very strong direct fit and strong separation from all alternatives; full information about respondent's core duties available and aligned closely with SOC code details; no conflict or ambiguity.
    - 0.8: strong direct fit and clear separation; most key evidence is present and aligned; only minor uncertainty remains.
    - 0.6: moderate direct fit or limited separation; plausible best option, but important details are missing or one close rival remains.
    - 0.4: weak direct fit or weak separation; chosen because it is the best available shortlist option, but substantial ambiguity remains.
    - 0.2: very weak direct fit and weak separation; shortlist fit is poor overall, but one option must still be selected.
8. Assign 0.8 or 0.9 only if both direct fit and separation are strong, with no additional information required to resolve ambiguity between the chosen code and the next-best alternative.
9. Use likelihood scores as calibrated strength-of-match within the shortlist only, not as abstract correctness probabilities.
10. Provide reasoning that explains which evidence supports the chosen code, which evidence weakens it, and why the next-best alternatives were ranked lower.

Rules:
- Do not invent codes outside the shortlist.
- Prefer the most specific four-digit unit group supported by the evidence.
- If the shortlist contains close alternatives, still choose the best one and explain the distinguishing evidence or remaining uncertainty.
- Use only the information provided in the respondent data and shortlist.
- Use the same likelihood value whenever the evidence profile and the margin over the next-best candidate are materially similar.

===Respondent Data===
{respondent_data}

===Shortlist===
{soc_candidates}

===Output Format===
{format_instructions}
"""
parser_top_one = PydanticOutputParser(
    pydantic_object=TopOneResponse  # type: ignore[arg-type]
)

SOC_PROMPT_TOP_ONE_ONLY = PromptTemplate.from_template(
    template=_core_prompt + _soc_template_top_one_only,
    partial_variables={
        "format_instructions": parser_top_one.get_format_instructions(),
    },
)

_open_follow_up = """"You are an expert survey methodologist specialising in
UK occupational classification (SOC 2020). Generate one open-ended follow-up question
to help assign the most relevant four-digit SOC unit group.

Objective
- Produce exactly one question that elicits the key information needed to distinguish
between the shortlisted SOC candidates, focusing on the respondent's job and the
employer's context where relevant.

Inputs
- Respondent data:
    {respondent_data}
- Shortlist from previous model: {llm_output}
- Note: These are candidate occupational categories; do not mention codes or "SOC"
to the respondent.

How to decide what to ask
- Identify the smallest, most informative difference among the candidates and target that with a single question.
- Prioritise discriminators in this order:
1) Main tasks and duties (what the respondent actually does day to day, including tools, materials, and who or what they work with).
2) Work setting or sector (for example healthcare, education, construction, agriculture, office, retail premises, outdoor or site-based work).
3) Skill level or seniority (for example supervisory or line-management vs hands-on; professional or technical vs elementary).
4) Who or what they mainly serve or produce for (patients, pupils, customers, machinery, data or systems, goods).
- Ask about only one discriminator—the one most likely to resolve the ambiguity.

Quality standards
- Language and clarity:
    - Use plain British English; avoid or define jargon and abbreviations.
    - Keep the single question concise (max 25 words), grammatically correct, and neutral.
    - Use "employer" for for-profit; use "organisation" for non-profits, charities, public bodies, and education.
        Default to "employer", if ambiguous.
    - Refer to the present situation (e.g., "currently", "main").
    - Do not mention SOC or any code numbers.
    - Do not ask for personal names, client names, or other sensitive data beyond what is needed to describe the job.
- Question structure:
    - Start with "What", "How", "Which", or "Where".
    - Focus on the respondent's job and main activities; use employer context only where it helps distinguish the role.
    - One issue per question; no A/B or either/or phrasing; avoid binary questions.
    - Limit to one sentence ending with a question mark.
    - You may add one additional sentence with broad, non-leading examples covering a wide range of options;
        omit examples if they would be leading.
- Respondent considerations:
    - Make it easy to answer in a few words.
    - Ask only what a typical employee would reasonably know about their own role.
    - Avoid requiring calculations or percentages.

Edge cases
- If the shortlist is empty or clearly points to one category, ask a general clarifying question about
    main tasks or work setting to confirm classification.
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
