"""This module defines response models for occupational classification utilities.

The models are implemented using Pydantic's `BaseModel` and are used to represent
various response structures for SOC (Standard Occupational Classification) code
assignment and classification tasks. These models include validation logic and
field-level constraints to ensure data integrity.

Classes:
    SocCandidate: Represents a candidate SOC code with associated information.
    SocResponse: Represents a response model for SOC code assignment.
    RagCandidate: Represents a candidate classification code with associated information.
    RagResponse: Represents a response model for classification code assignment.
    SurveyAssistSocResponse: Represents a response model for Survey Assist
                             classification code assignment.

Constants:
    MAX_ALT_CANDIDATES: Maximum number of alternative candidates allowed in certain models.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

from occupational_classification_utils.utils.constants import MAX_ALT_CANDIDATES


class SocCandidate(BaseModel):
    """Represents a candidate SOC code based on provided job title and description.

    Attributes:
        soc_code (str): Plausible SOC code based on the provided job title and
            description.
        soc_descriptive (str): Descriptive label of the SOC category associated
            with soc_code.
        likelihood (float): Likelihood of this soc_code with a value between 0 and 1.
    """

    soc_code: str = Field(
        description="Plausible SOC code based on provided job title and description."
    )
    soc_descriptive: str = Field(
        description="Descriptive label of the SOC category associated with soc_code."
    )
    likelihood: float = Field(
        description="Likelihood of this soc_code with value between 0 and 1."
    )


class SocResponse(BaseModel):
    """Represents a response model for SOC code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide SOC code,
            False otherwise.
        followup (Optional[str]): Question to ask the user in order to collect
            additional  information to enable reliable SOC assignment.
            Empty if codable=True.
        soc_code (Optional[str]): Full four-digit SOC code assigned based on provided
            job title, description, etc. Empty if codable=False.
        soc_descriptive (Optional[str]): Descriptive label of the SOC category
            associated with soc_code if provided. Empty if codable=False.
        soc_candidates (List[SocCandidate]): List of possible or alternative SOC
            codes that may be applicable with their descriptive label and estimated
            likelihood.
        soc_code_2digits (Optional[str]): First two digits of the hierarchical SOC code
            assigned. This field should be non-empty if the larger (two-digit) group of
            SOC codes can be determined even in cases where additional information is
            needed to code to four digits (for example when all SOC candidates share
            the same first two digits).
        reasoning (str): Step by step reasoning behind classification selected.
            Specifies the information used to assign the SOC code or any additional
            information required to assign a SOC code.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        SOC code, False otherwise.""",
        default=False,
    )
    followup: str | None = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable SOC assignment. Empty if codable=True.""",
        default=None,
    )
    soc_code: str | None = Field(
        description="""Full four digit SOC code assigned based on provided job title,
        description, etc. Empty if codable=False.""",
        default=None,
    )
    soc_descriptive: str | None = Field(
        description="""Descriptive label of the SOC category associated with soc_code
        if provided. Empty if codable=False.""",
        default=None,
    )
    soc_candidates: list[SocCandidate] = Field(
        description="""List of possible or alternative SOC codes that may be applicable
        with their descriptive label and estimated likelihood.""",
        default_factory=list,
    )
    soc_code_2digits: str | None = Field(
        description="""First two digits of the hierarchical SOC code assigned.
        This field should be non empty if the larger (two-digit) group of SOC codes
        can be determined even in cases where additional information is needed to
        to code to four digits (for example when all SOC candidates share
        the same first two digits).""",
        default=None,
    )
    reasoning: str = Field(
        description="""Step by step reasoning behind classification selected. Specifies
            the information used to assign the SOC code or any additional information
            required to assign a SOC code.""",
        default="No reasoning provided.",
    )

    @classmethod
    def soc_code_validator(cls, v):
        """Validates that a valid SOC code is provided if the response is codable.

        Args:
            v (str): The SOC code to validate.

        Returns:
            str: The validated SOC code.

        Raises:
            ValueError: If the SOC code is empty when codable is True.
        """
        if v == "":
            raise ValueError("If codable, then valid soc_code needs to be provided")
        return v

    @model_validator(mode="before")
    @classmethod
    def check_valid_fields(cls, values):
        """Validates the fields of the model before instantiation.

        Ensures that:
        - If `codable` is True, a valid `soc_code` is provided.
        - If `codable` is False, a follow-up question is provided.

        Args:
            values (dict): The dictionary of field values.

        Returns:
            dict: The validated field values.

        Raises:
            ValueError: If validation conditions are not met.
        """
        if values.get("codable"):
            cls.soc_code_validator(values.get("soc_code"))
        elif not values.get("followup"):  # This checks for None or empty string
            raise ValueError("If uncodable, a follow-up question needs to be provided.")
        return values


class RagCandidate(BaseModel):
    """Represents a candidate classification code with associated information.

    Attributes:
        class_code (str): Plausible classification code based on the respondent's data.
        class_descriptive (str): Descriptive label of the classification category
            associated with class_code.
        likelihood (float): Likelihood of this class_code with a value between 0 and 1.

    """

    class_code: str = Field(
        description="Plausible classification code based on the respondent's data."
    )
    class_descriptive: str = Field(
        description="""Descriptive label of the classification category
        associated with class_code."""
    )
    likelihood: float = Field(
        description="Likelihood of this class_code with value between 0 and 1."
    )


class RagResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide
            classification code, False otherwise.
        followup (Optional[str]): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
            Empty if codable=True.
        class_code (Optional[str]): Full classification code (to the required
            number of digits) assigned based on provided respondent's data.
            Empty if codable=False.
        class_descriptive (Optional[str]): Descriptive label of the classification
            category associated with class_code if provided.
            Empty if codable=False.
        alt_candidates (list[RagCandidate]): Short list of less than ten possible
            or alternative classification codes that may be applicable with their
            descriptive label and estimated likelihood.
        reasoning (str): Step by step reasoning behind the classification selected.
            Specifies the information used to assign the SOC code or any additional
            information required to assign a SOC code.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        classification code, False otherwise."""
    )
    followup: str | None = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment. Empty if codable=True.""",
        default=None,
    )
    class_code: str | None = Field(
        description="""Full classification code (to the required number of digits)
        assigned based on provided respondent's data. Empty if codable=False.""",
        default=None,
    )
    class_descriptive: str | None = Field(
        description="""Descriptive label of the classification category associated
        with class_code if provided. Empty if codable=False.""",
        default=None,
    )
    alt_candidates: list[RagCandidate] = Field(
        description="""Short list of less than ten possible or alternative
        classification codes that may be applicable with their descriptive label
        and estimated likelihood."""
    )
    reasoning: str = Field(
        description="""Step by step reasoning behind classification selected. Specifies
            the information used to assign the SOC code or any additional information
            required to assign a SOC code."""
    )


class SurveyAssistSocResponse(BaseModel):
    """Represents a response model for Survey Assist classification SOC code assignment.

    Attributes:
        followup (str): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
        soc_code (str): Full classification code (to the required
            number of digits) assigned based on provided respondent's data.
            This is the most likely coding.
        soc_descriptive (str): Descriptive label of the classification
            category associated with class_code if provided.
            This is the most likely coding.
        soc_candidates (list[RagCandidate]): Short list of less than ten possible
            or alternative classification codes that may be applicable with their
            descriptive label and estimated likelihood.
        reasoning (str): Step by step reasoning for the most likely classification
            selected.
            Specifies the information used to assign the SOC code or any additional
            information required to assign a SOC code.
    """

    followup: str | None = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
        default="",
    )
    soc_code: str | None = Field(
        description="""Full classification code (to the required number of digits)
        of the most likely canddate assigned based on provided respondent's data.""",
        default="",
    )
    soc_descriptive: str | None = Field(
        description="""Descriptive label of the most likely classification category
        associated with soc_code.""",
        default="",
    )
    soc_candidates: list[SocCandidate] | None = Field(
        description="""Short list of less than ten possible or alternative SOC codes
        that may be applicable with their descriptive label and estimated likelihood."""
    )
    reasoning: str | None = Field(
        description="""Step by step reasoning behind the most likely classification
        selected. Specifies the information used to assign the SOC code or any
        additional information required to assign a SOC code.""",
    )


class TopOneResponse(BaseModel):
    """Top-ranked SOC code selected from a supplied shortlist."""

    soc_code: str = Field(
        description="Selected four-digit SOC code from the provided shortlist.",
        min_length=1,
    )
    soc_title: str = Field(
        description="Title label associated with the selected SOC code.",
        min_length=1,
    )
    likelihood: float = Field(
        description=(
            "Likelihood of the selected SOC code relative to the other shortlisted "
            "candidates, between 0 and 1."
        ),
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description=(
            "Reasoning explaining why the selected SOC code is the strongest "
            "match from the shortlist and why the likelihood is as reported."
        ),
        min_length=1,
    )


class UnambiguousResponse(BaseModel):
    """Represents a response model for classification code assignment (two-step SOC).

    Same generic field names as SIC ``UnambiguousResponse`` for parity across schemes.
    """

    codable: bool = Field(
        description=(
            "True only if enough information is provided to decide an unambiguous "
            "classification code, False otherwise."
        )
    )
    class_code: str | None = Field(
        default=None,
        description=(
            "Full classification code assigned from respondent data. "
            "Present if codable=True, None if codable=False."
        ),
    )
    class_descriptive: str | None = Field(
        default=None,
        description=(
            "Descriptive label for class_code. Present if codable=True, "
            "None if codable=False."
        ),
    )
    alt_candidates: list[RagCandidate] = Field(
        default_factory=list,
        description="Short list of possible classification codes with likelihoods.",
        min_length=1,
        max_length=10,
    )
    reasoning: str = Field(
        description="Step by step reasoning behind the classification selected.",
        min_length=50,
    )

    @field_validator("alt_candidates")
    @classmethod
    def validate_alt_candidates(cls, v: list[RagCandidate]) -> list[RagCandidate]:
        """Validate alternative candidate count."""
        if not 1 <= len(v) <= MAX_ALT_CANDIDATES:
            raise ValueError("alt_candidates must contain between 1 and 10 items.")
        return v


class OpenFollowUp(BaseModel):
    """Open-ended follow-up question when SOC cannot be assigned unambiguously."""

    followup: str | None = Field(
        description=(
            "Question to collect additional information for reliable SOC assignment."
        ),
        default="",
    )
    reasoning: str = Field(
        description="Reasoning explaining how the follow-up question helps classification.",
        default="",
    )
