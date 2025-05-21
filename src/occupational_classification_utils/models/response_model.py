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

from typing import Optional

from pydantic import BaseModel, Field, model_validator


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
        SOC code, False otherwise."""
    )
    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable SOC assignment. Empty if codable=True.""",
        default=None,
    )
    soc_code: Optional[str] = Field(
        description="""Full four digit SOC code assigned based on provided job title,
        description, etc. Empty if codable=False.""",
        default=None,
    )
    soc_descriptive: Optional[str] = Field(
        description="""Descriptive label of the SOC category associated with soc_code
        if provided. Empty if codable=False.""",
        default=None,
    )
    soc_candidates: list[SocCandidate] = Field(
        description="""List of possible or alternative SOC codes that may be applicable
        with their descriptive label and estimated likelihood."""
    )
    soc_code_2digits: Optional[str] = Field(
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
            required to assign a SOC code."""
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
    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment. Empty if codable=True.""",
        default=None,
    )
    class_code: Optional[str] = Field(
        description="""Full classification code (to the required number of digits)
        assigned based on provided respondent's data. Empty if codable=False.""",
        default=None,
    )
    class_descriptive: Optional[str] = Field(
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

    followup: str = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
    )
    soc_code: str = Field(
        description="""Full classification code (to the required number of digits)
        of the most likely canddate assigned based on provided respondent's data.""",
    )
    soc_descriptive: str = Field(
        description="""Descriptive label of the most likely classification category
        associated with soc_code.""",
    )
    soc_candidates: list[SocCandidate] = Field(
        description="""Short list of less than ten possible or alternative SOC codes
        that may be applicable with their descriptive label and estimated likelihood."""
    )
    reasoning: str = Field(
        description="""Step by step reasoning behind the most likely classification
        selected. Specifies the information used to assign the SOC code or any
        additional information required to assign a SOC code."""
    )
