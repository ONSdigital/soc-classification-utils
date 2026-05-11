# pylint: disable=C0116, W0621
"""Tests for occupational_classification_utils.llm.llm.py."""

import json
from importlib.resources import as_file, files
from unittest import mock

import pytest
import vertexai
from langchain_core.messages import AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from occupational_classification.data_access.soc_data_access import (
    load_soc_index as lib_load_soc_index,
)
from occupational_classification.data_access.soc_data_access import (
    load_soc_structure as lib_load_soc_structure,
)
from occupational_classification.hierarchy.soc_hierarchy import load_hierarchy

from occupational_classification_utils.llm.llm import ClassificationLLM
from occupational_classification_utils.models.response_model import SocResponse

MODEL_NAME = "gemini-2.5-flash"
LOCATION = "europe-west2"


# Mock LLM connections
@pytest.fixture
async def classification_llm_with_soc_sa_rag_soc():
    """ClassificationLLM with mocked ainvoke (async invoke) for sa_rag_soc_code
    (mirrors SIC classification_llm_with_sic_sa_rag_sic).

    Uses unittest.mock so we don't depend on the pytest-mock plugin.
    """
    mock_object_dict = {
        "codable": True,
        "followup": "Example follow-up from the LLM.",
        "soc_code": "2314",
        "soc_descriptive": "Primary education teaching professionals",
        "soc_candidates": [
            {
                "soc_code": "2314",
                "soc_descriptive": "Primary education teaching professionals",
                "likelihood": 0.9,
            },
            {
                "soc_code": "2313",
                "soc_descriptive": "Secondary education teaching professionals",
                "likelihood": 0.1,
            },
        ],
        "reasoning": "Example reasoning for the classification.",
    }
    mock_message = mock.MagicMock(spec=AIMessage)
    mock_message.content = json.dumps(mock_object_dict)
    with mock.patch(
        "occupational_classification_utils.llm.llm.ChatVertexAI.ainvoke",
        return_value=mock_message,
    ):
        llm_class = ClassificationLLM(model_name=MODEL_NAME)
        yield llm_class


# Test initialisation
def test_setup():
    vertexai.init(project="classifai-sandbox", location=LOCATION)


@pytest.fixture(autouse=True)
def mock_vertex_ai():
    with mock.patch(
        "google.cloud.aiplatform.gapic.PredictionServiceClient"
    ) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_content.return_value = mock.Mock()
        yield


@pytest.mark.parametrize(
    "model, openai_api_key, expected_model",
    [
        ("gemini", None, ChatVertexAI),
        ("text-", None, ChatVertexAI),
        ("gpt", "key", ChatOpenAI),
    ],
)
@pytest.mark.llm
def test_llm_model(model, openai_api_key, expected_model):
    llm_model_type = ClassificationLLM(
        model_name=model, openai_api_key=openai_api_key
    ).llm
    assert isinstance(llm_model_type, expected_model)


@pytest.mark.llm
def test_pass_llm_argument():
    llm_model = ClassificationLLM(llm="model").llm
    assert llm_model == "model"


@pytest.mark.llm
def test_llm_model_default():
    assert isinstance(ClassificationLLM().llm, ChatVertexAI)


@pytest.mark.llm
def test_prompt_candidate_strict_hierarchy_lookup_matches_sic_shape(mock_vertex_ai):
    """Prompt line comes from ``self.soc[code]`` (no vector-store title fallback)."""
    _ = mock_vertex_ai
    llm = ClassificationLLM(model_name=MODEL_NAME)
    ref = ("occupational_classification", "data/example_soc_lookup_data.csv")
    with as_file(files(ref[0]).joinpath(ref[1])) as path:
        p = str(path)
        idx = lib_load_soc_index(p)
        llm.soc = load_hierarchy(lib_load_soc_structure(p), idx)
    code = idx["code"].iloc[0]
    out = llm._prompt_candidate(  # pylint: disable=protected-access
        code, ["Example from search"]
    )
    assert code in out
    assert llm.soc[code].group_title in out
    assert "Example from search" in out


@pytest.mark.llm
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.0-pro"


@pytest.mark.llm
async def test_llm_response_mocked_sa_rag_soc_code(
    classification_llm_with_soc_sa_rag_soc,
):
    """Test sa_rag_soc_code with short_list returns (response, list, dict).

    Args:
        classification_llm_with_soc_sa_rag_soc: Fixture providing ClassificationLLM
            with mocked ainvoke (async invoke) for sa_rag_soc_code.

    Asserts:
        First element is SocResponse, second is list, third is dict;
        soc_code on response is as expected.
    """
    short_list = [
        {
            "distance": 0.6,
            "title": "Primary education teaching professionals",
            "code": "2314",
        }
    ]
    result = await classification_llm_with_soc_sa_rag_soc.sa_rag_soc_code(
        industry_descr="school",
        job_title="teacher",
        job_description="teach children",
        short_list=short_list,
    )
    assert isinstance(result[0], SocResponse)
    assert isinstance(result[1], list)
    assert isinstance(result[2], dict)
    assert result[0].soc_code == "2314"


# Tests for rising errors
@pytest.mark.llm
def test_open_api_key_raise_not_implemented_error():
    with pytest.raises(
        NotImplementedError,
        match="Need to provide an OpenAI API key",
    ):
        ClassificationLLM(model_name="gpt")


@pytest.mark.llm
def test_model_family_raise_not_implemented_error():
    with pytest.raises(
        NotImplementedError,
        match="Unsupported model family",
    ):
        ClassificationLLM(model_name="aaaa")


@pytest.mark.llm
async def test_llm_response_mocked_get_soc_code():
    """Test get_soc_code returns a SocResponse with mocked LLM output.

    Mirrors the SIC get_sic_code test but for SOC.
    """
    mock_object_dict = {
        "codable": True,
        "followup": "",
        "soc_code": "2314",
        "soc_descriptive": "Primary education teaching professionals",
        "soc_candidates": [
            {
                "soc_code": "2314",
                "soc_descriptive": "Primary education teaching professionals",
                "likelihood": 0.9,
            },
            {
                "soc_code": "2313",
                "soc_descriptive": "Secondary education teaching professionals",
                "likelihood": 0.1,
            },
        ],
        "soc_code_2digits": "23",
        "reasoning": "Example reasoning for the SOC answer.",
    }
    mock_message = mock.MagicMock(spec=AIMessage)
    mock_message.content = json.dumps(mock_object_dict)

    with mock.patch(
        "occupational_classification_utils.llm.llm.ChatVertexAI.ainvoke",
        return_value=mock_message,
    ):
        result = await ClassificationLLM(model_name=MODEL_NAME).get_soc_code(
            job_title="teacher",
            job_description="teach children",
            level_of_education="degree",
            manage_others=False,
            industry_descr="school",
        )

    assert isinstance(result, SocResponse)
    assert result.soc_code == "2314"


@pytest.mark.parametrize(
    "title, expected_job_title",
    [
        ("", "Unknown"),
        (" ", "Unknown"),
        (None, "Unknown"),
        ("teacher", "teacher"),
    ],
)
@pytest.mark.llm
async def test_sa_rag_soc_code_call_dict_job_title_normalised(
    classification_llm_with_soc_sa_rag_soc,
    title,
    expected_job_title,
):
    """sa_rag_soc_code call dict should normalise empty/None job_title to 'Unknown'.

    Mirrors SIC tests for job-title handling in sa_rag_sic_code.
    """
    short_list = [
        {
            "distance": 0.6,
            "title": "Primary education teaching professionals",
            "code": "2314",
        }
    ]
    (
        _response,
        _short_list,
        call_dict,
    ) = await classification_llm_with_soc_sa_rag_soc.sa_rag_soc_code(
        industry_descr="school",
        job_title=title,
        job_description="teach children",
        short_list=short_list,
    )
    assert call_dict["job_title"] == expected_job_title


@pytest.mark.llm
async def test_sa_rag_soc_code_followup_is_str(
    classification_llm_with_soc_sa_rag_soc,
):
    """sa_rag_soc_code followup field should be a string."""
    short_list = [
        {
            "distance": 0.6,
            "title": "Primary education teaching professionals",
            "code": "2314",
        }
    ]
    (
        response,
        _short_list,
        _call_dict,
    ) = await classification_llm_with_soc_sa_rag_soc.sa_rag_soc_code(
        industry_descr="school",
        job_title="teacher",
        job_description="teach children",
        short_list=short_list,
    )
    assert isinstance(response.followup, str) or response.followup is None


@pytest.mark.llm
async def test_sa_rag_soc_code_short_list_is_none_raise_value_error(
    classification_llm_with_soc_sa_rag_soc,
):
    """sa_rag_soc_code should raise ValueError when short_list is None."""
    with pytest.raises(
        ValueError,
        match=r"Short list is None - list provided from embedding search\.",
    ):
        await classification_llm_with_soc_sa_rag_soc.sa_rag_soc_code(
            industry_descr="school",
            job_title="teacher",
            job_description="teach children",
            short_list=None,
        )
