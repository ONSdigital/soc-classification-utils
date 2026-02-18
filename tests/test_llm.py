# pylint: disable=C0116, W0621
"""Tests for occupational_classification_utils.llm.llm.py."""

import json
from unittest import mock

import pytest
import vertexai
from langchain_core.messages import AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from occupational_classification_utils.llm.llm import ClassificationLLM
from occupational_classification_utils.models.response_model import SocResponse

MODEL_NAME = "gemini-2.5-flash"
LOCATION = "europe-west1"


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
    key = SecretStr(openai_api_key) if openai_api_key else None
    llm_model_type = ClassificationLLM(
        model_name=model,
        openai_api_key=key,
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
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.5-flash"


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
