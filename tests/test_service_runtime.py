from types import SimpleNamespace
from unittest.mock import Mock

from automem.service_runtime import init_openai


def test_init_openai_passes_base_url_when_configured():
    state = SimpleNamespace(openai_client=None)
    logger = Mock()
    openai_cls = Mock()
    openai_client = Mock()
    openai_cls.return_value = openai_client

    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": " https://openrouter.ai/api/v1 ",
    }

    init_openai(
        state=state,
        logger=logger,
        openai_cls=openai_cls,
        get_env_fn=env.get,
    )

    openai_cls.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    assert state.openai_client is openai_client


def test_init_openai_omits_base_url_when_unset():
    state = SimpleNamespace(openai_client=None)
    logger = Mock()
    openai_cls = Mock()
    openai_client = Mock()
    openai_cls.return_value = openai_client

    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "",
    }

    init_openai(
        state=state,
        logger=logger,
        openai_cls=openai_cls,
        get_env_fn=env.get,
    )

    openai_cls.assert_called_once_with(api_key="test-key")
    assert state.openai_client is openai_client
