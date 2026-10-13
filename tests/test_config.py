import pytest

from research_agent.config import Settings


def test_missing_credentials_produce_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY.*SERPER_API_KEY"):
        Settings.from_env(require_keys=True)


def test_settings_can_be_loaded_without_keys_for_non_network_tests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    settings = Settings.from_env(require_keys=False)
    assert settings.openai_api_key == ""
    assert settings.serper_api_key == ""
    assert settings.openai_model == "gpt-4o"
