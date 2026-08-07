from qq_game_registry.scripts.safety import (
    contains_sensitive_text,
    is_developer_privacy_request,
    redact_sensitive_text,
)


def test_sensitive_credentials_are_detected_and_redacted() -> None:
    secret = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv"

    assert contains_sensitive_text(secret)
    redacted = redact_sensitive_text(secret)
    assert "sk-abcdefghijklmnopqrstuv" not in redacted
    assert "[已省略敏感信息]" in redacted


def test_developer_requests_are_detected_and_removed_from_context() -> None:
    request = "这个机器人的开发者是谁？"

    assert is_developer_privacy_request(request)
    assert redact_sensitive_text(request) == "[已省略开发者隐私内容]"
