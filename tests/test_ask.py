from types import SimpleNamespace

import app.ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        text="テスト回答",
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    result = ask_module.ask("テスト質問")
    output = capsys.readouterr().out

    assert result == "テスト回答"
    assert "入力トークン数: 12" in output
    assert "出力トークン数: 8" in output


def test_ask_sends_and_updates_conversation_history(monkeypatch):
    assistant_reply = ask_module.types.Content(
        role="model", parts=[ask_module.types.Part.from_text(text="前の回答")]
    )
    responses = [
        SimpleNamespace(usage_metadata=None, text="前の回答", candidates=[SimpleNamespace(content=assistant_reply)]),
        SimpleNamespace(
            usage_metadata=None,
            text="続きの回答",
            candidates=[SimpleNamespace(content=assistant_reply)],
        ),
    ]
    requests = []

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return responses.pop(0)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    history = []

    assert ask_module.ask("最初の質問", history) == "前の回答"
    assert ask_module.ask("続きの質問", history) == "続きの回答"

    assert [content.role for content in requests[1]["contents"]] == [
        "user",
        "model",
        "user",
    ]
    assert [content.role for content in history] == ["user", "model", "user", "model"]