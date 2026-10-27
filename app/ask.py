"""第1回: 素のLLMに会話履歴つきで質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは使わず、同じ実行中の会話履歴を保って質問する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    user_content = types.Content(
        role="user", parts=[types.Part.from_text(text=question)]
    )
    contents = [*(history or []), user_content]
    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] 入力トークン数: {usage.prompt_token_count}, "
            f"出力トークン数: {usage.candidates_token_count}"
        )

    if history is not None:
        history.append(user_content)
        if response.candidates and response.candidates[0].content:
            history.append(response.candidates[0].content)

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    history: list[types.Content] = []
    try:
        print(ask(args.question, history))
        while True:
            try:
                question = input("\n[質問] (終了: /exit): ").strip()
            except EOFError:
                break
            if question.lower() in {"/exit", "exit"}:
                break
            if not question:
                continue
            print(f"\n[質問] {question}\n")
            print(ask(question, history))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
