"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

# Auto load environment variables from .env
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    output_text = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return (output_text, latency, {"input_tokens": input_tokens, "output_tokens": output_tokens})


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    actual_model = "gemini-3.8-flash" if model == "gemini-2.5-flash" else model
    start_time = time.time()
    response = client.models.generate_content(
        model=actual_model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    output_text = response.text or ""
    input_tokens = 0
    output_tokens = 0
    if response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    return (output_text, latency, {"input_tokens": input_tokens, "output_tokens": output_tokens})


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    output_text = response.content[0].text if response.content else ""
    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0

    return (output_text, latency, {"input_tokens": input_tokens, "output_tokens": output_tokens})


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    # 1. Call gpt-4o
    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # 2. Call gpt-4o-mini
    mini_text, mini_latency, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # 3. Call gemini-2.5-flash
    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"]
        + gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_latency,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' or 'exit' to exit) ===")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Exiting chatbot.")
                break

            # Limit history to the last 3 turns (6 messages)
            history = history[-6:]
            history.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            ))

            actual_model = "gemini-3.8-flash" if GEMINI_MODEL == "gemini-2.5-flash" else GEMINI_MODEL
            response_stream = client.models.generate_content_stream(
                model=actual_model,
                contents=history,
            )

            assistant_chunks = []
            print("Gemini: ", end="", flush=True)
            for chunk in response_stream:
                chunk_text = chunk.text or ""
                print(chunk_text, end="", flush=True)
                assistant_chunks.append(chunk_text)
            print("\n")

            history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text="".join(assistant_chunks))]
            ))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot.")
            break


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for prompt in prompts:
        try:
            res = compare_models(prompt)
        except TypeError:
            res = compare_models()
        res_dict = dict(res)
        res_dict["prompt"] = prompt
        results.append(res_dict)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    model_labels = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    ]

    for item in results:
        prompt_str = item.get("prompt", "")
        for key, label in model_labels:
            if key in item:
                data = item[key]
                resp = data.get("response", "").replace("\n", " ")
                if len(resp) > 50:
                    resp = resp[:47] + "..."
                latency_str = f"{data.get('latency', 0.0):.2f}s"
                tokens_str = f"{data.get('input_tokens', 0)}/{data.get('output_tokens', 0)}"
                cost_str = f"${data.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt_str} | {label} | {resp} | {latency_str} | {tokens_str} | {cost_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
