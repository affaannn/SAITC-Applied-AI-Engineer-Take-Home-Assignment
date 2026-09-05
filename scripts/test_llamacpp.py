from app.generation.llamacpp_client import (
    LlamaCppClient,
)


def main():

    client = LlamaCppClient(
        hf_repo=(
            "JackeyLai/"
            "Qwen2.5-3B-Instruct-Q4_K_M-GGUF"
        ),
        hf_file=(
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ),
        tokenizer_name=(
            "Qwen/Qwen2.5-3B-Instruct"
        ),
    )

    result = client.generate(
        system_prompt=(
            "You are a concise assistant."
        ),
        prompt=(
            "Explain RAG in one sentence."
        ),
        temperature=0.1,
        max_tokens=80,
    )

    print()
    print("Response:")
    print(result["text"])

    print()
    print(
        "Prompt tokens:",
        result["prompt_tokens"],
    )

    print(
        "Completion tokens:",
        result["completion_tokens"],
    )

    print(
        "Total tokens:",
        result["total_tokens"],
    )

    print(
        "Latency:",
        round(
            result["latency_sec"],
            2,
        ),
        "sec",
    )


if __name__ == "__main__":
    main()