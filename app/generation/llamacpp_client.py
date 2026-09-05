import time
import requests

from transformers import AutoTokenizer


class LlamaCppClient:
    """
    Client for a locally running llama.cpp HTTP server.

    The model is loaded once by llama-server instead of being
    reloaded for every query.
    """

    def __init__(
        self,
        tokenizer_name: str,
        base_url: str = "http://127.0.0.1:8080",
        timeout: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> dict:

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        start = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to generate response from llama.cpp server. "
                "Make sure llama-server is running on "
                f"{self.base_url}."
            ) from exc

        latency = time.perf_counter() - start

        data = response.json()

        try:
            output = (
                data["choices"][0]["message"]["content"]
                .strip()
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as exc:

            raise RuntimeError(
                f"Unexpected llama.cpp response: {data}"
            ) from exc

        if not output:
            raise RuntimeError(
                "llama.cpp returned an empty response."
            )

        prompt_tokens = self._count_tokens(
            system_prompt + "\n\n" + prompt
        )

        completion_tokens = self._count_tokens(
            output
        )

        return {
            "text": output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens
                + completion_tokens
            ),
            "latency_sec": latency,
        }