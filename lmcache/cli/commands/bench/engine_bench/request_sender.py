# SPDX-License-Identifier: Apache-2.0
"""Async streaming request sender for ``lmcache bench engine``."""

# Standard
from collections.abc import Callable
import codecs
import collections.abc
import json as json_mod
import os
import time

# Third Party
from openai import AsyncOpenAI
import aiohttp

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Callback signature: (result, response_text) -> None
OnFinishedCallback = Callable[[RequestResult, str], None]


def _normalize_url(engine_url: str) -> str:
    """Ensure *engine_url* has a scheme and ends with ``/v1``."""
    url = engine_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _extract_content(chunk: object, completions_mode: bool) -> str:
    """Return text content from a streaming chunk, or ``""`` if none.

    Ported from Tensormesh-Benchmark ``streaming_utils.py``.
    """
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""

    choice = choices[0]

    if completions_mode:
        text = getattr(choice, "text", None)
        return text if text is not None else ""

    # Chat mode: delta.content, with fallback for reasoning_content
    delta = getattr(choice, "delta", None)
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if content is not None:
        return content
    # Fallback for reasoning models
    for attr in ("reasoning_content", "reasoning"):
        fallback = getattr(delta, attr, None)
        if fallback is not None:
            return fallback
    return ""


class RequestSender:
    """Async streaming request sender for inference engines.

    Each ``send_request`` call is a self-contained coroutine.
    Concurrency is controlled externally by the workload module.
    """

    def __init__(
        self,
        engine_url: str,
        model: str,
        completions_mode: bool = False,
        on_finished: list[OnFinishedCallback] = [],  # noqa: B006
        raw_sse: bool = False,
    ) -> None:
        self._model = model
        self._completions_mode = completions_mode
        self._on_finished = list(on_finished)
        self._raw_sse = raw_sse

        base_url = _normalize_url(engine_url)
        self._base_url = base_url
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = "sk-dummy"
            logger.debug("API key source: default dummy key")
        else:
            logger.debug("API key source: OPENAI_API_KEY env var")
        self._api_key = api_key

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=None,
        )
        self._aiohttp_session: aiohttp.ClientSession | None = None

    def add_on_finished_callback(self, callback: OnFinishedCallback) -> None:
        """Register a callback to be invoked when a request finishes."""
        self._on_finished.append(callback)

    async def send_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
    ) -> RequestResult:
        """Send a single streaming request and return the result.

        Streams the response via SSE, measures TTFT, decode speed, and
        total latency.  Extracts token counts from server usage reports.
        After collecting the result, invokes all registered
        ``on_finished`` callbacks.
        """
        submit_time = time.time()
        first_token_time = 0.0
        tokens: list[str] = []
        num_input_tokens = 0
        num_output_tokens = 0

        try:
            if self._raw_sse:
                async for content, usage_data in self._raw_sse_stream(
                    messages,
                    max_tokens,
                ):
                    if usage_data:
                        num_input_tokens = usage_data.get("prompt_tokens", 0) or 0
                        num_output_tokens = usage_data.get("completion_tokens", 0) or 0
                    if content:
                        if not first_token_time:
                            first_token_time = time.time()
                        tokens.append(content)
            else:
                response = await self._create_stream(messages, max_tokens)

                async for chunk in response:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        pt = getattr(usage, "prompt_tokens", 0)
                        ct = getattr(usage, "completion_tokens", 0)
                        if pt:
                            num_input_tokens = pt
                        if ct:
                            num_output_tokens = ct

                    content = _extract_content(chunk, self._completions_mode)
                    if content:
                        if not first_token_time:
                            first_token_time = time.time()
                        tokens.append(content)

            finish_time = time.time()
            successful = first_token_time > 0.0
            ttft = (first_token_time - submit_time) if successful else -1.0
            request_latency = finish_time - submit_time
            decode_time = (finish_time - first_token_time) if successful else 0.0
            num_output = num_output_tokens if num_output_tokens > 0 else len(tokens)
            decode_speed = (num_output / decode_time) if decode_time > 0 else 0.0

            result = RequestResult(
                request_id=request_id,
                successful=successful,
                ttft=ttft,
                request_latency=request_latency,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output,
                decode_speed=decode_speed,
                inter_token_latency=0.0,
                submit_time=submit_time,
                first_token_time=first_token_time,
                finish_time=finish_time,
                error="",
            )
            response_text = "".join(tokens)

        except Exception as e:
            finish_time = time.time()
            result = RequestResult(
                request_id=request_id,
                successful=False,
                ttft=-1.0,
                request_latency=finish_time - submit_time,
                num_input_tokens=0,
                num_output_tokens=0,
                decode_speed=0.0,
                inter_token_latency=0.0,
                submit_time=submit_time,
                first_token_time=0.0,
                finish_time=finish_time,
                error=str(e),
            )
            response_text = ""
            logger.debug(
                "Request %s failed: %s",
                request_id,
                e,
            )

        for cb in self._on_finished:
            cb(result, response_text)

        return result

    async def send_warmup_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1,
    ) -> RequestResult:
        """Send a warmup request (``max_tokens=1`` by default)."""
        return await self.send_request(
            request_id,
            messages,
            max_tokens=max_tokens,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the aiohttp session."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None),
            )
        return self._aiohttp_session

    async def _raw_sse_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> collections.abc.AsyncIterator[tuple[str, dict | None]]:
        """Stream via raw HTTP, handling mixed SSE formats.

        Yields ``(content, usage_dict)`` tuples. Handles both
        ``chat.completion.chunk`` and ``text_completion`` formats
        in the same stream, as produced by production routers.
        """
        session = await self._get_aiohttp_session()
        if self._completions_mode:
            url = f"{self._base_url}/completions"
        else:
            url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._completions_mode:
            payload["prompt"] = messages[0]["content"] if messages else ""
        else:
            payload["messages"] = messages

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                raise RuntimeError(f"raw_sse: HTTP {resp.status}: {error_body[:200]}")
            _decoder = codecs.getincrementaldecoder("utf-8")("replace")
            buffer = ""
            async for raw_chunk in resp.content:
                buffer += _decoder.decode(raw_chunk)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("{"):
                        try:
                            data = json_mod.loads(line)
                            content, usage = self._extract_from_json(data)
                            if content or usage:
                                yield content, usage
                        except json_mod.JSONDecodeError:
                            pass
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return

                    try:
                        data = json_mod.loads(data_str)
                    except json_mod.JSONDecodeError:
                        continue

                    content, usage = self._extract_from_json(data)
                    if content or usage:
                        yield content, usage

            if buffer.strip():
                line = buffer.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str != "[DONE]":
                        try:
                            data = json_mod.loads(data_str)
                            content, usage = self._extract_from_json(data)
                            if content or usage:
                                yield content, usage
                        except json_mod.JSONDecodeError:
                            pass

    @staticmethod
    def _extract_from_json(
        data: dict,
    ) -> tuple[str, dict | None]:
        """Extract content and usage from a JSON chunk.

        Handles both chat completion (delta.content) and text completion
        (choices[0].text) formats.
        """
        usage = data.get("usage")
        choices = data.get("choices", [])
        content = ""

        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if delta and isinstance(delta, dict):
                content = delta.get("content", "") or ""
            if not content:
                content = choice.get("text", "") or ""

        return content, usage if usage else None

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> collections.abc.AsyncIterator:
        """Dispatch the streaming API call (chat or completions)."""
        if self._completions_mode:
            prompt = messages[0]["content"] if messages else ""
            return await self._client.completions.create(
                model=self._model,
                prompt=prompt,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.0,
                stream_options={"include_usage": True},
            )
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.0,
            stream_options={"include_usage": True},
        )
