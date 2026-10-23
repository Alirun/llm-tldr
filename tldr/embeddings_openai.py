"""
OpenAI embeddings backend for semantic search.

An alternative to the local sentence-transformers model: text is sent to the
OpenAI embeddings API over HTTPS (plain ``requests``, no SDK), so semantic
search works without torch / transformers / sentence-transformers installed.

Selected with ``TLDR_EMBEDDING_PROVIDER=openai`` (or ``--provider openai`` on
``tldr semantic index``). Note: the text of every indexed code unit and every
search query is sent to OpenAI.

Configuration (environment):
    OPENAI_API_KEY                   API key
    TLDR_OPENAI_API_KEY_FILE         path to a file containing the API key
                                     (takes precedence over OPENAI_API_KEY)
    TLDR_OPENAI_EMBEDDING_MODEL      model (default: text-embedding-3-small)
    TLDR_OPENAI_EMBEDDING_DIMENSIONS optional output size (text-embedding-3-*)

The API key is never logged and never included in error messages.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("tldr.semantic.openai")

PROVIDER_NAME = "openai"

API_URL = "https://api.openai.com/v1/embeddings"

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Default output dimensions per model (https://platform.openai.com/docs/guides/embeddings)
OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# API limits: 8192 tokens per input, 2048 inputs and 300k tokens per request.
# Stay slightly below them.
MAX_INPUT_TOKENS = 8000
MAX_BATCH_INPUTS = 256
MAX_BATCH_TOKENS = 250_000

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_RETRIES = 6
REQUEST_TIMEOUT = 60

_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*.]+")

_encoder = None
_encoder_loaded = False


class OpenAIEmbeddingError(RuntimeError):
    """Raised when OpenAI embeddings cannot be computed."""


def _redact(message: str, api_key: Optional[str] = None) -> str:
    """Remove anything that looks like an API key from a message."""
    if api_key:
        message = message.replace(api_key, "***")
    return _KEY_PATTERN.sub("sk-***", message)


def resolve_api_key() -> str:
    """Return the API key from TLDR_OPENAI_API_KEY_FILE or OPENAI_API_KEY."""
    key_file = os.environ.get("TLDR_OPENAI_API_KEY_FILE")
    if key_file:
        path = Path(key_file).expanduser()
        try:
            key = path.read_text().strip()
        except OSError as e:
            raise OpenAIEmbeddingError(
                f"Cannot read TLDR_OPENAI_API_KEY_FILE ({path}): {e.strerror}"
            ) from None
        if not key:
            raise OpenAIEmbeddingError(f"TLDR_OPENAI_API_KEY_FILE ({path}) is empty")
        return key

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise OpenAIEmbeddingError(
            "No OpenAI API key: set OPENAI_API_KEY or TLDR_OPENAI_API_KEY_FILE"
        )
    return key


def resolve_model(model: Optional[str] = None) -> str:
    """Return the model name: explicit argument, environment, or default."""
    return model or os.environ.get("TLDR_OPENAI_EMBEDDING_MODEL") or DEFAULT_OPENAI_MODEL


def resolve_dimensions(dimensions: Optional[int] = None) -> Optional[int]:
    """Return requested output dimensions, or None for the model default."""
    if dimensions is not None:
        return dimensions
    value = os.environ.get("TLDR_OPENAI_EMBEDDING_DIMENSIONS")
    if not value:
        return None
    try:
        dims = int(value)
    except ValueError:
        raise OpenAIEmbeddingError(
            f"TLDR_OPENAI_EMBEDDING_DIMENSIONS must be an integer, got {value!r}"
        ) from None
    if dims <= 0:
        raise OpenAIEmbeddingError("TLDR_OPENAI_EMBEDDING_DIMENSIONS must be positive")
    return dims


def _get_encoder():
    """Return the cl100k_base tokenizer, or None if it cannot be loaded."""
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # tokenizer data may be unavailable offline
            logger.debug(f"tiktoken unavailable, using byte-length estimate: {e}")
            _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens; without a tokenizer, the UTF-8 length is an upper bound."""
    encoder = _get_encoder()
    if encoder is None:
        return len(text.encode("utf-8"))
    return len(encoder.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Truncate text to at most max_tokens tokens."""
    encoder = _get_encoder()
    if encoder is None:
        data = text.encode("utf-8")
        if len(data) <= max_tokens:
            return text
        return data[:max_tokens].decode("utf-8", errors="ignore")
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


class OpenAIEmbedder:
    """Compute embeddings with the OpenAI embeddings API.

    Attributes after use:
        total_tokens: tokens billed (from the API usage field)
        requests: number of successful API requests
        truncated: number of inputs truncated to MAX_INPUT_TOKENS
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        session=None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = MAX_RETRIES,
        batch_inputs: int = MAX_BATCH_INPUTS,
        batch_tokens: int = MAX_BATCH_TOKENS,
    ):
        self.model = resolve_model(model)
        self.dimensions = resolve_dimensions(dimensions)
        self._api_key = api_key or resolve_api_key()
        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._sleep = sleep
        self.max_retries = max_retries
        self.batch_inputs = batch_inputs
        self.batch_tokens = batch_tokens
        self.total_tokens = 0
        self.requests = 0
        self.truncated = 0

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self.model!r}, dimensions={self.dimensions!r})"

    def _prepare(self, texts: List[str]) -> List[tuple]:
        """Return (text, token_count) pairs, truncated to the input limit."""
        prepared = []
        for text in texts:
            text = text if text.strip() else " "  # API rejects empty input
            tokens = count_tokens(text)
            if tokens > MAX_INPUT_TOKENS:
                text = truncate_to_tokens(text, MAX_INPUT_TOKENS)
                tokens = min(count_tokens(text), MAX_INPUT_TOKENS)
                self.truncated += 1
            prepared.append((text, tokens))
        return prepared

    def _batches(self, prepared: List[tuple]):
        batch: List[str] = []
        batch_tokens = 0
        for text, tokens in prepared:
            if batch and (
                len(batch) >= self.batch_inputs
                or batch_tokens + tokens > self.batch_tokens
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            yield batch

    def _post(self, inputs: List[str]) -> dict:
        import requests

        payload = {"model": self.model, "input": inputs, "encoding_format": "float"}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while True:
            try:
                response = self._session.post(
                    API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
                )
            except requests.RequestException as e:
                if attempt >= self.max_retries:
                    raise OpenAIEmbeddingError(
                        _redact(f"OpenAI request failed: {type(e).__name__}", self._api_key)
                    ) from None
                delay = self._backoff(attempt, None)
            else:
                status = response.status_code
                if status == 200:
                    return response.json()
                if status not in RETRY_STATUS or attempt >= self.max_retries:
                    raise OpenAIEmbeddingError(
                        _redact(
                            f"OpenAI embeddings request failed: HTTP {status}: "
                            f"{self._error_message(response)}",
                            self._api_key,
                        )
                    )
                delay = self._backoff(attempt, response.headers.get("retry-after"))
            attempt += 1
            logger.debug(f"OpenAI embeddings: retry {attempt} in {delay:.1f}s")
            self._sleep(delay)

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
        return min(2.0 ** attempt, 60.0)

    @staticmethod
    def _error_message(response) -> str:
        try:
            body = response.json()
            message = body.get("error", {}).get("message")
            if message:
                return str(message)[:500]
        except Exception:
            pass
        return (getattr(response, "text", "") or "")[:200]

    def embed(self, texts: List[str]):
        """Embed texts; returns an L2-normalized float32 matrix (len(texts), dim)."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimensions or OPENAI_MODEL_DIMENSIONS.get(self.model, 0)), dtype=np.float32)

        vectors = []
        for batch in self._batches(self._prepare(texts)):
            body = self._post(batch)
            data = sorted(body.get("data", []), key=lambda item: item["index"])
            if len(data) != len(batch):
                raise OpenAIEmbeddingError(
                    f"OpenAI returned {len(data)} embeddings for {len(batch)} inputs"
                )
            vectors.extend(item["embedding"] for item in data)
            self.total_tokens += int(body.get("usage", {}).get("prompt_tokens", 0))
            self.requests += 1

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise OpenAIEmbeddingError("OpenAI returned embeddings of inconsistent size")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
