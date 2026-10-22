"""
Tests for the OpenAI embeddings backend of semantic search.

The HTTP layer is replaced by a fake session: no network access and no real
API key are needed.

Run with:
    pytest tests/test_openai_embeddings.py -v
"""

import builtins
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from tldr import embeddings_openai
from tldr.embeddings_openai import OpenAIEmbedder, OpenAIEmbeddingError, resolve_api_key

FAKE_KEY = "sk-test-0123456789abcdefSECRET"


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    """Records requests; returns queued responses, else deterministic embeddings."""

    def __init__(self, dim=8, responses=None):
        self.dim = dim
        self.responses = list(responses or [])
        self.calls = []

    @staticmethod
    def vector(text, dim):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return (rng.standard_normal(dim) * 3).tolist()

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            if response is not None:
                return response
        inputs = json["input"]
        dim = json.get("dimensions", self.dim)
        data = [
            {"object": "embedding", "index": i, "embedding": self.vector(t, dim)}
            for i, t in enumerate(inputs)
        ]
        data.reverse()  # the client must order by "index"
        return FakeResponse(200, {"data": data, "usage": {"prompt_tokens": 10 * len(inputs)}})


class WordEncoder:
    """Tokenizer stand-in: one token per whitespace-separated word."""

    def encode(self, text, disallowed_special=()):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "TLDR_OPENAI_API_KEY_FILE",
        "TLDR_OPENAI_EMBEDDING_MODEL",
        "TLDR_OPENAI_EMBEDDING_DIMENSIONS",
        "TLDR_EMBEDDING_PROVIDER",
        "CLAUDE_PROJECT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(embeddings_openai, "_encoder", WordEncoder())
    monkeypatch.setattr(embeddings_openai, "_encoder_loaded", True)


def make_embedder(session, **kwargs):
    kwargs.setdefault("api_key", FAKE_KEY)
    return OpenAIEmbedder(session=session, sleep=lambda s: None, **kwargs)


class TestApiKey:
    def test_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        assert resolve_api_key() == FAKE_KEY

    def test_key_file_takes_precedence(self, monkeypatch, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("sk-from-file\n")
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        monkeypatch.setenv("TLDR_OPENAI_API_KEY_FILE", str(key_file))
        assert resolve_api_key() == "sk-from-file"

    def test_missing_key(self):
        with pytest.raises(OpenAIEmbeddingError, match="OPENAI_API_KEY"):
            resolve_api_key()

    def test_unreadable_key_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TLDR_OPENAI_API_KEY_FILE", str(tmp_path / "missing"))
        with pytest.raises(OpenAIEmbeddingError, match="Cannot read"):
            resolve_api_key()

    def test_empty_key_file(self, monkeypatch, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("\n")
        monkeypatch.setenv("TLDR_OPENAI_API_KEY_FILE", str(key_file))
        with pytest.raises(OpenAIEmbeddingError, match="empty"):
            resolve_api_key()

    def test_key_not_in_repr(self):
        embedder = make_embedder(FakeSession())
        assert FAKE_KEY not in repr(embedder)
        assert FAKE_KEY not in str(vars(embedder).get("model"))

    def test_key_sent_only_in_authorization_header(self):
        session = FakeSession()
        make_embedder(session).embed(["a"])
        call = session.calls[0]
        assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert FAKE_KEY not in json.dumps(call["json"])
        assert FAKE_KEY not in call["url"]


class TestEmbed:
    def test_default_model_and_payload(self):
        session = FakeSession()
        embedder = make_embedder(session)
        embedder.embed(["hello world"])
        payload = session.calls[0]["json"]
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["hello world"]
        assert "dimensions" not in payload

    def test_model_and_dimensions_from_environment(self, monkeypatch):
        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_DIMENSIONS", "256")
        session = FakeSession()
        result = make_embedder(session).embed(["x"])
        assert session.calls[0]["json"]["model"] == "text-embedding-3-large"
        assert session.calls[0]["json"]["dimensions"] == 256
        assert result.shape == (1, 256)

    def test_invalid_dimensions(self, monkeypatch):
        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_DIMENSIONS", "abc")
        with pytest.raises(OpenAIEmbeddingError, match="integer"):
            make_embedder(FakeSession())

    def test_order_normalization_and_usage(self):
        session = FakeSession(dim=8)
        embedder = make_embedder(session)
        texts = ["one", "two", "three"]
        result = embedder.embed(texts)
        assert result.dtype == np.float32
        assert result.shape == (3, 8)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-5)
        for row, text in zip(result, texts):
            expected = np.asarray(FakeSession.vector(text, 8), dtype=np.float32)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(row, expected, rtol=1e-5)
        assert embedder.total_tokens == 30
        assert embedder.requests == 1

    def test_batches_by_input_count(self):
        session = FakeSession()
        embedder = make_embedder(session, batch_inputs=2)
        result = embedder.embed([f"t{i}" for i in range(5)])
        assert [len(c["json"]["input"]) for c in session.calls] == [2, 2, 1]
        assert result.shape[0] == 5
        assert embedder.requests == 3

    def test_batches_by_token_budget(self):
        session = FakeSession()
        embedder = make_embedder(session, batch_tokens=5)
        embedder.embed(["a b c", "d e", "f g h i"])  # 3, 2, 4 tokens
        assert [c["json"]["input"] for c in session.calls] == [["a b c", "d e"], ["f g h i"]]

    def test_long_input_truncated(self, monkeypatch):
        monkeypatch.setattr(embeddings_openai, "MAX_INPUT_TOKENS", 4)
        session = FakeSession()
        embedder = make_embedder(session)
        embedder.embed(["one two three four five six", "short"])
        assert session.calls[0]["json"]["input"] == ["one two three four", "short"]
        assert embedder.truncated == 1

    def test_byte_length_fallback_without_tokenizer(self, monkeypatch):
        monkeypatch.setattr(embeddings_openai, "_encoder", None)
        assert embeddings_openai.count_tokens("abcd") == 4
        assert embeddings_openai.truncate_to_tokens("abcdef", 3) == "abc"

    def test_empty_input_replaced(self):
        session = FakeSession()
        make_embedder(session).embed([""])
        assert session.calls[0]["json"]["input"] == [" "]

    def test_count_mismatch(self):
        bad = FakeResponse(200, {"data": [], "usage": {"prompt_tokens": 1}})
        with pytest.raises(OpenAIEmbeddingError, match="0 embeddings for 1"):
            make_embedder(FakeSession(responses=[bad])).embed(["a"])


class TestRetries:
    def test_retry_on_429_honours_retry_after(self):
        delays = []
        session = FakeSession(responses=[FakeResponse(429, {}, {"retry-after": "3"}), None])
        embedder = OpenAIEmbedder(api_key=FAKE_KEY, session=session, sleep=delays.append)
        result = embedder.embed(["a"])
        assert result.shape[0] == 1
        assert len(session.calls) == 2
        assert delays == [3.0]

    def test_retry_on_5xx_with_backoff(self):
        delays = []
        session = FakeSession(responses=[FakeResponse(500), FakeResponse(503), None])
        embedder = OpenAIEmbedder(api_key=FAKE_KEY, session=session, sleep=delays.append)
        embedder.embed(["a"])
        assert delays == [1.0, 2.0]

    def test_retry_on_connection_error(self):
        import requests

        session = FakeSession(responses=[requests.ConnectionError("boom"), None])
        embedder = make_embedder(session)
        assert embedder.embed(["a"]).shape[0] == 1

    def test_gives_up_after_max_retries(self):
        session = FakeSession(responses=[FakeResponse(429)] * 3)
        embedder = make_embedder(session, max_retries=2)
        with pytest.raises(OpenAIEmbeddingError, match="HTTP 429"):
            embedder.embed(["a"])
        assert len(session.calls) == 3

    def test_client_error_not_retried_and_key_redacted(self):
        body = {"error": {"message": f"Incorrect API key provided: {FAKE_KEY}. Also sk-other-key."}}
        session = FakeSession(responses=[FakeResponse(401, body)])
        with pytest.raises(OpenAIEmbeddingError) as excinfo:
            make_embedder(session).embed(["a"])
        message = str(excinfo.value)
        assert "HTTP 401" in message
        assert FAKE_KEY not in message
        assert "sk-other-key" not in message
        assert len(session.calls) == 1

    def test_key_without_known_prefix_redacted(self):
        key = "proxy-token-XYZ987"
        body = {"error": {"message": f"Token {key} is not valid"}}
        session = FakeSession(responses=[FakeResponse(403, body)])
        embedder = OpenAIEmbedder(api_key=key, session=session, sleep=lambda s: None)
        with pytest.raises(OpenAIEmbeddingError) as excinfo:
            embedder.embed(["a"])
        assert key not in str(excinfo.value)
        assert "HTTP 403" in str(excinfo.value)


SAMPLE_CODE = '''
def connect_database(url):
    """Open a database connection pool."""
    return url


def send_email(to, subject):
    """Send a notification email."""
    return to, subject
'''


@pytest.fixture
def py_project(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text(SAMPLE_CODE)
    return tmp_path


@pytest.fixture
def fake_api(monkeypatch):
    """Route every OpenAIEmbedder to one fake session and a fake key."""
    pytest.importorskip("faiss")
    session = FakeSession(dim=16)
    original_init = OpenAIEmbedder.__init__

    def init(self, *args, **kwargs):
        kwargs["session"] = session
        kwargs["api_key"] = FAKE_KEY
        kwargs["sleep"] = lambda s: None
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(OpenAIEmbedder, "__init__", init)
    return session


@pytest.fixture
def no_local_model(monkeypatch):
    """Fail if anything imports sentence-transformers or torch."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("sentence_transformers", "torch", "transformers"):
            raise AssertionError(f"unexpected import of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def read_metadata(project):
    path = Path(project) / ".tldr" / "cache" / "semantic" / "metadata.json"
    return json.loads(path.read_text())


class TestSemanticIntegration:
    def test_provider_resolution(self, monkeypatch):
        from tldr.semantic import resolve_provider

        assert resolve_provider() == "local"
        monkeypatch.setenv("TLDR_EMBEDDING_PROVIDER", "OpenAI")
        assert resolve_provider() == "openai"
        assert resolve_provider("local") == "local"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            resolve_provider("voyage")

    def test_index_and_search_with_openai(self, py_project, fake_api, no_local_model, monkeypatch):
        from tldr.semantic import build_semantic_index, semantic_search

        monkeypatch.setenv("TLDR_EMBEDDING_PROVIDER", "openai")
        count = build_semantic_index(str(py_project), lang="python", show_progress=False)
        assert count == 2

        metadata = read_metadata(py_project)
        assert metadata["provider"] == "openai"
        assert metadata["model"] == "text-embedding-3-small"
        assert metadata["dimension"] == 16

        results = semantic_search(str(py_project), "database connection", k=2)
        assert {r["name"] for r in results} == {"connect_database", "send_email"}
        # query sent as-is, without the local-model instruction prefix
        assert fake_api.calls[-1]["json"]["input"] == ["database connection"]

    def test_provider_argument_without_environment(self, py_project, fake_api, no_local_model):
        from tldr.semantic import build_semantic_index, semantic_search

        build_semantic_index(str(py_project), lang="python", show_progress=False, provider="openai")
        assert read_metadata(py_project)["provider"] == "openai"
        # search follows the provider recorded in the index
        assert len(semantic_search(str(py_project), "email", k=1)) == 1

    def test_missing_key_fails_before_indexing(self, py_project, monkeypatch):
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_EMBEDDING_PROVIDER", "openai")
        with pytest.raises(OpenAIEmbeddingError, match="No OpenAI API key"):
            build_semantic_index(str(py_project), lang="python", show_progress=False)
        assert not (py_project / ".tldr" / "cache" / "semantic" / "index.faiss").exists()

    def test_search_with_other_model_is_rejected(self, py_project, fake_api, monkeypatch):
        from tldr.semantic import build_semantic_index, semantic_search

        build_semantic_index(str(py_project), lang="python", show_progress=False, provider="openai")
        calls = len(fake_api.calls)
        with pytest.raises(ValueError, match="text-embedding-3-small"):
            semantic_search(str(py_project), "email", model="text-embedding-3-large")
        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        with pytest.raises(ValueError, match="cannot search it with model"):
            semantic_search(str(py_project), "email")
        assert len(fake_api.calls) == calls  # nothing sent

    def test_search_with_other_provider_is_rejected(self, py_project, fake_api, monkeypatch):
        from tldr.semantic import build_semantic_index, semantic_search

        build_semantic_index(str(py_project), lang="python", show_progress=False, provider="openai")
        monkeypatch.setenv("TLDR_EMBEDDING_PROVIDER", "local")
        with pytest.raises(ValueError, match="provider 'openai'"):
            semantic_search(str(py_project), "email")

    def test_legacy_index_with_openai_provider_is_rejected(self, py_project, fake_api, monkeypatch):
        import faiss
        from tldr.semantic import semantic_search

        cache = py_project / ".tldr" / "cache" / "semantic"
        cache.mkdir(parents=True)
        index = faiss.IndexFlatIP(4)
        index.add(np.eye(4, dtype=np.float32))
        faiss.write_index(index, str(cache / "index.faiss"))
        unit = {"name": "f", "qualified_name": "f", "file": "app.py", "line": 1,
                "unit_type": "function", "signature": "def f()"}
        (cache / "metadata.json").write_text(json.dumps(
            {"units": [unit] * 4, "model": "BAAI/bge-large-en-v1.5", "dimension": 4, "count": 4}
        ))
        monkeypatch.setenv("TLDR_EMBEDDING_PROVIDER", "openai")
        with pytest.raises(ValueError, match="provider 'local'"):
            semantic_search(str(py_project), "email")
        assert fake_api.calls == []

    def test_dimension_mismatch_is_rejected(self, py_project, fake_api, monkeypatch):
        from tldr.semantic import build_semantic_index, semantic_search

        build_semantic_index(str(py_project), lang="python", show_progress=False, provider="openai")
        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_DIMENSIONS", "8")
        with pytest.raises(ValueError, match="DIMENSIONS=8 does not match"):
            semantic_search(str(py_project), "email")

    def test_reduced_dimensions_round_trip(self, py_project, fake_api, monkeypatch):
        from tldr.semantic import build_semantic_index, semantic_search

        monkeypatch.setenv("TLDR_OPENAI_EMBEDDING_DIMENSIONS", "12")
        build_semantic_index(str(py_project), lang="python", show_progress=False, provider="openai")
        assert read_metadata(py_project)["dimension"] == 12
        monkeypatch.delenv("TLDR_OPENAI_EMBEDDING_DIMENSIONS")
        semantic_search(str(py_project), "email", k=1)
        assert fake_api.calls[-1]["json"]["dimensions"] == 12

    def test_local_index_records_provider(self, py_project, monkeypatch):
        """Default provider still uses the local model and records it."""
        pytest.importorskip("faiss")
        import tldr.semantic as semantic

        class FakeModel:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 4), dtype=np.float32) / 2

        requested = []
        monkeypatch.setattr(semantic, "get_model", lambda name=None: requested.append(name) or FakeModel())
        semantic.build_semantic_index(str(py_project), lang="python", show_progress=False)
        metadata = read_metadata(py_project)
        assert metadata["provider"] == "local"
        assert metadata["model"] == "BAAI/bge-large-en-v1.5"
        assert requested == [None]

    def test_cli_provider_flag(self, py_project, fake_api, monkeypatch, capsys):
        from tldr import cli

        monkeypatch.setattr(sys, "argv", ["tldr", "semantic", "index", str(py_project),
                                          "--lang", "python", "--provider", "openai"])
        cli.main()
        assert "Indexed 2 code units" in capsys.readouterr().out
        assert read_metadata(py_project)["provider"] == "openai"
