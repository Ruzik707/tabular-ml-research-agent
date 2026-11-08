import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from automl_agent.providers import OpenAICompatibleProvider, ProviderError


def test_scoped_cancel_survives_worker_delay_before_complete(monkeypatch):
    http_started = threading.Event()
    worker_entered = threading.Event()
    allow_complete = threading.Event()

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            http_started.set()
            await asyncio.sleep(30)

    monkeypatch.setattr("automl_agent.providers.httpx.AsyncClient", FakeAsyncClient)
    provider = OpenAICompatibleProvider("http://fake.local/v1", "", "test", retries=0)
    token_factory = getattr(provider, "new_cancel_token", None)
    assert callable(token_factory), "provider needs a request-scoped cancellation token"
    token = token_factory()
    scope_factory = getattr(provider, "request_scope", None)
    assert callable(scope_factory), "complete() needs a scope that binds the request token"

    def delayed_worker():
        with scope_factory(token):
            worker_entered.set()
            assert allow_complete.wait(2)
            return provider.complete([], [], timeout=20, max_tokens=10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(delayed_worker)
        assert worker_entered.wait(2)
        token.cancel()
        allow_complete.set()
        with pytest.raises(ProviderError) as caught:
            request.result(timeout=2)

    assert caught.value.code == "cancelled"
    assert not http_started.is_set(), "a pre-cancelled request must not send HTTP"


def test_legacy_cancel_pending_still_aborts_active_http_request(monkeypatch):
    http_started = threading.Event()
    http_cancelled = threading.Event()

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            http_started.set()
            try:
                await asyncio.sleep(30)
            finally:
                http_cancelled.set()

    monkeypatch.setattr("automl_agent.providers.httpx.AsyncClient", FakeAsyncClient)
    provider = OpenAICompatibleProvider("http://fake.local/v1", "", "test", retries=0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(provider.complete, [], [], timeout=20, max_tokens=10)
        assert http_started.wait(2), "fake transport did not start"
        provider.cancel_pending()
        with pytest.raises(ProviderError) as caught:
            request.result(timeout=2)

    assert caught.value.code == "cancelled"
    assert http_cancelled.wait(1), "cancel_pending() must still cancel direct active calls"
