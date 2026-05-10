"""Test fixtures for the maritime mission planner backend."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient as StarletteTestClient
from src.server import app


@pytest_asyncio.fixture
async def client():
    """Async HTTP client (runs the ASGI event loop, so background tasks work)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def ws_client():
    """Sync WebSocket session via Starlette TestClient."""
    with StarletteTestClient(app).websocket_connect("/ws") as ws:
        yield ws
