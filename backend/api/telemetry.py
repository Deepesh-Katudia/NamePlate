"""In-process fan-out of per-window telemetry to WebSocket subscribers."""

from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)

SUBSCRIBER_QUEUE_SIZE = 256


class TelemetryHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, message: dict) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # A slow client loses its oldest message rather than stalling the producer.
                logger.warning("telemetry subscriber queue full; dropping oldest message")
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                queue.put_nowait(message)
