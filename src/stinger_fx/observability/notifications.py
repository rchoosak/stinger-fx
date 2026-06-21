"""Notification sinks — bus event → Telegram / Discord webhook.

Each sink:
  • Subscribes to the bus events listed in its `events:` config.
  • Formats a one-line message per event.
  • POSTs to its webhook via httpx.AsyncClient.

A NotificationDispatcher owns one sink per configured channel and runs as an
engine lifecycle component. Sinks share the same httpx client and tolerate
HTTP failures (logged, never raised) — notifications are best-effort.

Deduplication: identical messages within a short window are dropped so a
chatty event burst (e.g. 50 ticks-per-second rejections) doesn't spam your
phone.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import httpx

from stinger_fx.config.models import NotificationChannelConfig
from stinger_fx.core.event_bus import AsyncEventBus, Subscription
from stinger_fx.core.events import (
    ConfigReloadedEvent,
    ConfigReloadFailedEvent,
    DecisionEvent,
    Event,
    OrderFilledEvent,
    OrderRejectedEvent,
    StrategyDriftEvent,
    StrategyStateChangedEvent,
)

logger = logging.getLogger("stinger.observability.notifications")


# Mapping from short event names (used in YAML) to (Event subclass, formatter).
# Formatters return None to skip publishing (e.g. when DecisionEvent.action
# isn't "rejected" we don't want to spam).
Formatter = Callable[[Event], str | None]


def _format_order_filled(evt: OrderFilledEvent) -> str | None:
    o = evt.order
    return (
        f"✅ FILLED · {o.symbol} {o.side.value.upper()} vol={o.volume} "
        f"@ {o.fill_price} · strategy={o.strategy_id} · ticket={o.ticket}"
    )


def _format_order_rejected(evt: OrderRejectedEvent) -> str | None:
    o = evt.order
    return (
        f"❌ REJECTED · {o.symbol} {o.side.value.upper()} vol={o.volume} "
        f"· strategy={o.strategy_id} · reason={evt.reason}"
    )


def _format_signal_rejected(evt: DecisionEvent) -> str | None:
    if evt.decision.action != "rejected":
        return None
    d = evt.decision
    return (
        f"🛑 RISK BLOCKED · strategy={d.signal.strategy_id} "
        f"{d.signal.symbol} {d.signal.side.value.upper()} · {d.reason}"
    )


def _format_kill_switch(evt: DecisionEvent) -> str | None:
    if evt.decision.action != "rejected":
        return None
    if "kill_switch" not in evt.decision.reason:
        return None
    return (
        f"🚨 KILL SWITCH TRIPPED · {evt.decision.signal.symbol} · "
        f"{evt.decision.reason}"
    )


def _format_strategy_state(evt: StrategyStateChangedEvent) -> str | None:
    if evt.state not in ("paused", "quarantined", "stopped"):
        return None
    emoji = {"paused": "⏸️", "quarantined": "⚠️", "stopped": "⏹️"}[evt.state]
    suffix = f" · {evt.reason}" if evt.reason else ""
    return f"{emoji} STRATEGY {evt.state.upper()} · id={evt.strategy_id}{suffix}"


def _format_config_reloaded(evt: ConfigReloadedEvent) -> str | None:
    applied = evt.changes.get("applied", [])
    needs_restart = evt.changes.get("needs_restart", [])
    if not applied and not needs_restart:
        return None
    parts: list[str] = []
    if applied:
        parts.append(f"applied={list(applied)}")
    if needs_restart:
        parts.append(f"needs_restart={list(needs_restart)}")
    return "🔁 CONFIG RELOAD · " + " · ".join(parts)


def _format_config_reload_failed(evt: ConfigReloadFailedEvent) -> str | None:
    return f"⛔ CONFIG RELOAD FAILED · {evt.file} · {evt.error}"


def _format_strategy_drift(evt: StrategyDriftEvent) -> str | None:
    return (
        f"📉 DRIFT · strategy={evt.strategy_id} · n={evt.sample_size} · "
        f"live wr={evt.live_win_rate:.2f}/exp_lot={evt.live_expectancy_per_lot:.2f} "
        f"vs backtest wr={evt.baseline_win_rate:.2f}/exp_lot={evt.baseline_expectancy_per_lot:.2f} "
        f"· {evt.reason}"
    )


_EVENT_REGISTRY: dict[str, tuple[type[Event], Formatter]] = {
    "order_filled":            (OrderFilledEvent,           _format_order_filled),       # type: ignore[dict-item]
    "order_rejected":          (OrderRejectedEvent,         _format_order_rejected),     # type: ignore[dict-item]
    "signal_rejected_by_risk": (DecisionEvent,              _format_signal_rejected),    # type: ignore[dict-item]
    "kill_switch_tripped":     (DecisionEvent,              _format_kill_switch),        # type: ignore[dict-item]
    "strategy_state_changed":  (StrategyStateChangedEvent,  _format_strategy_state),     # type: ignore[dict-item]
    "strategy_drift":          (StrategyDriftEvent,         _format_strategy_drift),     # type: ignore[dict-item]
    "config_reloaded":         (ConfigReloadedEvent,        _format_config_reloaded),    # type: ignore[dict-item]
    "config_reload_failed":    (ConfigReloadFailedEvent,    _format_config_reload_failed),  # type: ignore[dict-item]
}


def known_event_names() -> list[str]:
    return list(_EVENT_REGISTRY.keys())


# --- Sinks ------------------------------------------------------------------


class NotificationSink(ABC):
    """One per configured channel."""

    kind: str = "base"

    def __init__(self, cfg: NotificationChannelConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self._client = client

    @abstractmethod
    async def send(self, message: str) -> bool:
        """Return True if delivered (HTTP 2xx)."""


class TelegramSink(NotificationSink):
    kind = "telegram"

    async def send(self, message: str) -> bool:
        if not self.cfg.bot_token or not self.cfg.chat_id:
            logger.warning("telegram sink missing bot_token or chat_id; skipping")
            return False
        url = f"https://api.telegram.org/bot{self.cfg.bot_token}/sendMessage"
        try:
            r = await self._client.post(
                url,
                json={"chat_id": self.cfg.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
        except httpx.HTTPError as e:
            logger.warning("telegram send failed: %s", e)
            return False
        if r.status_code >= 300:
            logger.warning("telegram non-2xx %s: %s", r.status_code, r.text[:200])
            return False
        return True


class DiscordSink(NotificationSink):
    kind = "discord"

    async def send(self, message: str) -> bool:
        if not self.cfg.webhook_url:
            logger.warning("discord sink missing webhook_url; skipping")
            return False
        try:
            r = await self._client.post(
                self.cfg.webhook_url,
                json={"content": message},
                timeout=10,
            )
        except httpx.HTTPError as e:
            logger.warning("discord send failed: %s", e)
            return False
        if r.status_code >= 300:
            logger.warning("discord non-2xx %s: %s", r.status_code, r.text[:200])
            return False
        return True


_KIND_REGISTRY: dict[str, type[NotificationSink]] = {
    "telegram": TelegramSink,
    "discord": DiscordSink,
}


def build_sink(cfg: NotificationChannelConfig, client: httpx.AsyncClient) -> NotificationSink:
    sink_cls = _KIND_REGISTRY.get(cfg.kind)
    if sink_cls is None:
        raise ValueError(f"unknown notification sink kind: {cfg.kind!r}")
    return sink_cls(cfg, client)


# --- Dispatcher --------------------------------------------------------------


class NotificationDispatcher:
    """Owns the sinks + bus subscriptions for the engine's lifetime."""

    DEDUPE_WINDOW = timedelta(seconds=30)
    DEDUPE_MAX = 50

    def __init__(
        self,
        bus: AsyncEventBus,
        channels: list[NotificationChannelConfig],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bus = bus
        self._channels = [c for c in channels if c.enabled]
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._sinks: list[NotificationSink] = []
        self._subs: list[Subscription] = []
        self._recent: deque[tuple[str, datetime]] = deque(maxlen=self.DEDUPE_MAX)

    async def start(self) -> None:
        if not self._channels:
            return
        # Build one sink per channel
        self._sinks = [build_sink(c, self._client) for c in self._channels]
        # Subscribe to each unique event class that any sink cares about.
        wanted = {
            name for ch in self._channels for name in ch.events if name in _EVENT_REGISTRY
        }
        seen_classes: set[type[Event]] = set()
        for name in wanted:
            event_cls, _ = _EVENT_REGISTRY[name]
            if event_cls in seen_classes:
                continue
            seen_classes.add(event_cls)
            self._subs.append(
                self._bus.subscribe(
                    event_cls,
                    self._make_handler(event_cls),
                    name=f"notifications.{event_cls.__name__}",
                )
            )
        logger.info(
            "notification_dispatcher_started channels=%d events=%s",
            len(self._sinks), sorted(wanted),
        )

    async def stop(self) -> None:
        for sub in self._subs:
            await sub.unsubscribe()
        self._subs.clear()
        self._sinks.clear()
        if self._owns_client:
            await self._client.aclose()

    def _make_handler(self, event_cls: type[Event]) -> Callable[[Event], Awaitable[None]]:
        async def handler(evt: Event) -> None:
            for ch, sink in zip(self._channels, self._sinks, strict=True):
                # Find all configured event names that map to this event class
                for name in ch.events:
                    entry = _EVENT_REGISTRY.get(name)
                    if entry is None or entry[0] is not event_cls:
                        continue
                    _, formatter = entry
                    message = formatter(evt)
                    if message is None:
                        continue
                    if self._is_duplicate(message):
                        continue
                    await sink.send(message)
        return handler

    def _is_duplicate(self, message: str) -> bool:
        now = datetime.now(UTC)
        cutoff = now - self.DEDUPE_WINDOW
        # Prune
        while self._recent and self._recent[0][1] < cutoff:
            self._recent.popleft()
        if any(m == message for m, _ in self._recent):
            return True
        self._recent.append((message, now))
        return False
