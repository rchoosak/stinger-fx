"""NotificationDispatcher — bus event → formatted message → webhook POST."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from stinger_fx.config.models import NotificationChannelConfig
from stinger_fx.core import AsyncEventBus
from stinger_fx.core.events import (
    ConfigReloadedEvent,
    ConfigReloadFailedEvent,
    DecisionEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    StrategyStateChangedEvent,
)
from stinger_fx.domain import (
    Decision,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Signal,
    SignalStrength,
)
from stinger_fx.observability.notifications import (
    DiscordSink,
    NotificationDispatcher,
    TelegramSink,
    _format_kill_switch,
    _format_order_filled,
    _format_order_rejected,
    _format_signal_rejected,
    _format_strategy_state,
    build_sink,
    known_event_names,
)

# --- Formatters --------------------------------------------------------------


def _order(side: Side = Side.BUY) -> Order:
    return Order(
        ticket=1, strategy_id="ma_test", symbol="EURUSD",
        side=side, type=OrderType.MARKET, volume=0.1,
        fill_price=1.105, status=OrderStatus.FILLED,
    )


def _signal() -> Signal:
    return Signal(
        strategy_id="ma_test", time=datetime.now(UTC),
        symbol="EURUSD", side=Side.BUY, strength=SignalStrength.NORMAL,
    )


def test_format_order_filled_contains_key_fields() -> None:
    msg = _format_order_filled(OrderFilledEvent(order=_order()))
    assert msg is not None
    assert "EURUSD" in msg
    assert "BUY" in msg
    assert "ma_test" in msg
    assert "1.105" in msg


def test_format_order_rejected_includes_reason() -> None:
    msg = _format_order_rejected(
        OrderRejectedEvent(order=_order(), reason="margin_insufficient")
    )
    assert msg is not None
    assert "margin_insufficient" in msg


def test_format_signal_rejected_only_fires_on_rejected_action() -> None:
    sig = _signal()
    placed = DecisionEvent(
        decision=Decision(signal=sig, time=sig.time, action="placed", reason="")
    )
    rejected = DecisionEvent(
        decision=Decision(
            signal=sig,
            time=sig.time,
            action="rejected",
            reason="max_open_positions_per_strategy=5",
            risk_check_passed=False,
        )
    )
    assert _format_signal_rejected(placed) is None
    msg = _format_signal_rejected(rejected)
    assert msg is not None
    assert "max_open_positions_per_strategy" in msg


def test_format_kill_switch_only_fires_on_kill_switch_reason() -> None:
    sig = _signal()
    other = DecisionEvent(
        decision=Decision(
            signal=sig, time=sig.time, action="rejected",
            reason="max_daily_loss_pct=5.0 hit", risk_check_passed=False,
        )
    )
    kill = DecisionEvent(
        decision=Decision(
            signal=sig, time=sig.time, action="rejected",
            reason="kill_switch_tripped", risk_check_passed=False,
        )
    )
    assert _format_kill_switch(other) is None
    kill_msg = _format_kill_switch(kill)
    assert kill_msg is not None
    assert "KILL SWITCH" in kill_msg


def test_format_strategy_state_skips_started() -> None:
    started = StrategyStateChangedEvent(strategy_id="s1", state="started")
    paused = StrategyStateChangedEvent(strategy_id="s1", state="paused")
    quarantined = StrategyStateChangedEvent(
        strategy_id="s1", state="quarantined", reason="too many errors",
    )
    assert _format_strategy_state(started) is None
    paused_msg = _format_strategy_state(paused)
    quarantined_msg = _format_strategy_state(quarantined)
    assert paused_msg is not None
    assert quarantined_msg is not None
    assert "PAUSED" in paused_msg
    assert "QUARANTINED" in quarantined_msg
    assert "too many errors" in quarantined_msg


def test_known_event_names_complete() -> None:
    # If we add a new event, this test catches missing entries.
    names = known_event_names()
    for required in (
        "order_filled",
        "order_rejected",
        "signal_rejected_by_risk",
        "kill_switch_tripped",
        "strategy_state_changed",
        "config_reloaded",
        "config_reload_failed",
    ):
        assert required in names


# --- Sinks (HTTP) -----------------------------------------------------------


def _telegram_cfg(events: list[str] | None = None) -> NotificationChannelConfig:
    return NotificationChannelConfig(
        kind="telegram", enabled=True,
        bot_token="t0k3n", chat_id=12345,
        events=events or ["order_filled"],
    )


def _discord_cfg(events: list[str] | None = None) -> NotificationChannelConfig:
    return NotificationChannelConfig(
        kind="discord", enabled=True,
        webhook_url="https://discord.invalid/webhooks/123/abc",
        events=events or ["order_filled"],
    )


def _mock_transport(responses: list[tuple[int, str]]) -> httpx.MockTransport:
    """Return a transport that replies with the next (status, body) for each call."""
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            status, body = next(calls)
        except StopIteration:
            status, body = (200, "{}")
        return httpx.Response(status, text=body, request=request)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_telegram_sink_posts_to_api() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sink = TelegramSink(_telegram_cfg(), client)
    ok = await sink.send("hello world")
    assert ok is True
    assert len(captured) == 1
    assert "api.telegram.org/bott0k3n/sendMessage" in str(captured[0].url)
    body = captured[0].read().decode()
    assert "hello world" in body
    assert '"chat_id":' in body or '"chat_id"' in body
    await client.aclose()


@pytest.mark.asyncio
async def test_discord_sink_posts_to_webhook() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(204, request=request)  # Discord returns 204 No Content

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sink = DiscordSink(_discord_cfg(), client)
    ok = await sink.send("hello")
    assert ok is True
    assert "discord.invalid/webhooks/123/abc" in str(captured[0].url)
    await client.aclose()


@pytest.mark.asyncio
async def test_sink_non_2xx_does_not_raise() -> None:
    client = httpx.AsyncClient(transport=_mock_transport([(500, "boom")]))
    sink = DiscordSink(_discord_cfg(), client)
    # Non-2xx must come back as False, not raise
    assert await sink.send("nope") is False
    await client.aclose()


def test_build_sink_rejects_unknown_kind() -> None:
    # Bypass schema validation to inject an unsupported kind — that's the
    # whole point of this test (build_sink must reject it).
    cfg = NotificationChannelConfig.model_construct(
        kind="unknown",  # type: ignore[arg-type]
        enabled=True, bot_token="", chat_id="", webhook_url="", events=[],
    )
    with pytest.raises(ValueError):
        build_sink(cfg, httpx.AsyncClient())


# --- Dispatcher ------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_delivers_filled_order_to_telegram() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bus = AsyncEventBus()
    dispatcher = NotificationDispatcher(
        bus,
        [_telegram_cfg(events=["order_filled"])],
        client=client,
    )
    await dispatcher.start()
    await bus.publish(OrderFilledEvent(order=_order()))
    await asyncio.sleep(0.05)
    assert len(captured) == 1
    body = captured[0]
    assert "EURUSD" in body and "BUY" in body
    await dispatcher.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_dispatcher_respects_event_filter() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(204, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bus = AsyncEventBus()
    # Only subscribes to order_rejected, NOT order_filled
    dispatcher = NotificationDispatcher(
        bus,
        [_discord_cfg(events=["order_rejected"])],
        client=client,
    )
    await dispatcher.start()
    await bus.publish(OrderFilledEvent(order=_order()))
    await asyncio.sleep(0.05)
    assert captured == []
    await bus.publish(
        OrderRejectedEvent(order=_order(), reason="margin_insufficient")
    )
    await asyncio.sleep(0.05)
    assert len(captured) == 1
    assert "margin_insufficient" in captured[0]
    await dispatcher.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_dispatcher_dedupes_identical_messages_within_window() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(204, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bus = AsyncEventBus()
    dispatcher = NotificationDispatcher(
        bus,
        [_discord_cfg(events=["order_filled"])],
        client=client,
    )
    await dispatcher.start()
    # Three identical fills back-to-back → only one delivery
    for _ in range(3):
        await bus.publish(OrderFilledEvent(order=_order()))
    await asyncio.sleep(0.05)
    assert len(captured) == 1
    await dispatcher.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_dispatcher_disabled_channel_never_subscribes() -> None:
    """An `enabled: false` channel must not consume bus events."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(204, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = NotificationChannelConfig(
        kind="discord", enabled=False,
        webhook_url="https://discord.invalid/webhooks/x/y",
        events=["order_filled"],
    )
    bus = AsyncEventBus()
    dispatcher = NotificationDispatcher(bus, [cfg], client=client)
    await dispatcher.start()
    await bus.publish(OrderFilledEvent(order=_order()))
    await asyncio.sleep(0.05)
    assert captured == []
    await dispatcher.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_dispatcher_handles_config_reloaded_changes() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(204, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bus = AsyncEventBus()
    dispatcher = NotificationDispatcher(
        bus,
        [_discord_cfg(events=["config_reloaded", "config_reload_failed"])],
        client=client,
    )
    await dispatcher.start()

    # Empty reload — skipped by formatter
    await bus.publish(ConfigReloadedEvent(changes={}))
    await asyncio.sleep(0.05)
    assert captured == []

    # Real reload — delivered
    await bus.publish(ConfigReloadedEvent(changes={"applied": ["strategy[ma]"], "needs_restart": []}))
    await asyncio.sleep(0.05)
    assert len(captured) == 1
    assert "applied" in captured[0]

    # Reload failed — delivered
    await bus.publish(ConfigReloadFailedEvent(file="strategies.yaml", error="bad yaml"))
    await asyncio.sleep(0.05)
    assert len(captured) == 2
    assert "bad yaml" in captured[1]
    await dispatcher.stop()
    await bus.close()
