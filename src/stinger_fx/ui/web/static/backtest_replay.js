(() => {
  const root = document.getElementById("backtest-replay");
  if (!root) return;

  const runId = root.dataset.runId;
  const colors = {
    blue: "#58a6ff",
    grid: "#21262d",
    text: "#c9d1d9",
    muted: "#8b949e",
    buy: "#3fb950",
    sell: "#f85149",
    profit: "#3fb950",
    loss: "#f85149",
    neutral: "#8b949e",
  };

  function toTime(value) {
    const t = new Date(value).getTime();
    return Number.isFinite(t) ? t : null;
  }

  function money(value) {
    const n = Number(value || 0);
    return n.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function normalizeEquity(rows) {
    return (rows || [])
      .map((row) => ({ x: toTime(row.time), y: Number(row.equity) }))
      .filter((row) => row.x !== null && Number.isFinite(row.y));
  }

  function normalizeTrades(trades) {
    return (trades || [])
      .map((trade) => ({
        openTs: trade.open_ts,
        closeTs: trade.close_ts,
        openTime: toTime(trade.open_ts),
        closeTime: toTime(trade.close_ts),
        side: String(trade.side || "").toLowerCase(),
        openPrice: Number(trade.open_price),
        closePrice: Number(trade.close_price),
        pnl: Number(trade.pnl || 0),
        volume: Number(trade.volume || 0),
      }))
      .filter((trade) => trade.openTime !== null && trade.closeTime !== null);
  }

  function normalizeCandles(candles) {
    return (candles || [])
      .map((bar) => ({
        x: toTime(bar.time),
        o: Number(bar.open),
        h: Number(bar.high),
        l: Number(bar.low),
        c: Number(bar.close),
      }))
      .filter((bar) => (
        bar.x !== null
        && Number.isFinite(bar.o)
        && Number.isFinite(bar.h)
        && Number.isFinite(bar.l)
        && Number.isFinite(bar.c)
      ));
  }

  function interpolateEquity(points, timeIso) {
    const t = toTime(timeIso);
    if (t === null || !points.length) return null;
    let prev = points[0];
    for (let i = 1; i < points.length; i += 1) {
      const cur = points[i];
      const tPrev = toTime(prev.x);
      const tCur = toTime(cur.x);
      if (tPrev === null || tCur === null) continue;
      if (t >= tPrev && t <= tCur) {
        const frac = tCur === tPrev ? 0 : (t - tPrev) / (tCur - tPrev);
        return prev.y + frac * (cur.y - prev.y);
      }
      prev = cur;
    }
    return prev.y;
  }

  function baseScales() {
    return {
      x: {
        type: "time",
        ticks: { color: colors.muted, maxRotation: 0 },
        grid: { color: colors.grid },
      },
      y: {
        ticks: { color: colors.muted },
        grid: { color: colors.grid },
      },
    };
  }

  function basePlugins() {
    const compactLegend = new Map([
      ["Open buy", "Buy"],
      ["Open sell", "Sell"],
      ["Close profit", "Win"],
      ["Close loss", "Loss"],
      ["Equity", "Eq"],
      ["OHLC", "Bars"],
    ]);
    return {
      legend: {
        position: "bottom",
        labels: {
          color: colors.text,
          boxWidth: 10,
          boxHeight: 10,
          padding: 10,
          usePointStyle: true,
          font: { size: window.matchMedia("(max-width: 520px)").matches ? 11 : 12 },
          generateLabels: (chart) => {
            const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
            if (!window.matchMedia("(max-width: 520px)").matches) return labels;
            return labels.map((item) => ({
              ...item,
              text: compactLegend.get(item.text) || item.text,
            }));
          },
        },
      },
    };
  }

  function splitEquityMarkers(trades, equityPoints) {
    const openBuy = [];
    const openSell = [];
    const closeProfit = [];
    const closeLoss = [];
    for (const trade of trades) {
      const entryY = interpolateEquity(equityPoints, trade.openTs);
      const exitY = interpolateEquity(equityPoints, trade.closeTs);
      if (entryY !== null) {
        const point = {
          x: trade.openTime,
          y: entryY,
          side: trade.side,
          price: trade.openPrice,
          volume: trade.volume,
        };
        (trade.side === "sell" ? openSell : openBuy).push(point);
      }
      if (exitY !== null) {
        const point = {
          x: trade.closeTime,
          y: exitY,
          pnl: trade.pnl,
          price: trade.closePrice,
          volume: trade.volume,
        };
        (trade.pnl >= 0 ? closeProfit : closeLoss).push(point);
      }
    }
    return { openBuy, openSell, closeProfit, closeLoss };
  }

  function splitPriceMarkers(trades) {
    const openBuy = [];
    const openSell = [];
    const closeProfit = [];
    const closeLoss = [];
    for (const trade of trades) {
      if (Number.isFinite(trade.openPrice)) {
        const openPoint = {
          x: trade.openTime,
          y: trade.openPrice,
          side: trade.side,
          price: trade.openPrice,
          volume: trade.volume,
        };
        (trade.side === "sell" ? openSell : openBuy).push(openPoint);
      }
      if (Number.isFinite(trade.closePrice)) {
        const closePoint = {
          x: trade.closeTime,
          y: trade.closePrice,
          pnl: trade.pnl,
          price: trade.closePrice,
          volume: trade.volume,
        };
        (trade.pnl >= 0 ? closeProfit : closeLoss).push(closePoint);
      }
    }
    return { openBuy, openSell, closeProfit, closeLoss };
  }

  function markerDataset(label, data, color, pointStyle, rotation = 0) {
    return {
      label,
      type: "scatter",
      data,
      backgroundColor: color,
      borderColor: "#0d1117",
      borderWidth: 1,
      pointStyle,
      rotation,
      pointRotation: rotation,
      radius: 6,
      hoverRadius: 8,
      parsing: false,
      order: 0,
    };
  }

  function renderEquityChart(data) {
    const canvas = document.getElementById("replay-chart");
    if (!canvas) return;
    const equityPoints = normalizeEquity(data.equity);
    const trades = normalizeTrades(data.meta?.trades);
    const markers = splitEquityMarkers(trades, equityPoints);

    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [
          {
            label: "Equity",
            data: equityPoints,
            borderColor: colors.blue,
            backgroundColor: "rgba(88, 166, 255, 0.10)",
            fill: true,
            tension: 0,
            pointRadius: 0,
            borderWidth: 1.5,
            parsing: false,
            order: 10,
          },
          markerDataset("Open buy", markers.openBuy, colors.buy, "triangle"),
          markerDataset("Open sell", markers.openSell, colors.sell, "triangle", 180),
          markerDataset("Close profit", markers.closeProfit, colors.profit, "rectRot"),
          markerDataset("Close loss", markers.closeLoss, colors.loss, "rectRot"),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: "nearest" },
        scales: baseScales(),
        plugins: {
          ...basePlugins(),
          tooltip: {
            callbacks: {
              label: (context) => {
                const raw = context.raw || {};
                if (context.dataset.label === "Equity") return `Equity: ${money(raw.y)}`;
                if (context.dataset.label.startsWith("Open")) {
                  return `${context.dataset.label}: ${String(raw.side || "").toUpperCase()} ${raw.volume || ""} @ ${money(raw.price)}`;
                }
                return `${context.dataset.label}: PnL ${raw.pnl >= 0 ? "+" : ""}${money(raw.pnl)} @ ${money(raw.price)}`;
              },
            },
          },
        },
      },
    });
  }

  function renderCandleChart(candlePayload, replayPayload) {
    const canvas = document.getElementById("candle-chart");
    const status = document.getElementById("candle-status");
    if (!canvas || !status) return;

    const candles = normalizeCandles(candlePayload.candles);
    if (!candles.length) {
      status.textContent = "No bar data found for this run.";
      canvas.closest(".chart-wrap")?.classList.add("is-empty");
      return;
    }
    status.textContent = `${candles.length.toLocaleString()} bars${candles.length === 5000 ? " (truncated)" : ""}`;

    const trades = normalizeTrades(replayPayload.meta?.trades);
    const markers = splitPriceMarkers(trades);

    new Chart(canvas.getContext("2d"), {
      type: "candlestick",
      data: {
        datasets: [
          {
            label: "OHLC",
            data: candles,
            color: {
              up: colors.buy,
              down: colors.sell,
              unchanged: colors.neutral,
            },
            borderColor: {
              up: colors.buy,
              down: colors.sell,
              unchanged: colors.neutral,
            },
            order: 10,
          },
          markerDataset("Open buy", markers.openBuy, colors.buy, "triangle"),
          markerDataset("Open sell", markers.openSell, colors.sell, "triangle", 180),
          markerDataset("Close profit", markers.closeProfit, colors.profit, "rectRot"),
          markerDataset("Close loss", markers.closeLoss, colors.loss, "rectRot"),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: "nearest" },
        scales: baseScales(),
        plugins: {
          ...basePlugins(),
          tooltip: {
            callbacks: {
              label: (context) => {
                const raw = context.raw || {};
                if (context.dataset.label === "OHLC") {
                  return `O ${money(raw.o)} H ${money(raw.h)} L ${money(raw.l)} C ${money(raw.c)}`;
                }
                if (context.dataset.label.startsWith("Open")) {
                  return `${context.dataset.label}: ${String(raw.side || "").toUpperCase()} ${raw.volume || ""} @ ${money(raw.price)}`;
                }
                return `${context.dataset.label}: PnL ${raw.pnl >= 0 ? "+" : ""}${money(raw.pnl)} @ ${money(raw.price)}`;
              },
            },
          },
        },
      },
    });
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return response.json();
  }

  async function boot() {
    try {
      const replayPayload = await fetchJson(`/backtest/${encodeURIComponent(runId)}/data.json`);
      renderEquityChart(replayPayload);
      const candlePayload = await fetchJson(`/backtest/${encodeURIComponent(runId)}/candles.json`);
      renderCandleChart(candlePayload, replayPayload);
    } catch (error) {
      const status = document.getElementById("candle-status");
      if (status) status.textContent = "Replay data unavailable.";
      console.error("backtest replay render failed", error);
    }
  }

  boot();
})();
