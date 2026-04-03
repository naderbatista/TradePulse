"""
TradePulse - Dashboard Web
Servidor FastAPI com WebSocket para visualização em tempo real
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .exchange import ExchangeClient
from .strategy import MACrossoverRSI, Signal
from .risk import RiskManager
from .paper_trading import PaperTrader
from .logger import setup_logger, log_decision

logger = logging.getLogger("tradepulse")

app = FastAPI(title="TradePulse Dashboard", version="1.0.0")


class DashboardState:
    """Estado global do dashboard compartilhado via WebSocket."""

    def __init__(self):
        self.config: Config | None = None
        self.exchange_client: ExchangeClient | None = None
        self.strategy: MACrossoverRSI | None = None
        self.risk_manager: RiskManager | None = None
        self.paper_trader: PaperTrader | None = None
        self.running: bool = False
        self.connected_clients: list[WebSocket] = []
        self.trade_history: list[dict] = []
        self.signal_history: list[dict] = []
        self.price_history: list[dict] = []
        self.current_price: float = 0.0
        self.current_signal: str = "HOLD"
        self.last_update: str = ""
        self.task: asyncio.Task | None = None

    async def broadcast(self, data: dict) -> None:
        """Envia dados para todos os clientes WebSocket conectados."""
        disconnected = []
        message = json.dumps(data, ensure_ascii=False)
        for ws in self.connected_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.connected_clients.remove(ws)

    def get_snapshot(self) -> dict:
        """Retorna estado completo atual para novos clientes."""
        summary = {}
        if self.paper_trader:
            summary = self.paper_trader.get_summary()

        open_pos = None
        if self.risk_manager and self.risk_manager.open_position and not self.risk_manager.open_position.closed:
            pos = self.risk_manager.open_position
            open_pos = {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "amount": round(pos.amount, 6),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }

        return {
            "type": "snapshot",
            "running": self.running,
            "exchange": self.config.exchange if self.config else "",
            "symbol": self.config.symbol if self.config else "",
            "timeframe": self.config.timeframe if self.config else "",
            "mode": self.config.trading_mode if self.config else "paper",
            "current_price": self.current_price,
            "current_signal": self.current_signal,
            "last_update": self.last_update,
            "open_position": open_pos,
            "paper_summary": summary,
            "price_history": self.price_history[-100:],
            "trade_history": self.trade_history[-50:],
            "signal_history": self.signal_history[-50:],
            "daily_pnl": self.risk_manager.daily_pnl if self.risk_manager else 0,
            "daily_trades": self.risk_manager.daily_trades if self.risk_manager else 0,
        }


state = DashboardState()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve a página principal do dashboard."""
    html_path = Path(__file__).parent.parent / "templates" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para atualizações em tempo real."""
    await websocket.accept()
    state.connected_clients.append(websocket)
    logger.info("Cliente WebSocket conectado. Total: %d", len(state.connected_clients))

    # Envia snapshot inicial
    await websocket.send_text(json.dumps(state.get_snapshot(), ensure_ascii=False))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "start":
                await start_bot()
            elif action == "stop":
                await stop_bot()
            elif action == "status":
                await websocket.send_text(json.dumps(state.get_snapshot(), ensure_ascii=False))
    except WebSocketDisconnect:
        state.connected_clients.remove(websocket)
        logger.info("Cliente WebSocket desconectado. Total: %d", len(state.connected_clients))


async def start_bot():
    """Inicia o loop de trading."""
    if state.running:
        return

    config = Config()
    state.config = config
    # Paper trading é simulado localmente, não precisa de sandbox da exchange
    state.exchange_client = ExchangeClient(config, sandbox=False)
    state.strategy = MACrossoverRSI(config)
    state.risk_manager = RiskManager(config=config)

    if config.trading_mode == "paper":
        state.paper_trader = PaperTrader(config)

    await state.exchange_client.connect()
    state.running = True

    await state.broadcast({"type": "status", "running": True, "message": "Bot iniciado"})
    logger.info("Bot iniciado via dashboard")

    state.task = asyncio.create_task(_trading_loop())


async def stop_bot():
    """Para o loop de trading."""
    state.running = False
    if state.task:
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
        state.task = None

    if state.exchange_client:
        await state.exchange_client.close()

    await state.broadcast({"type": "status", "running": False, "message": "Bot parado"})
    logger.info("Bot parado via dashboard")


async def _trading_loop():
    """Loop principal de trading que envia updates ao dashboard."""
    while state.running:
        try:
            await _trading_cycle()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Erro no ciclo: %s", e)
            await state.broadcast({"type": "error", "message": str(e)})

        await asyncio.sleep(state.config.check_interval if state.config else 60)


async def _trading_cycle():
    """Um ciclo de análise + possível execução com broadcast."""
    if not state.exchange_client or not state.strategy or not state.config:
        return

    # 1. Buscar dados
    ohlcv = await state.exchange_client.fetch_ohlcv()
    ticker = await state.exchange_client.fetch_ticker()
    current_price = float(ticker.get("last", 0))

    if current_price <= 0:
        return

    state.current_price = current_price
    state.last_update = datetime.now(timezone.utc).strftime("%H:%M:%S")

    # Registrar preço
    price_entry = {
        "time": state.last_update,
        "price": current_price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state.price_history.append(price_entry)
    if len(state.price_history) > 500:
        state.price_history = state.price_history[-500:]

    # 2. Verificar saída
    if state.risk_manager:
        exit_reason = state.risk_manager.check_exit_conditions(current_price)
        if exit_reason:
            await _close_position(current_price, exit_reason)

    # 3. Analisar estratégia
    signal, reasons = state.strategy.analyze(ohlcv)
    state.current_signal = signal.value

    signal_entry = {
        "time": state.last_update,
        "signal": signal.value,
        "price": current_price,
        "reasons": reasons,
    }
    state.signal_history.append(signal_entry)
    if len(state.signal_history) > 200:
        state.signal_history = state.signal_history[-200:]

    # 4. Executar
    if signal == Signal.BUY:
        await _execute_buy(current_price)
    elif signal == Signal.SELL:
        await _execute_sell(current_price)

    # 5. Broadcast update
    await state.broadcast({
        "type": "update",
        "current_price": current_price,
        "signal": signal.value,
        "reasons": reasons,
        "last_update": state.last_update,
        "open_position": _get_position_dict(),
        "paper_summary": state.paper_trader.get_summary() if state.paper_trader else {},
        "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
        "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
    })


async def _execute_buy(price: float):
    if not state.risk_manager or not state.config:
        return

    can_trade, reason = state.risk_manager.can_trade()
    if not can_trade:
        return

    if state.paper_trader:
        balance = state.paper_trader.get_balance()
    else:
        balance = await state.exchange_client.get_free_balance("USDT")

    amount = state.risk_manager.calculate_position_size(balance, price)
    if amount <= 0:
        return

    cost = amount * price
    if cost > balance:
        amount = (balance * 0.99) / price
        if amount <= 0:
            return

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, "buy", amount, price)
    elif state.exchange_client:
        if state.config.order_type == "limit":
            await state.exchange_client.create_limit_order(state.config.symbol, "buy", amount, price)
        else:
            await state.exchange_client.create_market_order(state.config.symbol, "buy", amount)

    state.risk_manager.open_trade(state.config.symbol, "buy", price, amount)

    trade_entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": "COMPRA",
        "symbol": state.config.symbol,
        "price": price,
        "amount": round(amount, 6),
        "cost": round(cost, 2),
    }
    state.trade_history.append(trade_entry)
    await state.broadcast({"type": "trade", "trade": trade_entry})


async def _execute_sell(price: float):
    if not state.risk_manager or not state.config:
        return
    if state.risk_manager.open_position is None or state.risk_manager.open_position.closed:
        return

    pos = state.risk_manager.open_position

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, "sell", pos.amount, price)
    elif state.exchange_client:
        if state.config.order_type == "limit":
            await state.exchange_client.create_limit_order(state.config.symbol, "sell", pos.amount, price)
        else:
            await state.exchange_client.create_market_order(state.config.symbol, "sell", pos.amount)

    trade = state.risk_manager.close_trade(price, "sinal_estrategia")

    trade_entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": "VENDA",
        "symbol": state.config.symbol,
        "price": price,
        "amount": round(pos.amount, 6),
        "pnl": trade.pnl,
    }
    state.trade_history.append(trade_entry)
    await state.broadcast({"type": "trade", "trade": trade_entry})


async def _close_position(price: float, reason: str):
    if not state.risk_manager or not state.config:
        return
    if state.risk_manager.open_position is None:
        return

    pos = state.risk_manager.open_position

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, "sell", pos.amount, price)
    elif state.exchange_client:
        await state.exchange_client.create_market_order(state.config.symbol, "sell", pos.amount)

    trade = state.risk_manager.close_trade(price, reason)

    trade_entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": f"SAÍDA ({reason.upper()})",
        "symbol": state.config.symbol,
        "price": price,
        "amount": round(pos.amount, 6),
        "pnl": trade.pnl,
    }
    state.trade_history.append(trade_entry)
    await state.broadcast({"type": "trade", "trade": trade_entry})


def _get_position_dict() -> dict | None:
    if not state.risk_manager or not state.risk_manager.open_position:
        return None
    pos = state.risk_manager.open_position
    if pos.closed:
        return None
    return {
        "symbol": pos.symbol,
        "side": pos.side,
        "entry_price": pos.entry_price,
        "amount": round(pos.amount, 6),
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
    }
