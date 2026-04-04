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

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .exchange import ExchangeClient
from .strategy import MACrossoverRSI, ScalpingMomentum, Signal
from .risk import RiskManager
from .paper_trading import PaperTrader
from .predictor import MLPredictor
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
        self.predictor: MLPredictor | None = None
        self.running: bool = False
        self.connected_clients: list[WebSocket] = []
        self.trade_history: list[dict] = []
        self.signal_history: list[dict] = []
        self.price_history: list[dict] = []
        self.current_price: float = 0.0
        self.current_signal: str = "HOLD"
        self.last_update: str = ""
        self.task: asyncio.Task | None = None
        # Seleções do usuário (independentes do Config/YAML)
        self.selected_exchange: str = "binance"
        self.selected_mode: str = "paper"  # paper | testnet | live
        self.auto_trade: bool = False  # toggle para abrir posições automaticamente
        self.scalping_mode: bool = False  # modo scalping: trades rápidos com TP fixo em $
        self.daily_profit_target: float = 20.0  # meta de ganho diário para desativar auto-trade

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
            summary = self.paper_trader.get_summary(self.current_price)

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
            "exchange": self.selected_exchange,
            "symbol": self.config.symbol if self.config else "BTC/USDT",
            "timeframe": self.config.timeframe if self.config else "1h",
            "mode": self.selected_mode,
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
            "auto_trade": self.auto_trade,
            "scalping_mode": self.scalping_mode,
            "scalp_tp": self.config.scalping_take_profit_usd if self.config else 3.0,
            "scalp_sl": self.config.scalping_stop_loss_usd if self.config else 2.0,
            "daily_profit_target": self.daily_profit_target,
            "ml_status": self.predictor.get_status() if self.predictor else {},
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
            elif action == "change_symbol":
                symbol = msg.get("symbol", "")
                if symbol:
                    await change_symbol(symbol)
            elif action == "change_timeframe":
                timeframe = msg.get("timeframe", "")
                if timeframe:
                    await change_timeframe(timeframe)
            elif action == "change_exchange":
                exchange = msg.get("exchange", "")
                if exchange:
                    await change_exchange(exchange)
            elif action == "fetch_pairs":
                exchange = msg.get("exchange", state.selected_exchange)
                pairs = await fetch_pairs(exchange)
                await websocket.send_text(json.dumps({"type": "pairs", "exchange": exchange, **pairs}, ensure_ascii=False))
            elif action == "change_mode":
                mode = msg.get("mode", "")
                if mode:
                    await change_mode(mode)
            elif action == "toggle_auto_trade":
                state.auto_trade = msg.get("enabled", False)
                logger.info("Auto-trade %s", "ativado" if state.auto_trade else "desativado")
                await state.broadcast({"type": "auto_trade", "enabled": state.auto_trade})
            elif action == "toggle_scalping":
                enabled = msg.get("enabled", False)
                state.scalping_mode = enabled
                if state.config:
                    state.config.scalping_enabled = enabled
                # Atualiza o risk_manager SEM reiniciar o bot (preserva posição aberta)
                if state.risk_manager:
                    state.risk_manager.scalping_mode = enabled
                    # Recalcula TP/SL da posição aberta se houver
                    if state.risk_manager.open_position and not state.risk_manager.open_position.closed:
                        pos = state.risk_manager.open_position
                        pos.take_profit = state.risk_manager.calculate_take_profit(
                            pos.entry_price, pos.side, pos.amount
                        )
                        pos.stop_loss = state.risk_manager.calculate_stop_loss(
                            pos.entry_price, pos.side, pos.amount
                        )
                        logger.info(
                            "Scalping %s: TP/SL recalculados -> TP=%.2f, SL=%.2f",
                            "ON" if enabled else "OFF", pos.take_profit, pos.stop_loss,
                        )
                # Troca a estratégia sem perder o estado
                if state.config and state.running:
                    if enabled:
                        from .strategy import ScalpingMomentum
                        state.strategy = ScalpingMomentum(state.config)
                        state.config.check_interval = state.config.scalping_check_interval
                    else:
                        from .strategy import MACrossoverRSI
                        state.strategy = MACrossoverRSI(state.config)
                        tf_intervals = {"1m": 15, "5m": 30, "15m": 60, "30m": 90, "1h": 120, "4h": 300, "1d": 600}
                        state.config.check_interval = tf_intervals.get(state.config.timeframe, 60)
                logger.info("Scalping %s", "ativado" if enabled else "desativado")
                await state.broadcast({"type": "scalping_mode", "enabled": enabled})
            elif action == "update_scalp_tpsl":
                tp = float(msg.get("tp", 3.0))
                sl = float(msg.get("sl", 2.0))
                if tp > 0 and sl > 0 and state.config:
                    state.config.scalping_take_profit_usd = tp
                    state.config.scalping_stop_loss_usd = sl
                    # Recalcula TP/SL da posição aberta se houver
                    if (state.risk_manager and state.risk_manager.open_position
                            and not state.risk_manager.open_position.closed
                            and state.risk_manager.scalping_mode):
                        pos = state.risk_manager.open_position
                        pos.take_profit = state.risk_manager.calculate_take_profit(
                            pos.entry_price, pos.side, pos.amount
                        )
                        pos.stop_loss = state.risk_manager.calculate_stop_loss(
                            pos.entry_price, pos.side, pos.amount
                        )
                    logger.info("Scalping TP/SL atualizado: TP=$%.2f, SL=$%.2f", tp, sl)
                    await state.broadcast({"type": "scalp_tpsl", "tp": tp, "sl": sl})
            elif action == "update_profit_target":
                target = float(msg.get("target", 20.0))
                if target > 0:
                    state.daily_profit_target = target
                    if state.config:
                        state.config.daily_profit_target = target
                    logger.info("Meta de ganho diário atualizada: $%.2f", target)
                    await state.broadcast({"type": "profit_target", "target": target})
            elif action == "manual_buy":
                if not state.running or state.current_price <= 0:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Bot não está rodando"}))
                else:
                    logger.info("COMPRA MANUAL solicitada pelo usuário @ %.2f", state.current_price)
                    await _execute_buy(state.current_price, manual=True)
            elif action == "manual_sell":
                if not state.running or state.current_price <= 0:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Bot não está rodando"}))
                else:
                    logger.info("VENDA MANUAL (short) solicitada pelo usuário @ %.2f", state.current_price)
                    await _execute_short(state.current_price, manual=True)
            elif action == "close_position":
                if not state.running or state.current_price <= 0:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Bot não está rodando"}))
                elif not state.risk_manager or not state.risk_manager.open_position or state.risk_manager.open_position.closed:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Nenhuma posição aberta para fechar"}))
                else:
                    logger.info("FECHAR POSIÇÃO solicitado pelo usuário @ %.2f", state.current_price)
                    await _close_position(state.current_price, "manual")
            elif action == "status":
                await websocket.send_text(json.dumps(state.get_snapshot(), ensure_ascii=False))
    except WebSocketDisconnect:
        state.connected_clients.remove(websocket)
        logger.info("Cliente WebSocket desconectado. Total: %d", len(state.connected_clients))


async def start_bot():
    """Inicia o loop de trading."""
    if state.running:
        return

    # Reutiliza config existente (ex: par/timeframe alterados pelo usuário)
    if state.config is None:
        state.config = Config()

    config = state.config

    # Aplica exchange e modo selecionados pelo usuário
    config.exchange = state.selected_exchange
    config.trading_mode = state.selected_mode if state.selected_mode != "testnet" else "paper"

    # Determina se usa sandbox (testnet) ou API real
    use_sandbox = (state.selected_mode == "testnet")

    state.exchange_client = ExchangeClient(config, exchange_name=state.selected_exchange, sandbox=use_sandbox)

    # Escolhe estratégia baseado no modo scalping
    if state.scalping_mode:
        state.strategy = ScalpingMomentum(config)
        config.check_interval = config.scalping_check_interval
        logger.info("Modo SCALPING ativado: TP=$%.2f, SL=$%.2f, intervalo=%ds",
                     config.scalping_take_profit_usd, config.scalping_stop_loss_usd,
                     config.scalping_check_interval)
    else:
        state.strategy = MACrossoverRSI(config)

    state.risk_manager = RiskManager(config=config)
    state.risk_manager.scalping_mode = state.scalping_mode

    # ML Predictor
    state.predictor = MLPredictor(config)
    if not config.ml_enabled:
        state.predictor.enabled = False
        logger.info("ML Predictor desativado via config")
    else:
        logger.info(
            "ML Predictor inicializado: prob_min=%.0f%%, retrain=%d ciclos, horizon=%d velas",
            config.ml_min_probability * 100,
            config.ml_retrain_interval,
            config.ml_prediction_horizon,
        )

    # Meta de ganho diário
    state.daily_profit_target = config.daily_profit_target

    # Paper trading: simulação local (sem ordens reais)
    # Testnet: ordens reais na testnet da exchange
    # Live: ordens reais na exchange principal
    if state.selected_mode == "paper":
        state.paper_trader = PaperTrader(config)
    else:
        state.paper_trader = None

    # Ajusta intervalo de checagem com base no timeframe (scalping já foi configurado acima)
    if not state.scalping_mode:
        tf_intervals = {"1m": 15, "5m": 30, "15m": 60, "30m": 90, "1h": 120, "4h": 300, "1d": 600}
        config.check_interval = tf_intervals.get(config.timeframe, 60)

    await state.exchange_client.connect()
    state.running = True

    mode_labels = {"paper": "Paper (Simulado)", "testnet": "Testnet (API Teste)", "live": "Live (API Real)"}
    await state.broadcast({
        "type": "status",
        "running": True,
        "message": f"Bot iniciado - {state.selected_exchange.upper()} - {mode_labels.get(state.selected_mode, state.selected_mode)}",
    })
    logger.info("Bot iniciado: exchange=%s, modo=%s, sandbox=%s", state.selected_exchange, state.selected_mode, use_sandbox)

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


async def change_symbol(symbol: str):
    """Muda o par de trading em tempo real."""
    if not state.config:
        state.config = Config()
    was_running = state.running

    if was_running:
        await stop_bot()

    state.config.symbol = symbol
    # Para derivativos/perpétuos, configura defaultType como swap
    if ":" in symbol:
        state.config._raw.setdefault("exchange_options", {})["defaultType"] = "swap"
    else:
        state.config._raw.setdefault("exchange_options", {})["defaultType"] = "spot"

    # Limpa históricos ao mudar de par
    state.price_history.clear()
    state.signal_history.clear()
    state.trade_history.clear()
    state.current_price = 0.0
    state.current_signal = "HOLD"

    logger.info("Par alterado para %s", symbol)
    await state.broadcast({"type": "config_change", "symbol": symbol, "timeframe": state.config.timeframe})

    if was_running:
        await start_bot()


async def change_timeframe(timeframe: str):
    """Muda o timeframe em tempo real."""
    if not state.config:
        state.config = Config()
    was_running = state.running

    if was_running:
        await stop_bot()

    state.config.timeframe = timeframe
    # Limpa histórico de preços ao mudar timeframe
    state.price_history.clear()
    state.signal_history.clear()

    logger.info("Timeframe alterado para %s", timeframe)
    await state.broadcast({"type": "config_change", "symbol": state.config.symbol, "timeframe": timeframe})

    if was_running:
        await start_bot()


async def change_exchange(exchange: str):
    """Muda a exchange em tempo real."""
    if exchange not in ("binance", "bybit"):
        return
    was_running = state.running

    if was_running:
        await stop_bot()

    state.selected_exchange = exchange
    if state.config:
        state.config.exchange = exchange

    # Limpa históricos ao mudar exchange
    state.price_history.clear()
    state.signal_history.clear()
    state.trade_history.clear()
    state.current_price = 0.0
    state.current_signal = "HOLD"

    logger.info("Exchange alterada para %s", exchange)
    await state.broadcast({
        "type": "config_change",
        "symbol": state.config.symbol if state.config else "BTC/USDT",
        "timeframe": state.config.timeframe if state.config else "1h",
        "exchange": exchange,
        "mode": state.selected_mode,
    })

    if was_running:
        await start_bot()


async def fetch_pairs(exchange_name: str) -> dict:
    """Busca todos os pares disponíveis de uma exchange (spot + perpétuos)."""
    if exchange_name not in ("binance", "bybit"):
        return {"spot": [], "perpetual": []}

    config = state.config or Config()
    client = ExchangeClient(config, exchange_name=exchange_name, sandbox=False)
    try:
        await client.connect()
        markets = await client.fetch_markets(quote="USDT")
        return markets
    except Exception as e:
        logger.error("Erro ao buscar pares da %s: %s", exchange_name, e)
        return {"spot": [], "perpetual": []}
    finally:
        await client.close()


async def change_mode(mode: str):
    """Muda o modo de operação (paper/testnet/live)."""
    if mode not in ("paper", "testnet", "live"):
        return
    was_running = state.running

    if was_running:
        await stop_bot()

    state.selected_mode = mode

    # Limpa históricos ao mudar modo
    state.price_history.clear()
    state.signal_history.clear()
    state.trade_history.clear()
    state.current_price = 0.0
    state.current_signal = "HOLD"

    logger.info("Modo alterado para %s", mode)
    await state.broadcast({
        "type": "config_change",
        "symbol": state.config.symbol if state.config else "BTC/USDT",
        "timeframe": state.config.timeframe if state.config else "1h",
        "exchange": state.selected_exchange,
        "mode": mode,
    })

    if was_running:
        await start_bot()


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
    df = state.strategy.prepare_dataframe(ohlcv)
    signal, reasons = state.strategy.generate_signal(df)
    state.current_signal = signal.value

    # 3.5 ML Prediction — filtra BUY se probabilidade baixa
    ml_prediction = {"probability": 0.5, "direction": "neutral", "confidence": "desativado", "trained": False}
    if state.predictor and state.predictor.enabled:
        # Treinar/re-treinar quando necessário
        if not state.predictor._is_trained or state.predictor.should_retrain():
            state.predictor.train(df)

        ml_prediction = state.predictor.predict(df)

        # Filtro ML: se estratégia diz BUY mas ML diz probabilidade < min → HOLD
        if signal == Signal.BUY and ml_prediction.get("trained", False):
            ml_prob = ml_prediction.get("probability", 0.5)
            if ml_prob < state.predictor.min_probability:
                reasons["ml_bloqueado"] = True
                reasons["ml_prob"] = ml_prediction["probability"]
                reasons["sinal_original"] = reasons.get("sinal", "")
                reasons["sinal"] = (
                    f"ML BLOQUEOU: prob={ml_prob:.1%} < mín {state.predictor.min_probability:.0%} "
                    f"({ml_prediction['confidence']})"
                )
                signal = Signal.HOLD
                state.current_signal = signal.value
                logger.info(
                    "ML filtrou BUY: prob=%.1f%% < min=%.0f%%",
                    ml_prob * 100,
                    state.predictor.min_probability * 100,
                )

    # Adiciona info ML às reasons
    reasons["ml_probability"] = ml_prediction.get("probability", 0.5)
    reasons["ml_direction"] = ml_prediction.get("direction", "neutral")
    reasons["ml_confidence"] = ml_prediction.get("confidence", "desativado")
    reasons["ml_accuracy"] = ml_prediction.get("accuracy", 0.0)
    reasons["ml_trained"] = ml_prediction.get("trained", False)

    signal_entry = {
        "time": state.last_update,
        "signal": signal.value,
        "price": current_price,
        "reasons": reasons,
    }
    state.signal_history.append(signal_entry)
    if len(state.signal_history) > 200:
        state.signal_history = state.signal_history[-200:]

    # 4. Executar (somente se auto_trade estiver ativado)
    if state.auto_trade:
        # Verificar meta de ganho diário
        if (state.risk_manager
                and state.daily_profit_target > 0
                and state.risk_manager.daily_pnl >= state.daily_profit_target):
            state.auto_trade = False
            logger.info(
                "META DIÁRIA ATINGIDA! PnL=%.2f >= Meta=%.2f. Auto-trade desativado.",
                state.risk_manager.daily_pnl, state.daily_profit_target,
            )
            await state.broadcast({
                "type": "auto_trade", "enabled": False,
                "reason": f"Meta diária atingida: ${state.risk_manager.daily_pnl:.2f} >= ${state.daily_profit_target:.2f}",
            })
        elif signal == Signal.BUY:
            await _execute_buy(current_price)
        elif signal == Signal.SELL:
            # No scalping, NÃO fecha por sinal da estratégia
            # Saída é SOMENTE por TP/SL fixo ($5 lucro / $3 perda)
            if not state.scalping_mode:
                await _execute_sell(current_price)

    # 5. Preparar dados de candles + indicadores para o frontend
    candles = []
    indicators = {"ma_short": [], "ma_long": [], "rsi": [], "volume": []}
    for _, row in df.tail(200).iterrows():
        ts = int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else 0
        candles.append({
            "time": ts,
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
        })
        indicators["volume"].append({"time": ts, "value": float(row["volume"])})
        if not pd.isna(row.get("ma_short")):
            indicators["ma_short"].append({"time": ts, "value": round(float(row["ma_short"]), 2)})
        if not pd.isna(row.get("ma_long")):
            indicators["ma_long"].append({"time": ts, "value": round(float(row["ma_long"]), 2)})
        if not pd.isna(row.get("rsi")):
            indicators["rsi"].append({"time": ts, "value": round(float(row["rsi"]), 2)})

    # 6. Preparar dados de análise detalhada
    current_row = df.iloc[-1]
    previous_row = df.iloc[-2] if len(df) >= 2 else None

    crossover_type = "Sem cruzamento"
    if previous_row is not None:
        if previous_row["ma_short"] <= previous_row["ma_long"] and current_row["ma_short"] > current_row["ma_long"]:
            crossover_type = "Golden Cross"
        elif previous_row["ma_short"] >= previous_row["ma_long"] and current_row["ma_short"] < current_row["ma_long"]:
            crossover_type = "Death Cross"

    analysis_data = {
        "ma_short": round(float(current_row["ma_short"]), 2),
        "ma_long": round(float(current_row["ma_long"]), 2),
        "rsi": round(float(current_row["rsi"]), 2),
        "crossover_type": crossover_type,
        "price": current_price,
        "ma_diff": round(float(current_row["ma_short"] - current_row["ma_long"]), 2),
    }

    # 7. Broadcast update
    await state.broadcast({
        "type": "update",
        "current_price": current_price,
        "signal": signal.value,
        "reasons": reasons,
        "last_update": state.last_update,
        "open_position": _get_position_dict(),
        "paper_summary": state.paper_trader.get_summary(state.current_price) if state.paper_trader else {},
        "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
        "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
        "candles": candles,
        "indicators": indicators,
        "analysis": analysis_data,
        "ml_prediction": ml_prediction,
        "ml_status": state.predictor.get_status() if state.predictor else {},
    })


async def _execute_buy(price: float, manual: bool = False):
    if not state.risk_manager or not state.config:
        if manual:
            await state.broadcast({"type": "error", "message": "Risk manager não inicializado"})
        return

    can_trade, reason = state.risk_manager.can_trade()
    if not can_trade:
        if manual:
            await state.broadcast({"type": "error", "message": f"Compra bloqueada: {reason}"})
        return

    if state.paper_trader:
        balance = state.paper_trader.get_balance()
    else:
        balance = await state.exchange_client.get_free_balance("USDT")

    amount = state.risk_manager.calculate_position_size(balance, price)
    if amount <= 0:
        if manual:
            await state.broadcast({"type": "error", "message": f"Saldo insuficiente: ${balance:.2f}"})
        return

    cost = amount * price
    if cost > balance:
        amount = (balance * 0.99) / price
        if amount <= 0:
            if manual:
                await state.broadcast({"type": "error", "message": f"Saldo insuficiente: ${balance:.2f}"})
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
    # Atualiza saldo imediatamente no frontend
    if state.paper_trader:
        await state.broadcast({
            "type": "balance_update",
            "paper_summary": state.paper_trader.get_summary(price),
            "open_position": _get_position_dict(),
            "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
            "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
        })


async def _execute_short(price: float, manual: bool = False):
    """Abre uma posição SHORT (venda) ao preço atual."""
    if not state.risk_manager or not state.config:
        if manual:
            await state.broadcast({"type": "error", "message": "Risk manager não inicializado"})
        return

    can_trade, reason = state.risk_manager.can_trade()
    if not can_trade:
        if manual:
            await state.broadcast({"type": "error", "message": f"Venda bloqueada: {reason}"})
        return

    if state.paper_trader:
        balance = state.paper_trader.get_balance()
    else:
        balance = await state.exchange_client.get_free_balance("USDT")

    amount = state.risk_manager.calculate_position_size(balance, price)
    if amount <= 0:
        if manual:
            await state.broadcast({"type": "error", "message": f"Saldo insuficiente: ${balance:.2f}"})
        return

    cost = amount * price
    if cost > balance:
        amount = (balance * 0.99) / price
        if amount <= 0:
            if manual:
                await state.broadcast({"type": "error", "message": f"Saldo insuficiente: ${balance:.2f}"})
            return

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, "sell", amount, price)
    elif state.exchange_client:
        if state.config.order_type == "limit":
            await state.exchange_client.create_limit_order(state.config.symbol, "sell", amount, price)
        else:
            await state.exchange_client.create_market_order(state.config.symbol, "sell", amount)

    state.risk_manager.open_trade(state.config.symbol, "sell", price, amount)

    trade_entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": "VENDA (SHORT)",
        "symbol": state.config.symbol,
        "price": price,
        "amount": round(amount, 6),
        "cost": round(cost, 2),
    }
    state.trade_history.append(trade_entry)
    await state.broadcast({"type": "trade", "trade": trade_entry})
    if state.paper_trader:
        await state.broadcast({
            "type": "balance_update",
            "paper_summary": state.paper_trader.get_summary(price),
            "open_position": _get_position_dict(),
            "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
            "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
        })


async def _execute_sell(price: float):
    if not state.risk_manager or not state.config:
        return
    if state.risk_manager.open_position is None or state.risk_manager.open_position.closed:
        return

    pos = state.risk_manager.open_position

    # Lado oposto para fechar: LONG fecha com sell, SHORT fecha com buy
    close_side = "buy" if pos.side == "sell" else "sell"

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, close_side, pos.amount, price)
    elif state.exchange_client:
        if state.config.order_type == "limit":
            await state.exchange_client.create_limit_order(state.config.symbol, close_side, pos.amount, price)
        else:
            await state.exchange_client.create_market_order(state.config.symbol, close_side, pos.amount)

    trade = state.risk_manager.close_trade(price, "sinal_estrategia")

    trade_entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": "VENDA" if close_side == "sell" else "COMPRA (FECHA SHORT)",
        "symbol": state.config.symbol,
        "price": price,
        "amount": round(pos.amount, 6),
        "pnl": trade.pnl,
    }
    state.trade_history.append(trade_entry)
    await state.broadcast({"type": "trade", "trade": trade_entry})
    # Atualiza saldo imediatamente no frontend
    if state.paper_trader:
        await state.broadcast({
            "type": "balance_update",
            "paper_summary": state.paper_trader.get_summary(price),
            "open_position": _get_position_dict(),
            "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
            "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
        })


async def _close_position(price: float, reason: str):
    if not state.risk_manager or not state.config:
        return
    if state.risk_manager.open_position is None:
        return

    pos = state.risk_manager.open_position

    # Lado oposto para fechar: LONG fecha com sell, SHORT fecha com buy
    close_side = "buy" if pos.side == "sell" else "sell"

    if state.paper_trader:
        state.paper_trader.execute_order(state.config.symbol, close_side, pos.amount, price)
    elif state.exchange_client:
        await state.exchange_client.create_market_order(state.config.symbol, close_side, pos.amount)

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
    # Atualiza saldo imediatamente no frontend
    if state.paper_trader:
        await state.broadcast({
            "type": "balance_update",
            "paper_summary": state.paper_trader.get_summary(price),
            "open_position": _get_position_dict(),
            "daily_pnl": state.risk_manager.daily_pnl if state.risk_manager else 0,
            "daily_trades": state.risk_manager.daily_trades if state.risk_manager else 0,
        })


def _get_position_dict() -> dict | None:
    if not state.risk_manager or not state.risk_manager.open_position:
        return None
    pos = state.risk_manager.open_position
    if pos.closed:
        return None

    # Calcular PnL não realizado
    unrealized_pnl = 0.0
    current_value = 0.0
    cost = pos.entry_price * pos.amount
    if state.current_price > 0:
        current_value = state.current_price * pos.amount
        if pos.side.lower() == "buy":
            unrealized_pnl = (state.current_price - pos.entry_price) * pos.amount
        else:
            unrealized_pnl = (pos.entry_price - state.current_price) * pos.amount

    result = {
        "symbol": pos.symbol,
        "side": pos.side,
        "entry_price": pos.entry_price,
        "amount": round(pos.amount, 6),
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
        "unrealized_pnl": round(unrealized_pnl, 2),
        "cost": round(cost, 2),
        "current_value": round(current_value, 2),
        "pnl_pct": round((unrealized_pnl / cost) * 100, 2) if cost > 0 else 0,
    }

    # Info extra no scalping
    if state.scalping_mode and state.config:
        result["scalping_tp_usd"] = state.config.scalping_take_profit_usd
        result["scalping_sl_usd"] = state.config.scalping_stop_loss_usd
        result["scalping_progress"] = round(
            (unrealized_pnl / state.config.scalping_take_profit_usd) * 100, 1
        ) if state.config.scalping_take_profit_usd > 0 else 0

    return result
