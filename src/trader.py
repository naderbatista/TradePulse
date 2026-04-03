"""
TradePulse - Módulo de Trading Principal
Orquestra a estratégia, exchange, risco e execução
"""
import asyncio
import logging

from .config import Config
from .exchange import ExchangeClient
from .strategy import MACrossoverRSI, Signal
from .risk import RiskManager
from .paper_trading import PaperTrader
from .logger import log_decision

logger = logging.getLogger("tradepulse")

# Máximo de tentativas em caso de erro de API
MAX_RETRIES = 3
RETRY_DELAY = 5  # segundos


class Trader:
    """
    Motor de trading principal.
    Conecta estratégia, exchange e gestão de risco.
    """

    def __init__(self, config: Config):
        self.config = config
        self.exchange_client = ExchangeClient(config)
        self.strategy = MACrossoverRSI(config)
        self.risk_manager = RiskManager(config=config)
        self.paper_trader: PaperTrader | None = None
        self.running = False

        if config.trading_mode == "paper":
            self.paper_trader = PaperTrader(config)
            logger.info("Modo PAPER TRADING ativado.")
        else:
            logger.info("Modo LIVE TRADING ativado. CUIDADO: operações reais!")

    async def start(self) -> None:
        """Inicia o loop principal do bot."""
        await self.exchange_client.connect()
        self.running = True

        logger.info(
            "Bot iniciado | Exchange=%s | Par=%s | Timeframe=%s | Modo=%s",
            self.config.exchange,
            self.config.symbol,
            self.config.timeframe,
            self.config.trading_mode,
        )

        try:
            while self.running:
                await self._trading_cycle()
                logger.debug(
                    "Aguardando %d segundos até próxima verificação...",
                    self.config.check_interval,
                )
                await asyncio.sleep(self.config.check_interval)
        except asyncio.CancelledError:
            logger.info("Bot cancelado pelo usuário.")
        except KeyboardInterrupt:
            logger.info("Bot interrompido (Ctrl+C).")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Para o bot e fecha conexões."""
        self.running = False
        await self.exchange_client.close()

        if self.paper_trader:
            summary = self.paper_trader.get_summary()
            logger.info("=== Resumo Paper Trading ===")
            for k, v in summary.items():
                logger.info("  %s: %s", k, v)

        logger.info("Bot encerrado.")

    async def _trading_cycle(self) -> None:
        """Um ciclo completo de análise e possível execução."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 1. Buscar dados de mercado
                ohlcv = await self.exchange_client.fetch_ohlcv()
                ticker = await self.exchange_client.fetch_ticker()
                current_price = float(ticker.get("last", 0))

                if current_price <= 0:
                    logger.warning("Preço inválido recebido: %s", current_price)
                    return

                # 2. Verificar saída de posição (stop-loss / take-profit)
                exit_reason = self.risk_manager.check_exit_conditions(current_price)
                if exit_reason:
                    await self._close_position(current_price, exit_reason)
                    return

                # 3. Analisar estratégia
                signal, reasons = self.strategy.analyze(ohlcv)

                log_decision(
                    logger,
                    action=signal.value,
                    reason=reasons.get("sinal", "Análise concluída"),
                    data=reasons,
                )

                # 4. Executar se houver sinal
                if signal == Signal.BUY:
                    await self._execute_buy(current_price)
                elif signal == Signal.SELL:
                    await self._execute_sell(current_price)

                return  # Sucesso, sai do loop de retry

            except Exception as e:
                logger.error(
                    "Erro no ciclo de trading (tentativa %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error("Máximo de tentativas atingido. Pulando ciclo.")

    async def _execute_buy(self, price: float) -> None:
        """Executa uma operação de compra."""
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            log_decision(logger, "COMPRA BLOQUEADA", reason)
            return

        # Obter saldo
        if self.paper_trader:
            balance = self.paper_trader.get_balance()
        else:
            balance = await self.exchange_client.get_free_balance("USDT")

        # Calcular tamanho da posição
        amount = self.risk_manager.calculate_position_size(balance, price)
        if amount <= 0:
            log_decision(logger, "COMPRA CANCELADA", "Tamanho de posição inválido")
            return

        # Verificar se o custo não excede o saldo
        cost = amount * price
        if cost > balance:
            amount = (balance * 0.99) / price  # Usa 99% do saldo para margem
            if amount <= 0:
                log_decision(logger, "COMPRA CANCELADA", "Saldo insuficiente")
                return

        # Executar ordem
        if self.paper_trader:
            self.paper_trader.execute_order(self.config.symbol, "buy", amount, price)
        else:
            if self.config.order_type == "limit":
                await self.exchange_client.create_limit_order(
                    self.config.symbol, "buy", amount, price
                )
            else:
                await self.exchange_client.create_market_order(
                    self.config.symbol, "buy", amount
                )

        # Registrar no risk manager
        self.risk_manager.open_trade(self.config.symbol, "buy", price, amount)

        log_decision(
            logger,
            "COMPRA EXECUTADA",
            f"Compra de {amount:.6f} {self.config.symbol} @ {price:.2f}",
            {"amount": amount, "price": price, "cost": round(cost, 2)},
        )

    async def _execute_sell(self, price: float) -> None:
        """Executa uma operação de venda (fecha posição)."""
        if self.risk_manager.open_position is None or self.risk_manager.open_position.closed:
            log_decision(logger, "VENDA IGNORADA", "Nenhuma posição aberta para fechar")
            return

        pos = self.risk_manager.open_position

        # Executar ordem de venda
        if self.paper_trader:
            self.paper_trader.execute_order(self.config.symbol, "sell", pos.amount, price)
        else:
            if self.config.order_type == "limit":
                await self.exchange_client.create_limit_order(
                    self.config.symbol, "sell", pos.amount, price
                )
            else:
                await self.exchange_client.create_market_order(
                    self.config.symbol, "sell", pos.amount
                )

        # Fechar no risk manager
        trade = self.risk_manager.close_trade(price, "sinal_estrategia")

        log_decision(
            logger,
            "VENDA EXECUTADA",
            f"Venda de {pos.amount:.6f} {self.config.symbol} @ {price:.2f} | PnL={trade.pnl:.2f}",
            {"amount": pos.amount, "price": price, "pnl": trade.pnl},
        )

    async def _close_position(self, price: float, reason: str) -> None:
        """Fecha posição por stop-loss ou take-profit."""
        if self.risk_manager.open_position is None:
            return

        pos = self.risk_manager.open_position

        if self.paper_trader:
            self.paper_trader.execute_order(self.config.symbol, "sell", pos.amount, price)
        else:
            await self.exchange_client.create_market_order(
                self.config.symbol, "sell", pos.amount
            )

        trade = self.risk_manager.close_trade(price, reason)

        log_decision(
            logger,
            f"POSIÇÃO FECHADA ({reason.upper()})",
            f"{reason}: PnL={trade.pnl:.2f} USDT",
            {"reason": reason, "price": price, "pnl": trade.pnl},
        )
