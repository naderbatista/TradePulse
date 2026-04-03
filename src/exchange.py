"""
TradePulse - Módulo de Exchange
Integração assíncrona com Binance e Bybit via ccxt
"""
import asyncio
import logging
import sys
from typing import Any

import aiohttp
import ccxt.async_support as ccxt

from .config import Config

logger = logging.getLogger("tradepulse")

# Exchanges suportadas
SUPPORTED_EXCHANGES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
}


class ExchangeClient:
    """Cliente assíncrono para interação com exchanges de criptomoedas."""

    def __init__(self, config: Config, exchange_name: str | None = None, sandbox: bool | None = None):
        self.config = config
        self.exchange_name = (exchange_name or config.exchange).lower()
        self._exchange: ccxt.Exchange | None = None
        # sandbox pode ser forçado para False (ex: backtest usa dados públicos reais)
        self._sandbox = sandbox if sandbox is not None else (config.trading_mode == "paper")

    async def connect(self) -> None:
        """Inicializa a conexão com a exchange."""
        if self.exchange_name not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Exchange '{self.exchange_name}' não suportada. "
                f"Opções: {list(SUPPORTED_EXCHANGES.keys())}"
            )

        exchange_class = SUPPORTED_EXCHANGES[self.exchange_name]
        keys = self.config.get_api_keys(self.exchange_name)

        options: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        # Só envia chaves se estiverem preenchidas (modo paper pode não ter)
        if keys["apiKey"] and keys["secret"]:
            options["apiKey"] = keys["apiKey"]
            options["secret"] = keys["secret"]

        self._exchange = exchange_class(options)

        # No Windows, força resolver DNS compatível (ThreadedResolver)
        # para evitar problemas com ProactorEventLoop do uvicorn
        if sys.platform == "win32":
            resolver = aiohttp.resolver.ThreadedResolver()
            connector = aiohttp.TCPConnector(resolver=resolver)
            session = aiohttp.ClientSession(connector=connector)
            self._exchange.session = session

        logger.info(
            "Conectado à exchange %s (sandbox=%s)",
            self.exchange_name,
            self._sandbox,
        )

        # Ativa sandbox apenas se solicitado (não para backtest)
        if self._sandbox and hasattr(self._exchange, "set_sandbox_mode"):
            self._exchange.set_sandbox_mode(True)

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            raise RuntimeError("Exchange não conectada. Chame connect() primeiro.")
        return self._exchange

    async def close(self) -> None:
        """Fecha a conexão com a exchange."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            logger.info("Conexão com %s encerrada.", self.exchange_name)

    # ---- Dados de Mercado ----

    async def fetch_ohlcv(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> list[list]:
        """Busca candles OHLCV."""
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        limit = limit or self.config.candle_limit
        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug("Obtidas %d velas de %s (%s)", len(data), symbol, timeframe)
            return data
        except ccxt.BaseError as e:
            logger.error("Erro ao buscar OHLCV: %s", e)
            raise

    async def fetch_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Busca preço atual do ticker."""
        symbol = symbol or self.config.symbol
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.BaseError as e:
            logger.error("Erro ao buscar ticker: %s", e)
            raise

    async def fetch_order_book(
        self, symbol: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """Busca o livro de ofertas."""
        symbol = symbol or self.config.symbol
        try:
            book = await self.exchange.fetch_order_book(symbol, limit=limit)
            return book
        except ccxt.BaseError as e:
            logger.error("Erro ao buscar order book: %s", e)
            raise

    # ---- Saldo ----

    async def fetch_balance(self) -> dict[str, Any]:
        """Busca saldo da conta."""
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except ccxt.BaseError as e:
            logger.error("Erro ao buscar saldo: %s", e)
            raise

    async def get_free_balance(self, currency: str = "USDT") -> float:
        """Retorna saldo livre de uma moeda específica."""
        balance = await self.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0.0))

    # ---- Ordens ----

    async def create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> dict[str, Any]:
        """Cria uma ordem a mercado."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
            )
            logger.info(
                "Ordem MARKET %s executada: %s %.6f @ mercado (id=%s)",
                side.upper(),
                symbol,
                amount,
                order.get("id"),
            )
            return order
        except ccxt.BaseError as e:
            logger.error("Erro ao criar ordem market: %s", e)
            raise

    async def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> dict[str, Any]:
        """Cria uma ordem limitada."""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price,
            )
            logger.info(
                "Ordem LIMIT %s criada: %s %.6f @ %.2f (id=%s)",
                side.upper(),
                symbol,
                amount,
                price,
                order.get("id"),
            )
            return order
        except ccxt.BaseError as e:
            logger.error("Erro ao criar ordem limit: %s", e)
            raise

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Cancela uma ordem."""
        symbol = symbol or self.config.symbol
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info("Ordem %s cancelada.", order_id)
            return result
        except ccxt.BaseError as e:
            logger.error("Erro ao cancelar ordem %s: %s", order_id, e)
            raise

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Busca ordens abertas."""
        symbol = symbol or self.config.symbol
        try:
            return await self.exchange.fetch_open_orders(symbol)
        except ccxt.BaseError as e:
            logger.error("Erro ao buscar ordens abertas: %s", e)
            raise

    # ---- Utilitários ----

    async def switch_exchange(self, new_exchange: str) -> None:
        """Troca para outra exchange dinamicamente."""
        logger.info("Trocando exchange de %s para %s...", self.exchange_name, new_exchange)
        await self.close()
        self.exchange_name = new_exchange.lower()
        await self.connect()
