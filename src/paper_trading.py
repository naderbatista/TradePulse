"""
TradePulse - Módulo de Paper Trading
Simula operações sem usar dinheiro real
"""
import logging
from datetime import datetime, timezone

from .config import Config
from .risk import RiskManager, TradeRecord

logger = logging.getLogger("tradepulse")


class PaperTrader:
    """
    Simulador de trading que imita o comportamento de uma exchange real.
    Mantém saldo virtual e histórico de operações.
    Modelo de margem: abrir qualquer posição (LONG ou SHORT) deduz do saldo.
    """

    def __init__(self, config: Config, initial_balance: float | None = None):
        self.config = config
        self.balance = initial_balance or config.bt_initial_balance
        self.initial_balance = self.balance
        self.trades: list[dict] = []
        self.risk_manager = RiskManager(config=config)
        # Posição aberta atual (para cálculo correto de PnL)
        self._open_entry_price: float = 0.0
        self._open_side: str = ""
        self._open_amount: float = 0.0

    def get_balance(self) -> float:
        return self.balance

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        opening: bool = True,
    ) -> dict:
        """
        Simula a execução de uma ordem.
        
        opening=True:  Abrindo posição → deduz margem do saldo
        opening=False: Fechando posição → devolve margem ± PnL
        """
        cost = amount * price
        order_id = f"paper_{len(self.trades) + 1}_{int(datetime.now(timezone.utc).timestamp())}"

        if opening:
            if cost > self.balance:
                logger.warning(
                    "Paper: Saldo insuficiente. Necessário: %.2f, Disponível: %.2f",
                    cost,
                    self.balance,
                )
                raise ValueError("Saldo insuficiente para operação")
            self.balance -= cost
            self._open_entry_price = price
            self._open_side = side.lower()
            self._open_amount = amount
        else:
            # Fechando posição: devolve margem + PnL
            if self._open_side == "buy":
                # Fechando LONG: recebe valor atual do ativo
                self.balance += amount * price
            else:
                # Fechando SHORT: margem + lucro/prejuízo
                pnl = (self._open_entry_price - price) * amount
                self.balance += (self._open_amount * self._open_entry_price) + pnl
            self._open_entry_price = 0.0
            self._open_side = ""
            self._open_amount = 0.0

        trade_record = {
            "id": order_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side.lower(),
            "type": "market",
            "amount": amount,
            "price": price,
            "cost": cost,
            "status": "filled",
            "opening": opening,
        }
        self.trades.append(trade_record)

        logger.info(
            "Paper %s (%s): %s %.6f @ %.2f | Custo=%.2f | Saldo=%.2f",
            side.upper(),
            "ABRE" if opening else "FECHA",
            symbol,
            amount,
            price,
            cost,
            self.balance,
        )
        return trade_record

    def get_pnl(self, current_price: float = 0.0) -> float:
        """
        Retorna o PnL total desde o início.
        Se houver posição aberta, inclui o valor não realizado.
        """
        unrealized = 0.0
        if current_price > 0 and self._open_amount > 0:
            if self._open_side == "buy":
                # LONG: valor atual do ativo
                unrealized = self._open_amount * current_price
            else:
                # SHORT: margem + PnL não realizado
                unrealized = self._open_amount * (2 * self._open_entry_price - current_price)
        return round(self.balance + unrealized - self.initial_balance, 2)

    def get_summary(self, current_price: float = 0.0) -> dict:
        """Retorna resumo do paper trading."""
        total_trades = len(self.trades)
        buys = sum(1 for t in self.trades if t["side"] == "buy")
        sells = total_trades - buys
        pnl = self.get_pnl(current_price)
        equity = self.balance
        if current_price > 0 and self._open_amount > 0:
            if self._open_side == "buy":
                equity += self._open_amount * current_price
            else:
                equity += self._open_amount * (2 * self._open_entry_price - current_price)
        return {
            "saldo_inicial": self.initial_balance,
            "saldo_atual": round(equity, 2),
            "pnl_total": pnl,
            "retorno_pct": round((pnl / self.initial_balance) * 100, 2) if self.initial_balance else 0,
            "total_operacoes": total_trades,
            "compras": buys,
            "vendas": sells,
        }
