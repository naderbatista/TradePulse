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
    """

    def __init__(self, config: Config, initial_balance: float | None = None):
        self.config = config
        self.balance = initial_balance or config.bt_initial_balance
        self.initial_balance = self.balance
        self.trades: list[dict] = []
        self.risk_manager = RiskManager(config=config)

    def get_balance(self) -> float:
        return self.balance

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict:
        """Simula a execução de uma ordem."""
        cost = amount * price
        order_id = f"paper_{len(self.trades) + 1}_{int(datetime.now(timezone.utc).timestamp())}"

        if side.lower() == "buy" and cost > self.balance:
            logger.warning(
                "Paper: Saldo insuficiente para compra. Necessário: %.2f, Disponível: %.2f",
                cost,
                self.balance,
            )
            raise ValueError("Saldo insuficiente para operação")

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
        }
        self.trades.append(trade_record)

        if side.lower() == "buy":
            self.balance -= cost
        else:
            self.balance += cost

        logger.info(
            "Paper %s: %s %.6f @ %.2f | Custo=%.2f | Saldo=%.2f",
            side.upper(),
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
        if current_price > 0:
            # Calcula valor da posição aberta (compras sem vendas correspondentes)
            buy_amount = sum(t["amount"] for t in self.trades if t["side"] == "buy")
            sell_amount = sum(t["amount"] for t in self.trades if t["side"] == "sell")
            open_amount = buy_amount - sell_amount
            if open_amount > 0:
                unrealized = open_amount * current_price
        return round(self.balance + unrealized - self.initial_balance, 2)

    def get_summary(self, current_price: float = 0.0) -> dict:
        """Retorna resumo do paper trading."""
        total_trades = len(self.trades)
        buys = sum(1 for t in self.trades if t["side"] == "buy")
        sells = total_trades - buys
        pnl = self.get_pnl(current_price)
        equity = self.balance
        # Se houver posição aberta, equity inclui o valor dela
        if current_price > 0:
            buy_amount = sum(t["amount"] for t in self.trades if t["side"] == "buy")
            sell_amount = sum(t["amount"] for t in self.trades if t["side"] == "sell")
            open_amount = buy_amount - sell_amount
            if open_amount > 0:
                equity += open_amount * current_price
        return {
            "saldo_inicial": self.initial_balance,
            "saldo_atual": round(equity, 2),
            "pnl_total": pnl,
            "retorno_pct": round((pnl / self.initial_balance) * 100, 2) if self.initial_balance else 0,
            "total_operacoes": total_trades,
            "compras": buys,
            "vendas": sells,
        }
