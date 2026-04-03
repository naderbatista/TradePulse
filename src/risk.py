"""
TradePulse - Módulo de Gestão de Risco
Controle de stop-loss, take-profit, tamanho de posição e limites diários
"""
import logging
from datetime import date
from dataclasses import dataclass, field

from .config import Config

logger = logging.getLogger("tradepulse")


@dataclass
class TradeRecord:
    """Registro de uma operação realizada."""
    symbol: str
    side: str
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    closed: bool = False


@dataclass
class RiskManager:
    """
    Gerenciador de risco para o bot de trading.

    Responsabilidades:
    - Calcular tamanho da posição baseado no risco máximo por operação
    - Definir stop-loss e take-profit
    - Controlar limite de perda diária
    - Limitar número de operações por dia
    - Impedir operações duplicadas
    """
    config: Config
    daily_pnl: float = 0.0
    daily_trades: int = 0
    current_date: date = field(default_factory=date.today)
    open_position: TradeRecord | None = None

    def _reset_daily_if_needed(self) -> None:
        """Reseta contadores se o dia mudou."""
        today = date.today()
        if today != self.current_date:
            logger.info(
                "Novo dia detectado. Resetando contadores diários. "
                "PnL anterior: %.2f USDT, Trades: %d",
                self.daily_pnl,
                self.daily_trades,
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today

    def can_trade(self) -> tuple[bool, str]:
        """Verifica se o bot pode abrir uma nova operação."""
        self._reset_daily_if_needed()

        # Já tem posição aberta?
        if self.open_position is not None and not self.open_position.closed:
            return False, "Já existe uma posição aberta (prevenção de duplicatas)"

        # Limite de perda diária atingido?
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return False, (
                f"Limite de perda diária atingido: {self.daily_pnl:.2f} USDT "
                f"(limite: -{self.config.daily_loss_limit:.2f} USDT)"
            )

        # Limite de trades diários atingido?
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, (
                f"Limite diário de operações atingido: {self.daily_trades}/{self.config.max_trades_per_day}"
            )

        return True, "Operação permitida"

    def calculate_position_size(
        self, balance: float, entry_price: float
    ) -> float:
        """
        Calcula o tamanho da posição baseado no risco máximo.

        Fórmula:
          risco_valor = saldo * max_risk_per_trade
          stop_distance = entry_price * stop_loss_pct
          quantidade = risco_valor / stop_distance
        """
        risk_amount = balance * self.config.max_risk_per_trade
        stop_distance = entry_price * self.config.stop_loss_pct

        if stop_distance <= 0:
            logger.warning("Stop distance inválido. Retornando posição zero.")
            return 0.0

        position_size = risk_amount / stop_distance

        logger.info(
            "Cálculo de posição: saldo=%.2f, risco=%.2f USDT, "
            "stop_distance=%.2f, tamanho=%.6f",
            balance,
            risk_amount,
            stop_distance,
            position_size,
        )
        return position_size

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calcula preço de stop-loss."""
        if side.lower() == "buy":
            sl = entry_price * (1 - self.config.stop_loss_pct)
        else:
            sl = entry_price * (1 + self.config.stop_loss_pct)
        return round(sl, 2)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calcula preço de take-profit."""
        if side.lower() == "buy":
            tp = entry_price * (1 + self.config.take_profit_pct)
        else:
            tp = entry_price * (1 - self.config.take_profit_pct)
        return round(tp, 2)

    def open_trade(
        self, symbol: str, side: str, entry_price: float, amount: float
    ) -> TradeRecord:
        """Registra abertura de uma operação."""
        sl = self.calculate_stop_loss(entry_price, side)
        tp = self.calculate_take_profit(entry_price, side)

        trade = TradeRecord(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            stop_loss=sl,
            take_profit=tp,
        )
        self.open_position = trade
        self.daily_trades += 1

        logger.info(
            "Operação aberta: %s %s %.6f @ %.2f | SL=%.2f | TP=%.2f",
            side.upper(),
            symbol,
            amount,
            entry_price,
            sl,
            tp,
        )
        return trade

    def check_exit_conditions(self, current_price: float) -> str | None:
        """
        Verifica se a posição aberta atingiu stop-loss ou take-profit.
        Retorna 'stop_loss', 'take_profit' ou None.
        """
        if self.open_position is None or self.open_position.closed:
            return None

        pos = self.open_position

        if pos.side.lower() == "buy":
            if current_price <= pos.stop_loss:
                return "stop_loss"
            if current_price >= pos.take_profit:
                return "take_profit"
        else:  # sell / short
            if current_price >= pos.stop_loss:
                return "stop_loss"
            if current_price <= pos.take_profit:
                return "take_profit"

        return None

    def close_trade(self, exit_price: float, reason: str) -> TradeRecord:
        """Fecha a posição aberta e calcula PnL."""
        if self.open_position is None or self.open_position.closed:
            raise RuntimeError("Nenhuma posição aberta para fechar.")

        pos = self.open_position

        if pos.side.lower() == "buy":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        pos.pnl = round(pnl, 2)
        pos.closed = True
        self.daily_pnl += pnl

        logger.info(
            "Operação fechada (%s): %s %s | Entrada=%.2f | Saída=%.2f | "
            "PnL=%.2f USDT | PnL diário=%.2f USDT",
            reason,
            pos.side.upper(),
            pos.symbol,
            pos.entry_price,
            exit_price,
            pnl,
            self.daily_pnl,
        )
        return pos
