"""
TradePulse - Módulo de Gestão de Risco
Controle de stop-loss, take-profit, tamanho de posição e limites diários
"""
import logging
from datetime import date
from dataclasses import dataclass, field
from time import monotonic

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
    - Suporte a modo scalping com TP/SL em valor fixo (USDT)
    """
    config: Config
    daily_pnl: float = 0.0
    daily_trades: int = 0
    current_date: date = field(default_factory=date.today)
    open_position: TradeRecord | None = None
    scalping_mode: bool = False
    _cooldown_until: float = 0.0  # monotonic time until which trading is blocked
    _consecutive_losses: int = 0

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

        # Cooldown após stop-loss (evita re-entrada imediata em mercado ruim)
        if self.scalping_mode and monotonic() < self._cooldown_until:
            remaining = int(self._cooldown_until - monotonic())
            return False, f"Cooldown ativo após stop-loss ({remaining}s restantes)"

        # Limite de perda diária atingido?
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return False, (
                f"Limite de perda diária atingido: {self.daily_pnl:.2f} USDT "
                f"(limite: -{self.config.daily_loss_limit:.2f} USDT)"
            )

        # Limite de trades diários atingido?
        max_trades = self.config.scalping_max_trades if self.scalping_mode else self.config.max_trades_per_day
        if self.daily_trades >= max_trades:
            return False, (
                f"Limite diário de operações atingido: {self.daily_trades}/{max_trades}"
            )

        return True, "Operação permitida"

    def calculate_position_size(
        self, balance: float, entry_price: float
    ) -> float:
        """
        Calcula o tamanho da posição baseado no risco máximo.

        No modo scalping, usa risk_per_trade maior e calcula baseado no
        stop-loss fixo em USDT para garantir que o TP de $5 seja alcançável.

        Fórmula padrão:
          risco_valor = saldo * max_risk_per_trade
          stop_distance = entry_price * stop_loss_pct
          quantidade = risco_valor / stop_distance

        Fórmula scalping:
          risco_valor = saldo * scalping_risk_per_trade
          quantidade = risco_valor / entry_price
        """
        if self.scalping_mode:
            risk_amount = balance * self.config.scalping_risk_per_trade
            position_size = risk_amount / entry_price
            logger.info(
                "Scalping posição: saldo=%.2f, risco=%.2f USDT, tamanho=%.6f, "
                "TP=$%.2f, SL=$%.2f",
                balance, risk_amount, position_size,
                self.config.scalping_take_profit_usd,
                self.config.scalping_stop_loss_usd,
            )
            return position_size

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

    def calculate_stop_loss(self, entry_price: float, side: str, amount: float = 0.0) -> float:
        """Calcula preço de stop-loss. No scalping, usa valor fixo em USDT."""
        if self.scalping_mode and amount > 0:
            # SL fixo: perda máxima em USDT / quantidade = distância de preço
            sl_distance = self.config.scalping_stop_loss_usd / amount
            if side.lower() == "buy":
                sl = entry_price - sl_distance
            else:
                sl = entry_price + sl_distance
            return round(sl, 2)

        if side.lower() == "buy":
            sl = entry_price * (1 - self.config.stop_loss_pct)
        else:
            sl = entry_price * (1 + self.config.stop_loss_pct)
        return round(sl, 2)

    def calculate_take_profit(self, entry_price: float, side: str, amount: float = 0.0) -> float:
        """Calcula preço de take-profit. No scalping, usa valor fixo em USDT ($5)."""
        if self.scalping_mode and amount > 0:
            # TP fixo: lucro alvo em USDT / quantidade = distância de preço
            tp_distance = self.config.scalping_take_profit_usd / amount
            if side.lower() == "buy":
                tp = entry_price + tp_distance
            else:
                tp = entry_price - tp_distance
            return round(tp, 2)

        if side.lower() == "buy":
            tp = entry_price * (1 + self.config.take_profit_pct)
        else:
            tp = entry_price * (1 - self.config.take_profit_pct)
        return round(tp, 2)

    def open_trade(
        self, symbol: str, side: str, entry_price: float, amount: float
    ) -> TradeRecord:
        """Registra abertura de uma operação."""
        sl = self.calculate_stop_loss(entry_price, side, amount)
        tp = self.calculate_take_profit(entry_price, side, amount)

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

        No modo scalping, verifica o PnL em USDT diretamente
        (independente dos preços SL/TP da abertura) para garantir
        que a saída aconteça com $5 de lucro ou $3 de perda.
        """
        if self.open_position is None or self.open_position.closed:
            return None

        pos = self.open_position

        # Scalping: verificar PnL em dólares diretamente
        if self.scalping_mode:
            if pos.side.lower() == "buy":
                pnl_usd = (current_price - pos.entry_price) * pos.amount
            else:
                pnl_usd = (pos.entry_price - current_price) * pos.amount

            if pnl_usd >= self.config.scalping_take_profit_usd:
                logger.info(
                    "SCALP TP atingido! PnL=%.2f USDT (alvo: %.2f) | Preço: %.2f → %.2f",
                    pnl_usd, self.config.scalping_take_profit_usd,
                    pos.entry_price, current_price,
                )
                return "take_profit"
            if pnl_usd <= -self.config.scalping_stop_loss_usd:
                logger.info(
                    "SCALP SL atingido! PnL=%.2f USDT (limite: -%.2f) | Preço: %.2f → %.2f",
                    pnl_usd, self.config.scalping_stop_loss_usd,
                    pos.entry_price, current_price,
                )
                return "stop_loss"
            return None

        # Modo padrão: verificar preços de SL/TP
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

        # Cooldown após stop-loss no scalping
        if self.scalping_mode and reason == "stop_loss":
            self._consecutive_losses += 1
            # Cooldown progressivo: mais perdas seguidas = mais tempo de espera
            cooldown_multiplier = min(self._consecutive_losses, 5)
            cooldown_secs = self.config.scalping_cooldown_after_sl * self.config.scalping_check_interval * cooldown_multiplier
            self._cooldown_until = monotonic() + cooldown_secs
            logger.info(
                "Cooldown ativado: %ds (perdas consecutivas: %d)",
                cooldown_secs, self._consecutive_losses,
            )
        elif reason == "take_profit":
            self._consecutive_losses = 0  # Reset ao lucrar

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
