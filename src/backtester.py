"""
TradePulse - Módulo de Backtesting
Simula a estratégia em dados históricos
"""
import logging

import pandas as pd

from .config import Config
from .strategy import MACrossoverRSI, Signal

logger = logging.getLogger("tradepulse")


class Backtester:
    """
    Motor de backtesting para validar estratégias em dados históricos.
    Simula operações e calcula métricas de desempenho.
    """

    def __init__(self, config: Config):
        self.config = config
        self.strategy = MACrossoverRSI(config)
        self.initial_balance = config.bt_initial_balance
        self.commission = config.bt_commission

    def run(self, ohlcv_data: list[list]) -> dict:
        """
        Executa o backtest completo.

        Args:
            ohlcv_data: Dados OHLCV brutos (lista de listas).

        Returns:
            Dicionário com métricas de desempenho.
        """
        df = self.strategy.prepare_dataframe(ohlcv_data)

        balance = self.initial_balance
        position: dict | None = None
        trades: list[dict] = []
        equity_curve: list[float] = []

        logger.info(
            "Iniciando backtest: saldo=%.2f, velas=%d, par=%s",
            balance,
            len(df),
            self.config.symbol,
        )

        for i in range(1, len(df)):
            window = df.iloc[: i + 1]
            current = df.iloc[i]
            price = current["close"]
            timestamp = current.name

            signal, reasons = self.strategy.generate_signal(window)

            # Verifica stop-loss / take-profit na posição aberta
            if position is not None:
                if position["side"] == "buy":
                    if price <= position["stop_loss"]:
                        pnl = (price - position["entry_price"]) * position["amount"]
                        pnl -= position["entry_price"] * position["amount"] * self.commission
                        pnl -= price * position["amount"] * self.commission
                        balance += price * position["amount"] + pnl
                        trades.append({
                            "tipo": "STOP_LOSS",
                            "entrada": position["entry_price"],
                            "saida": price,
                            "pnl": round(pnl, 2),
                            "timestamp": str(timestamp),
                        })
                        position = None
                    elif price >= position["take_profit"]:
                        pnl = (price - position["entry_price"]) * position["amount"]
                        pnl -= position["entry_price"] * position["amount"] * self.commission
                        pnl -= price * position["amount"] * self.commission
                        balance += price * position["amount"] + pnl
                        trades.append({
                            "tipo": "TAKE_PROFIT",
                            "entrada": position["entry_price"],
                            "saida": price,
                            "pnl": round(pnl, 2),
                            "timestamp": str(timestamp),
                        })
                        position = None

            # Sinal de compra (sem posição aberta)
            if signal == Signal.BUY and position is None:
                risk_amount = balance * self.config.max_risk_per_trade
                stop_distance = price * self.config.stop_loss_pct
                if stop_distance > 0:
                    amount = risk_amount / stop_distance
                    cost = amount * price
                    if cost <= balance:
                        balance -= cost
                        sl = round(price * (1 - self.config.stop_loss_pct), 2)
                        tp = round(price * (1 + self.config.take_profit_pct), 2)
                        position = {
                            "side": "buy",
                            "entry_price": price,
                            "amount": amount,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "timestamp": str(timestamp),
                        }

            # Sinal de venda (fecha posição de compra)
            elif signal == Signal.SELL and position is not None and position["side"] == "buy":
                pnl = (price - position["entry_price"]) * position["amount"]
                pnl -= position["entry_price"] * position["amount"] * self.commission
                pnl -= price * position["amount"] * self.commission
                balance += price * position["amount"] + pnl
                trades.append({
                    "tipo": "SINAL_VENDA",
                    "entrada": position["entry_price"],
                    "saida": price,
                    "pnl": round(pnl, 2),
                    "timestamp": str(timestamp),
                })
                position = None

            equity_curve.append(balance if position is None else balance + position["amount"] * price)

        # Fecha posição aberta no final
        if position is not None:
            final_price = df.iloc[-1]["close"]
            pnl = (final_price - position["entry_price"]) * position["amount"]
            balance += final_price * position["amount"] + pnl
            trades.append({
                "tipo": "FECHAMENTO_FINAL",
                "entrada": position["entry_price"],
                "saida": final_price,
                "pnl": round(pnl, 2),
                "timestamp": str(df.index[-1]),
            })

        return self._calculate_metrics(trades, equity_curve, balance)

    def _calculate_metrics(
        self,
        trades: list[dict],
        equity_curve: list[float],
        final_balance: float,
    ) -> dict:
        """Calcula métricas de desempenho do backtest."""
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "saldo_final": round(final_balance, 2),
                "pnl_total": 0.0,
                "retorno_pct": 0.0,
                "total_operacoes": 0,
                "operacoes_positivas": 0,
                "operacoes_negativas": 0,
                "taxa_acerto": 0.0,
                "maior_lucro": 0.0,
                "maior_prejuizo": 0.0,
                "max_drawdown_pct": 0.0,
                "operacoes": [],
            }

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max Drawdown
        max_dd = 0.0
        if equity_curve:
            peak = equity_curve[0]
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

        pnl_total = round(final_balance - self.initial_balance, 2)

        metrics = {
            "saldo_inicial": self.initial_balance,
            "saldo_final": round(final_balance, 2),
            "pnl_total": pnl_total,
            "retorno_pct": round((pnl_total / self.initial_balance) * 100, 2),
            "total_operacoes": total_trades,
            "operacoes_positivas": len(wins),
            "operacoes_negativas": len(losses),
            "taxa_acerto": round((len(wins) / total_trades) * 100, 2) if total_trades else 0,
            "maior_lucro": round(max(pnls), 2) if pnls else 0,
            "maior_prejuizo": round(min(pnls), 2) if pnls else 0,
            "lucro_medio": round(sum(wins) / len(wins), 2) if wins else 0,
            "prejuizo_medio": round(sum(losses) / len(losses), 2) if losses else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "operacoes": trades,
        }

        logger.info("=== Resultado do Backtest ===")
        for k, v in metrics.items():
            if k != "operacoes":
                logger.info("  %s: %s", k, v)

        return metrics
