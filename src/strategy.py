"""
TradePulse - Módulo de Estratégia
Cruzamento de Médias Móveis com confirmação RSI
"""
import logging
from enum import Enum

import numpy as np
import pandas as pd
import ta

from .config import Config

logger = logging.getLogger("tradepulse")


class Signal(Enum):
    """Sinais possíveis da estratégia."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MACrossoverRSI:
    """
    Estratégia: Cruzamento de Médias Móveis + RSI

    Regras de COMPRA:
      - MA curta cruza acima da MA longa (golden cross)
      - RSI está abaixo do nível de sobrevenda (confirmação de força)

    Regras de VENDA:
      - MA curta cruza abaixo da MA longa (death cross)
      - RSI está acima do nível de sobrecompra (confirmação de fraqueza)
    """

    def __init__(self, config: Config):
        self.short_period = config.short_ma_period
        self.long_period = config.long_ma_period
        self.ma_type = config.ma_type.lower()
        self.rsi_period = config.rsi_period
        self.rsi_overbought = config.rsi_overbought
        self.rsi_oversold = config.rsi_oversold

    def prepare_dataframe(self, ohlcv_data: list[list]) -> pd.DataFrame:
        """Converte dados OHLCV brutos em DataFrame com indicadores."""
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Médias Móveis
        if self.ma_type == "ema":
            df["ma_short"] = ta.trend.ema_indicator(df["close"], window=self.short_period)
            df["ma_long"] = ta.trend.ema_indicator(df["close"], window=self.long_period)
        else:
            df["ma_short"] = ta.trend.sma_indicator(df["close"], window=self.short_period)
            df["ma_long"] = ta.trend.sma_indicator(df["close"], window=self.long_period)

        # RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=self.rsi_period)

        # Cruzamentos
        df["ma_cross"] = np.where(df["ma_short"] > df["ma_long"], 1, -1)
        df["ma_cross_shift"] = df["ma_cross"].shift(1)

        df.dropna(inplace=True)
        return df

    def generate_signal(self, df: pd.DataFrame) -> tuple[Signal, dict]:
        """
        Analisa o DataFrame e retorna o sinal atual + razões.
        Usa apenas as duas últimas velas para detectar cruzamento.
        """
        if len(df) < 2:
            return Signal.HOLD, {"motivo": "Dados insuficientes para análise"}

        current = df.iloc[-1]
        previous = df.iloc[-2]

        reasons: dict = {
            "ma_short": round(current["ma_short"], 2),
            "ma_long": round(current["ma_long"], 2),
            "rsi": round(current["rsi"], 2),
            "preco_atual": round(current["close"], 2),
        }

        # Golden Cross: MA curta cruzou acima da MA longa
        golden_cross = (
            previous["ma_short"] <= previous["ma_long"]
            and current["ma_short"] > current["ma_long"]
        )

        # Death Cross: MA curta cruzou abaixo da MA longa
        death_cross = (
            previous["ma_short"] >= previous["ma_long"]
            and current["ma_short"] < current["ma_long"]
        )

        rsi_value = current["rsi"]

        # Sinal de COMPRA
        if golden_cross and rsi_value < self.rsi_oversold:
            reasons["sinal"] = "Golden Cross + RSI em sobrevenda"
            logger.info(
                "Sinal de COMPRA: MA curta (%.2f) cruzou acima da MA longa (%.2f), RSI=%.2f",
                current["ma_short"],
                current["ma_long"],
                rsi_value,
            )
            return Signal.BUY, reasons

        # Compra menos agressiva: golden cross com RSI favorável (abaixo de 50)
        if golden_cross and rsi_value < 50:
            reasons["sinal"] = "Golden Cross + RSI favorável (<50)"
            logger.info(
                "Sinal de COMPRA (moderado): Golden Cross, RSI=%.2f",
                rsi_value,
            )
            return Signal.BUY, reasons

        # Sinal de VENDA
        if death_cross and rsi_value > self.rsi_overbought:
            reasons["sinal"] = "Death Cross + RSI em sobrecompra"
            logger.info(
                "Sinal de VENDA: MA curta (%.2f) cruzou abaixo da MA longa (%.2f), RSI=%.2f",
                current["ma_short"],
                current["ma_long"],
                rsi_value,
            )
            return Signal.SELL, reasons

        # Venda menos agressiva: death cross com RSI desfavorável (acima de 50)
        if death_cross and rsi_value > 50:
            reasons["sinal"] = "Death Cross + RSI desfavorável (>50)"
            logger.info(
                "Sinal de VENDA (moderado): Death Cross, RSI=%.2f",
                rsi_value,
            )
            return Signal.SELL, reasons

        # Sem cruzamento ou sem confirmação de RSI
        reasons["sinal"] = "Sem cruzamento ou sem confirmação RSI"
        return Signal.HOLD, reasons

    def analyze(self, ohlcv_data: list[list]) -> tuple[Signal, dict]:
        """Pipeline completo: prepara dados e gera sinal."""
        df = self.prepare_dataframe(ohlcv_data)
        return self.generate_signal(df)
