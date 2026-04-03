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
        # Forte: golden cross + RSI em sobrevenda
        if golden_cross and rsi_value < self.rsi_oversold:
            reasons["sinal"] = "Golden Cross + RSI em sobrevenda"
            logger.info(
                "Sinal de COMPRA (forte): MA curta (%.2f) cruzou acima da MA longa (%.2f), RSI=%.2f",
                current["ma_short"],
                current["ma_long"],
                rsi_value,
            )
            return Signal.BUY, reasons

        # Moderado: golden cross com RSI não sobrecomprado (abaixo de 65)
        if golden_cross and rsi_value < 65:
            reasons["sinal"] = "Golden Cross + RSI favorável (<65)"
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

        # Venda moderada: death cross com RSI abaixo de 55 (confirmação de fraqueza)
        if death_cross and rsi_value < 55:
            reasons["sinal"] = "Death Cross + RSI fraco (<55)"
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


class ScalpingMomentum:
    """
    Estratégia de Scalping: Momentum com Filtros de Qualidade

    Detecta micro-tendências, mas filtra ruído de mercados laterais.

    Filtros aplicados ANTES de gerar sinal:
      1. Tendência macro: preço deve estar acima da EMA 21 (só compra a favor da tendência)
      2. Volatilidade: ATR% deve estar acima do mínimo (evita mercado lateral/choppy)
      3. RSI em zona saudável: entre min_rsi e max_rsi (evita zonas fracas e sobrecompra)

    Regras de COMPRA (após filtros):
      - EMA 3 > EMA 8 (micro-tendência de alta)
      - RSI com momentum positivo forte (aceleração, não apenas >0)
      - Preço acima da EMA curta
      - Volume acima da média
      - OU: Golden Cross rápido com confirmação de tendência macro

    Regras de VENDA:
      - EMA 3 < EMA 8 + RSI caindo + preço abaixo da EMA
    """

    def __init__(self, config):
        self.short_period = config.scalping_short_ma
        self.long_period = config.scalping_long_ma
        self.rsi_period = config.scalping_rsi_period
        self.trend_period = config.scalping_trend_ma
        self.min_rsi = config.scalping_min_rsi
        self.max_rsi = config.scalping_max_rsi
        self.atr_period = config.scalping_atr_period
        self.min_atr_pct = config.scalping_min_atr_pct

    def prepare_dataframe(self, ohlcv_data: list[list]) -> pd.DataFrame:
        """Converte dados OHLCV brutos em DataFrame com indicadores de scalping."""
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # EMAs rápidas
        df["ma_short"] = ta.trend.ema_indicator(df["close"], window=self.short_period)
        df["ma_long"] = ta.trend.ema_indicator(df["close"], window=self.long_period)

        # EMA de tendência macro (filtro direcional)
        df["ma_trend"] = ta.trend.ema_indicator(df["close"], window=self.trend_period)

        # RSI rápido
        df["rsi"] = ta.momentum.rsi(df["close"], window=self.rsi_period)

        # Momentum: variação do RSI (RSI atual - RSI anterior)
        df["rsi_momentum"] = df["rsi"].diff()

        # Momentum de preço: variação percentual das últimas 3 velas
        df["price_momentum"] = df["close"].pct_change(periods=3) * 100

        # ATR para filtro de volatilidade
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=self.atr_period)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # Volume relativo (volume atual / média de 10 períodos)
        df["vol_avg"] = df["volume"].rolling(window=10).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"]

        # Cruzamentos
        df["ma_cross"] = np.where(df["ma_short"] > df["ma_long"], 1, -1)

        df.dropna(inplace=True)
        return df

    def generate_signal(self, df: pd.DataFrame) -> tuple[Signal, dict]:
        """
        Analisa o DataFrame e retorna sinal de scalping com filtros de qualidade.
        """
        if len(df) < 3:
            return Signal.HOLD, {"motivo": "Dados insuficientes para scalping"}

        current = df.iloc[-1]
        prev = df.iloc[-2]

        reasons: dict = {
            "ma_short": round(current["ma_short"], 2),
            "ma_long": round(current["ma_long"], 2),
            "ma_trend": round(current["ma_trend"], 2),
            "rsi": round(current["rsi"], 2),
            "rsi_momentum": round(current["rsi_momentum"], 2),
            "price_momentum": round(current["price_momentum"], 4),
            "vol_ratio": round(current["vol_ratio"], 2),
            "atr_pct": round(current["atr_pct"], 4),
            "preco_atual": round(current["close"], 2),
        }

        # ========== FILTROS DE QUALIDADE ==========

        # Filtro 1: Tendência macro - preço deve estar acima da EMA de tendência
        price_above_trend = current["close"] > current["ma_trend"]
        if not price_above_trend:
            reasons["sinal"] = f"Bloqueado: preço ({current['close']:.2f}) abaixo da EMA{self.trend_period} ({current['ma_trend']:.2f})"
            return Signal.HOLD, reasons

        # Filtro 2: Volatilidade mínima - ATR% acima do limiar
        if current["atr_pct"] < self.min_atr_pct:
            reasons["sinal"] = f"Bloqueado: mercado lateral (ATR={current['atr_pct']:.3f}% < {self.min_atr_pct}%)"
            return Signal.HOLD, reasons

        # Filtro 3: RSI em zona operável
        rsi_value = current["rsi"]
        if rsi_value < self.min_rsi:
            reasons["sinal"] = f"Bloqueado: RSI fraco ({rsi_value:.1f} < {self.min_rsi})"
            return Signal.HOLD, reasons
        if rsi_value > self.max_rsi:
            reasons["sinal"] = f"Bloqueado: RSI sobrecompra ({rsi_value:.1f} > {self.max_rsi})"
            return Signal.HOLD, reasons

        # ========== SINAIS (após filtros) ==========

        ema_bullish = current["ma_short"] > current["ma_long"]
        ema_bearish = current["ma_short"] < current["ma_long"]
        rsi_rising = current["rsi_momentum"] > 1.5  # Exige momentum real, não ruído
        rsi_falling = current["rsi_momentum"] < -1.5
        price_above_ema = current["close"] > current["ma_short"]
        price_below_ema = current["close"] < current["ma_short"]
        volume_ok = current["vol_ratio"] > 1.0  # Volume acima da média (era 0.8)
        price_momentum_up = current["price_momentum"] > 0.05  # Preço subindo de verdade

        # COMPRA: tendência de alta + momentum positivo + volume
        if ema_bullish and rsi_rising and price_above_ema and volume_ok and price_momentum_up:
            strength = "forte" if current["rsi_momentum"] > 3 and current["vol_ratio"] > 1.5 else "moderado"
            reasons["sinal"] = f"Scalp BUY ({strength}): EMA↑ + RSI↑ + Vol↑ + Trend OK"
            logger.info(
                "SCALP BUY (%s): EMA%d=%.2f > EMA%d=%.2f, RSI=%.2f (mom=%.2f), "
                "Vol=%.2fx, ATR=%.3f%%, Trend EMA%d=%.2f",
                strength, self.short_period, current["ma_short"],
                self.long_period, current["ma_long"],
                current["rsi"], current["rsi_momentum"],
                current["vol_ratio"], current["atr_pct"],
                self.trend_period, current["ma_trend"],
            )
            return Signal.BUY, reasons

        # COMPRA: golden cross rápido com confirmação de tendência
        golden_cross = (prev["ma_short"] <= prev["ma_long"] and current["ma_short"] > current["ma_long"])
        if golden_cross and rsi_value >= self.min_rsi and rsi_value <= self.max_rsi and price_momentum_up:
            reasons["sinal"] = "Scalp BUY: Golden Cross + Tendência macro OK"
            logger.info(
                "SCALP BUY (cross): EMA%d cruzou EMA%d, RSI=%.2f, ATR=%.3f%%, Trend OK",
                self.short_period, self.long_period, current["rsi"], current["atr_pct"],
            )
            return Signal.BUY, reasons

        # VENDA: tendência de baixa + momentum negativo
        if ema_bearish and rsi_falling and price_below_ema:
            reasons["sinal"] = "Scalp SELL: EMA↓ + RSI↓ + Preço < EMA"
            logger.info(
                "SCALP SELL: EMA%d=%.2f < EMA%d=%.2f, RSI=%.2f (mom=%.2f)",
                self.short_period, current["ma_short"],
                self.long_period, current["ma_long"],
                current["rsi"], current["rsi_momentum"],
            )
            return Signal.SELL, reasons

        reasons["sinal"] = "Scalp: Aguardando sinal de qualidade"
        return Signal.HOLD, reasons

    def analyze(self, ohlcv_data: list[list]) -> tuple[Signal, dict]:
        """Pipeline completo: prepara dados e gera sinal de scalping."""
        df = self.prepare_dataframe(ohlcv_data)
        return self.generate_signal(df)
