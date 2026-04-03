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
    Estratégia de Scalping: Sistema de Pontuação Multi-Indicador

    Inspirada em sistemas quantitativos profissionais:
    - Score-based: cada indicador contribui pontos, opera quando score >= mínimo
    - Multi-indicador: EMA stack, RSI, MACD, Bollinger Bands, VWAP, Volume
    - Filtro macro: só opera a favor da tendência (EMA 50)
    - Adaptativo: funciona em qualquer timeframe (1m, 5m, 15m)

    Pontuação (precisa ≥ min_score para operar):
      +1  EMA 3 > EMA 8 (micro-tendência)
      +1  EMA 8 > EMA 21 (tendência média alinhada)
      +1  RSI entre 40-65 (zona saudável, não sobrecomprado)
      +1  RSI momentum > 0 (acelerando)
      +1  MACD histograma positivo e crescente
      +1  Preço acima da banda média de Bollinger
      +1  Volume acima da média
      +1  Preço em momentum positivo (3 velas)
      +2  Golden Cross (EMA curta cruzou EMA longa nesta vela) — bônus
    """

    def __init__(self, config):
        self.short_period = config.scalping_short_ma       # EMA 3
        self.long_period = config.scalping_long_ma         # EMA 8
        self.rsi_period = config.scalping_rsi_period       # RSI 7
        self.trend_period = config.scalping_trend_ma       # EMA 50 (macro)
        self.min_score = getattr(config, 'scalping_min_score', 5)

    def prepare_dataframe(self, ohlcv_data: list[list]) -> pd.DataFrame:
        """Converte dados OHLCV em DataFrame com indicadores profissionais."""
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # === EMAs: micro, médio e macro ===
        df["ma_short"] = ta.trend.ema_indicator(df["close"], window=self.short_period)   # 3
        df["ma_long"] = ta.trend.ema_indicator(df["close"], window=self.long_period)     # 8
        df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
        df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=self.trend_period)  # 50

        # === RSI ===
        df["rsi"] = ta.momentum.rsi(df["close"], window=self.rsi_period)
        df["rsi_momentum"] = df["rsi"].diff()

        # === MACD (12, 26, 9) ===
        macd_ind = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd_hist"] = macd_ind.macd_diff()
        df["macd_hist_prev"] = df["macd_hist"].shift(1)

        # === Bollinger Bands (20, 2) ===
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100

        # === VWAP (calculado como média ponderada cumulativa do dia/sessão) ===
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

        # === Volume relativo ===
        df["vol_avg"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"]

        # === ATR ===
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # === Momentum de preço ===
        df["price_momentum"] = df["close"].pct_change(periods=3) * 100

        df.dropna(inplace=True)
        return df

    def generate_signal(self, df: pd.DataFrame) -> tuple[Signal, dict]:
        """
        Sistema de pontuação: calcula score com múltiplos indicadores.
        Só compra se score >= min_score E preço acima da EMA macro.
        """
        if len(df) < 3:
            return Signal.HOLD, {"motivo": "Dados insuficientes para scalping"}

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # === FILTRO MACRO OBRIGATÓRIO ===
        price_above_trend = current["close"] > current["ema_trend"]

        reasons: dict = {
            "ma_short": round(current["ma_short"], 2),
            "ma_long": round(current["ma_long"], 2),
            "ema_trend": round(current["ema_trend"], 2),
            "rsi": round(current["rsi"], 2),
            "rsi_momentum": round(current["rsi_momentum"], 2),
            "macd_hist": round(current["macd_hist"], 4),
            "bb_pos": "acima" if current["close"] > current["bb_mid"] else "abaixo",
            "vwap": round(current["vwap"], 2),
            "vol_ratio": round(current["vol_ratio"], 2),
            "atr_pct": round(current["atr_pct"], 4),
            "preco_atual": round(current["close"], 2),
        }

        if not price_above_trend:
            reasons["sinal"] = f"Bloqueado: preço abaixo da EMA{self.trend_period} ({current['ema_trend']:.2f})"
            reasons["score"] = 0
            return Signal.HOLD, reasons

        # === CÁLCULO DO SCORE ===
        score = 0
        details = []

        # 1. EMA micro bullish (EMA 3 > EMA 8)
        if current["ma_short"] > current["ma_long"]:
            score += 1
            details.append("EMA3>8 ✓")

        # 2. EMAs médias alinhadas (EMA 8 > EMA 21)
        if current["ma_long"] > current["ema_21"]:
            score += 1
            details.append("EMA8>21 ✓")

        # 3. RSI em zona saudável (40-65 = bom momentum sem sobrecompra)
        rsi = current["rsi"]
        if 40 <= rsi <= 65:
            score += 1
            details.append(f"RSI={rsi:.0f} ✓")

        # 4. RSI com momentum positivo (acelerando)
        if current["rsi_momentum"] > 0:
            score += 1
            details.append(f"RSImom={current['rsi_momentum']:.1f} ✓")

        # 5. MACD histograma positivo e crescente
        if current["macd_hist"] > 0 and current["macd_hist"] > current["macd_hist_prev"]:
            score += 1
            details.append("MACD↑ ✓")

        # 6. Preço acima da banda média de Bollinger
        if current["close"] > current["bb_mid"]:
            score += 1
            details.append("BB>mid ✓")

        # 7. Volume acima da média
        if current["vol_ratio"] > 1.0:
            score += 1
            details.append(f"Vol={current['vol_ratio']:.1f}x ✓")

        # 8. Price momentum positivo (últimas 3 velas subindo)
        if current["price_momentum"] > 0:
            score += 1
            details.append(f"Mom={current['price_momentum']:.2f}% ✓")

        # 9. BÔNUS: Golden Cross (EMA curta CRUZOU longa nesta vela) +2
        golden_cross = (prev["ma_short"] <= prev["ma_long"] and current["ma_short"] > current["ma_long"])
        if golden_cross:
            score += 2
            details.append("GoldenX +2 ✓")

        reasons["score"] = score
        reasons["min_score"] = self.min_score
        reasons["detalhes"] = " | ".join(details)

        # === DECISÃO DE COMPRA ===
        if score >= self.min_score:
            strength = "forte" if score >= 7 else "moderado"
            reasons["sinal"] = f"Scalp BUY ({strength}): Score {score}/{self.min_score} — {' | '.join(details)}"
            logger.info(
                "SCALP BUY (%s): Score=%d/%d | %s | Preço=%.2f | EMA_trend=%.2f",
                strength, score, self.min_score,
                " | ".join(details),
                current["close"], current["ema_trend"],
            )
            return Signal.BUY, reasons

        # === DECISÃO DE VENDA (fechar posição) ===
        sell_score = 0
        if current["ma_short"] < current["ma_long"]:
            sell_score += 1
        if current["rsi_momentum"] < -1:
            sell_score += 1
        if current["macd_hist"] < 0 and current["macd_hist"] < current["macd_hist_prev"]:
            sell_score += 1
        if current["close"] < current["bb_mid"]:
            sell_score += 1

        if sell_score >= 3:
            reasons["sinal"] = f"Scalp SELL: Score venda={sell_score}/4"
            logger.info(
                "SCALP SELL: sell_score=%d | RSI=%.2f | MACD=%.4f",
                sell_score, current["rsi"], current["macd_hist"],
            )
            return Signal.SELL, reasons

        reasons["sinal"] = f"Scalp HOLD: Score {score}/{self.min_score} — aguardando"
        return Signal.HOLD, reasons

    def analyze(self, ohlcv_data: list[list]) -> tuple[Signal, dict]:
        """Pipeline completo: prepara dados e gera sinal de scalping."""
        df = self.prepare_dataframe(ohlcv_data)
        return self.generate_signal(df)
