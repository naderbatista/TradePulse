"""
TradePulse - Módulo de Predição ML (XGBoost)
Prediz a probabilidade de alta/queda nas próximas velas usando
features derivados dos indicadores técnicos existentes.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("tradepulse")


class MLPredictor:
    """
    Preditor de direção usando XGBoost.

    Features (23 indicadores normalizados):
    - EMA ratios (3/8, 8/21, close/50)
    - RSI + momentum + aceleração
    - MACD histograma normalizado + variação
    - Bollinger Bands %B + width
    - Volume ratio + tendência
    - ATR %
    - Price momentum multi-horizon (3, 5, 10 velas)
    - Candlestick structure (body, wicks)
    - Close changes recentes

    Target: preço subiu nas próximas N velas (classificação binária)

    Pipeline:
      1. train(df) → treina modelo com dados históricos
      2. predict(df) → retorna probabilidade de alta
      3. should_retrain() → retorna True a cada N ciclos
    """

    def __init__(self, config):
        self.enabled = getattr(config, "ml_enabled", True)
        self.min_probability = getattr(config, "ml_min_probability", 0.55)
        self.retrain_interval = getattr(config, "ml_retrain_interval", 50)
        self.lookback = getattr(config, "ml_lookback_candles", 500)
        self.prediction_horizon = getattr(config, "ml_prediction_horizon", 3)

        self.model = None
        self.scaler = None
        self._cycles_since_train = 0
        self._is_trained = False
        self._last_accuracy = 0.0
        self._train_samples = 0
        self._feature_importances: dict[str, float] = {}

        self._feature_names = [
            "ema_3_8_ratio",
            "ema_8_21_ratio",
            "close_ema50_ratio",
            "rsi",
            "rsi_momentum",
            "rsi_acceleration",
            "macd_hist_norm",
            "macd_hist_change",
            "bb_percent_b",
            "bb_width",
            "vol_ratio",
            "vol_trend",
            "atr_pct",
            "price_mom_3",
            "price_mom_5",
            "price_mom_10",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "close_change_1",
            "close_change_2",
            "high_low_range",
            "hour_sin",
        ]

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features ML do DataFrame com indicadores técnicos."""
        features = pd.DataFrame(index=df.index)

        # === EMA ratios (normalizados, independentes de preço) ===
        features["ema_3_8_ratio"] = (df["ma_short"] / df["ma_long"] - 1) * 100
        features["ema_8_21_ratio"] = (df["ma_long"] / df["ema_21"] - 1) * 100
        features["close_ema50_ratio"] = (df["close"] / df["ema_trend"] - 1) * 100

        # === RSI features ===
        features["rsi"] = df["rsi"]
        features["rsi_momentum"] = df["rsi_momentum"]
        features["rsi_acceleration"] = df["rsi_momentum"].diff()

        # === MACD (normalizado pelo preço) ===
        features["macd_hist_norm"] = df["macd_hist"] / df["close"] * 10000
        features["macd_hist_change"] = (
            (df["macd_hist"] - df["macd_hist_prev"]) / df["close"] * 10000
        )

        # === Bollinger Bands ===
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        features["bb_percent_b"] = (df["close"] - df["bb_lower"]) / bb_range
        features["bb_width"] = df["bb_width"]

        # === Volume ===
        features["vol_ratio"] = df["vol_ratio"]
        features["vol_trend"] = df["vol_ratio"].rolling(3).mean()

        # === ATR ===
        features["atr_pct"] = df["atr_pct"]

        # === Price momentum multi-horizon ===
        features["price_mom_3"] = df["close"].pct_change(3) * 100
        features["price_mom_5"] = df["close"].pct_change(5) * 100
        features["price_mom_10"] = df["close"].pct_change(10) * 100

        # === Candlestick structure ===
        body = abs(df["close"] - df["open"])
        total_range = (df["high"] - df["low"]).replace(0, np.nan)
        features["body_pct"] = body / total_range
        features["upper_wick_pct"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        ) / total_range
        features["lower_wick_pct"] = (
            df[["open", "close"]].min(axis=1) - df["low"]
        ) / total_range

        # === Recent changes ===
        features["close_change_1"] = df["close"].pct_change(1) * 100
        features["close_change_2"] = df["close"].pct_change(2) * 100

        # === Range ===
        features["high_low_range"] = total_range / df["close"] * 100

        # === Temporal (ciclo de hora como seno para capturar padrões intraday) ===
        if hasattr(df.index, "hour"):
            features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        else:
            features["hour_sin"] = 0.0

        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)
        return features

    def _create_labels(self, df: pd.DataFrame, horizon: int = 3) -> pd.Series:
        """Cria labels: 1 se preço subiu nas próximas N velas, 0 se caiu."""
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        return (future_return > 0).astype(int)

    def train(self, df: pd.DataFrame) -> bool:
        """
        Treina o modelo XGBoost com dados históricos.

        Usa TimeSeriesSplit para validação temporal (sem data leakage).
        Parâmetros conservadores para evitar overfitting em dados ruidosos.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import StandardScaler

            features = self._extract_features(df)
            labels = self._create_labels(df, self.prediction_horizon)

            # Alinhar features e labels (remover linhas sem label futuro)
            valid_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[valid_idx]
            y = labels.loc[valid_idx]

            if len(X) < 100:
                logger.warning(
                    "ML: Dados insuficientes para treinar (%d amostras, mín 100)",
                    len(X),
                )
                return False

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train/test split temporal (sem shuffle para respeitar a ordem do tempo)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # XGBoost conservador
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )

            self.model.fit(X_train, y_train)

            # Avaliar no test set
            accuracy = self.model.score(X_test, y_test)
            self._last_accuracy = round(accuracy * 100, 1)
            self._train_samples = len(X_train)
            self._is_trained = True
            self._cycles_since_train = 0

            # Feature importances (top 5)
            importances = self.model.feature_importances_
            feat_names = features.columns.tolist()
            imp_dict = dict(zip(feat_names, importances))
            self._feature_importances = dict(
                sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            )

            logger.info(
                "ML treinado: accuracy=%.1f%%, amostras=%d (train=%d, test=%d), "
                "top features: %s",
                self._last_accuracy,
                len(X),
                len(X_train),
                len(X_test),
                ", ".join(f"{k}={v:.3f}" for k, v in self._feature_importances.items()),
            )
            return True

        except ImportError:
            logger.error(
                "ML: xgboost ou sklearn não instalados. "
                "Instale: pip install xgboost scikit-learn"
            )
            self.enabled = False
            return False
        except Exception as e:
            logger.error("ML: Erro no treinamento: %s", e)
            return False

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Prediz probabilidade de alta para o estado atual do mercado.

        Retorna:
            {
                "probability": float (0.0 - 1.0),
                "direction": "up" | "down" | "neutral",
                "confidence": "alta" | "moderada" | "baixa" | "contra",
                "accuracy": float,
                "trained": bool,
            }
        """
        if not self.enabled or not self._is_trained or self.model is None:
            return {
                "probability": 0.5,
                "direction": "neutral",
                "confidence": "sem_modelo",
                "accuracy": 0.0,
                "train_samples": 0,
                "trained": False,
            }

        try:
            features = self._extract_features(df)
            if len(features) == 0:
                return {
                    "probability": 0.5,
                    "direction": "neutral",
                    "confidence": "sem_dados",
                    "accuracy": self._last_accuracy,
                    "train_samples": self._train_samples,
                    "trained": True,
                }

            # Última linha = estado atual do mercado
            X_current = features.iloc[[-1]]
            X_scaled = self.scaler.transform(X_current)

            proba = self.model.predict_proba(X_scaled)[0]
            prob_up = float(proba[1]) if len(proba) > 1 else 0.5

            direction = "up" if prob_up > 0.5 else "down"
            if prob_up >= 0.65:
                confidence = "alta"
            elif prob_up >= 0.55:
                confidence = "moderada"
            elif prob_up >= 0.45:
                confidence = "baixa"
            else:
                confidence = "contra"

            return {
                "probability": round(prob_up, 3),
                "direction": direction,
                "confidence": confidence,
                "accuracy": self._last_accuracy,
                "train_samples": self._train_samples,
                "trained": True,
            }

        except Exception as e:
            logger.error("ML: Erro na predição: %s", e)
            return {
                "probability": 0.5,
                "direction": "neutral",
                "confidence": "erro",
                "accuracy": self._last_accuracy,
                "train_samples": self._train_samples,
                "trained": True,
            }

    def should_retrain(self) -> bool:
        """Verifica se é hora de re-treinar o modelo."""
        self._cycles_since_train += 1
        return self._cycles_since_train >= self.retrain_interval

    def get_status(self) -> dict:
        """Retorna status completo do modelo ML para o dashboard."""
        return {
            "enabled": self.enabled,
            "trained": self._is_trained,
            "accuracy": self._last_accuracy,
            "train_samples": self._train_samples,
            "cycles_since_train": self._cycles_since_train,
            "retrain_interval": self.retrain_interval,
            "min_probability": self.min_probability,
            "prediction_horizon": self.prediction_horizon,
            "top_features": {
                k: round(v, 3) for k, v in self._feature_importances.items()
            },
        }
