"""
TradePulse - Módulo de Configuração
Carrega variáveis de ambiente e parâmetros do config.yaml
"""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Carrega .env do diretório raiz do projeto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carrega configuração do arquivo YAML."""
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    """Configuração centralizada do bot."""

    def __init__(self, yaml_path: str | Path | None = None):
        self._raw = load_yaml_config(yaml_path)

        # Exchange
        self.exchange: str = self._raw.get("exchange", "binance")
        self.symbol: str = self._raw.get("symbol", "BTC/USDT")
        self.timeframe: str = self._raw.get("timeframe", "1h")
        self.candle_limit: int = self._raw.get("candle_limit", 200)

        # Estratégia
        strat = self._raw.get("strategy", {})
        self.short_ma_period: int = strat.get("short_ma_period", 9)
        self.long_ma_period: int = strat.get("long_ma_period", 21)
        self.ma_type: str = strat.get("ma_type", "ema")
        self.rsi_period: int = strat.get("rsi_period", 14)
        self.rsi_overbought: float = strat.get("rsi_overbought", 70)
        self.rsi_oversold: float = strat.get("rsi_oversold", 30)

        # Risco
        risk = self._raw.get("risk", {})
        self.max_risk_per_trade: float = risk.get("max_risk_per_trade", 0.02)
        self.stop_loss_pct: float = risk.get("stop_loss_pct", 0.02)
        self.take_profit_pct: float = risk.get("take_profit_pct", 0.04)
        self.daily_loss_limit: float = risk.get("daily_loss_limit", 50.0)
        self.max_trades_per_day: int = risk.get("max_trades_per_day", 10)

        # Execução
        exe = self._raw.get("execution", {})
        self.order_type: str = exe.get("order_type", "market")
        self.check_interval: int = exe.get("check_interval", 60)

        # Backtest
        bt = self._raw.get("backtest", {})
        self.bt_initial_balance: float = bt.get("initial_balance", 10000.0)
        self.bt_commission: float = bt.get("commission", 0.001)

        # Logging
        log = self._raw.get("logging", {})
        self.log_level: str = log.get("level", "INFO")
        self.log_dir: str = log.get("log_dir", "logs")

        # Modo de operação (vem do .env)
        self.trading_mode: str = os.getenv("TRADING_MODE", "paper").lower()

        # Chaves de API (vem do .env)
        self.binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
        self.binance_secret: str = os.getenv("BINANCE_SECRET_KEY", "")
        self.bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
        self.bybit_secret: str = os.getenv("BYBIT_SECRET_KEY", "")

    def get_api_keys(self, exchange: str | None = None) -> dict[str, str]:
        """Retorna as chaves de API para a exchange selecionada."""
        ex = (exchange or self.exchange).lower()
        if ex == "binance":
            return {"apiKey": self.binance_api_key, "secret": self.binance_secret}
        elif ex == "bybit":
            return {"apiKey": self.bybit_api_key, "secret": self.bybit_secret}
        else:
            raise ValueError(f"Exchange não suportada: {ex}")

    def __repr__(self) -> str:
        return (
            f"Config(exchange={self.exchange}, symbol={self.symbol}, "
            f"mode={self.trading_mode}, timeframe={self.timeframe})"
        )
