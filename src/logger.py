"""
TradePulse - Módulo de Logging Estruturado
Logs em JSON com rotação de arquivos
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formata logs como objetos JSON estruturados."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        # Campos extras adicionados pelo desenvolvedor
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(
    name: str = "tradepulse",
    level: str = "INFO",
    log_dir: str = "logs",
) -> logging.Logger:
    """Configura e retorna o logger principal."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Evita handlers duplicados
    if logger.handlers:
        return logger

    # Handler para o console (texto legível)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Handler para arquivo (JSON)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_name = log_path / f"tradepulse_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(file_name, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


def log_decision(
    logger: logging.Logger,
    action: str,
    reason: str,
    data: dict | None = None,
) -> None:
    """Registra uma decisão de trading com contexto estruturado."""
    msg = f"[DECISÃO] {action}: {reason}"
    extra_data = data or {}
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        msg,
        (),
        None,
    )
    record.extra_data = extra_data  # type: ignore[attr-defined]
    logger.handle(record)
