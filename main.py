"""
TradePulse - Bot de Trading de Criptomoedas
Interface de linha de comando (CLI)

Uso:
    python main.py                          # Inicia com configuração padrão
    python main.py --exchange bybit         # Usa Bybit
    python main.py --mode paper             # Paper trading
    python main.py --mode live              # Trading real (cuidado!)
    python main.py --backtest               # Executa backtest
    python main.py --dashboard              # Abre dashboard web
    python main.py --symbol ETH/USDT        # Par específico
"""
import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.logger import setup_logger
from src.trader import Trader
from src.backtester import Backtester
from src.exchange import ExchangeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="TradePulse",
        description="Bot de trading de criptomoedas com gestão de risco",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                           Inicia em paper trading (padrão)
  python main.py --exchange bybit          Usa a exchange Bybit
  python main.py --mode live               Trading real (requer chaves API)
  python main.py --backtest                Executa backtest na estratégia
  python main.py --dashboard               Abre dashboard web em localhost:8080
  python main.py --symbol ETH/USDT -t 15m  ETH em velas de 15 minutos
        """,
    )
    parser.add_argument(
        "--exchange", "-e",
        choices=["binance", "bybit"],
        default=None,
        help="Exchange para operar (padrão: config.yaml)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["paper", "live"],
        default=None,
        help="Modo de operação (padrão: variável TRADING_MODE do .env)",
    )
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        help="Par de negociação (ex: BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        default=None,
        help="Timeframe das velas (ex: 1h, 15m, 4h)",
    )
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Executa backtest ao invés de trading ao vivo",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Inicia o dashboard web (http://localhost:8080)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Porta do dashboard web (padrão: 8080)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Caminho para arquivo de configuração YAML",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """Aplica overrides da CLI sobre a configuração carregada."""
    if args.exchange:
        config.exchange = args.exchange
    if args.mode:
        config.trading_mode = args.mode
    if args.symbol:
        config.symbol = args.symbol
    if args.timeframe:
        config.timeframe = args.timeframe


async def run_backtest(config: Config) -> None:
    """Executa backtest: busca dados históricos e simula a estratégia."""
    logger = logging.getLogger("tradepulse")
    logger.info("Iniciando backtest para %s (%s)...", config.symbol, config.timeframe)

    # Backtest usa API pública real (sem sandbox) apenas para dados de mercado
    exchange_client = ExchangeClient(config, sandbox=False)
    await exchange_client.connect()

    try:
        ohlcv = await exchange_client.fetch_ohlcv()
        logger.info("Dados obtidos: %d velas", len(ohlcv))

        backtester = Backtester(config)
        results = backtester.run(ohlcv)

        print("\n" + "=" * 60)
        print("  RESULTADO DO BACKTEST - TradePulse")
        print("=" * 60)
        for key, value in results.items():
            if key == "operacoes":
                print(f"\n  Total de operações detalhadas: {len(value)}")
                for i, trade in enumerate(value, 1):
                    print(f"    #{i}: {trade['tipo']} | Entrada={trade['entrada']:.2f} "
                          f"| Saída={trade['saida']:.2f} | PnL={trade['pnl']:.2f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60 + "\n")

    finally:
        await exchange_client.close()


async def run_trading(config: Config) -> None:
    """Executa o loop de trading principal."""
    logger = logging.getLogger("tradepulse")

    print("\n" + "=" * 60)
    print("  TradePulse - Bot de Trading de Criptomoedas")
    print("=" * 60)
    print(f"  Exchange:   {config.exchange}")
    print(f"  Par:        {config.symbol}")
    print(f"  Timeframe:  {config.timeframe}")
    print(f"  Modo:       {config.trading_mode.upper()}")
    print(f"  Intervalo:  {config.check_interval}s")
    print("=" * 60)

    if config.trading_mode == "live":
        print("\n  ⚠️  ATENÇÃO: Modo LIVE ativado! Operações com dinheiro real!")
        print("  Pressione Ctrl+C para interromper a qualquer momento.\n")

    trader = Trader(config)

    # Captura sinais de interrupção
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Sinal de encerramento recebido...")
        trader.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_handler)
        except NotImplementedError:
            # Windows não suporta add_signal_handler
            pass

    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
        await trader.stop()


def main() -> None:
    args = parse_args()

    # Carrega configuração
    config = Config(yaml_path=args.config)
    apply_cli_overrides(config, args)

    # Configura logger
    logger = setup_logger(
        name="tradepulse",
        level=config.log_level,
        log_dir=config.log_dir,
    )

    logger.info("Configuração carregada: %s", config)

    # Executa
    if args.dashboard:
        run_dashboard(config, args.port)
    elif args.backtest:
        asyncio.run(run_backtest(config))
    else:
        asyncio.run(run_trading(config))


def run_dashboard(config: Config, port: int = 8080) -> None:
    """Inicia o dashboard web com FastAPI + Uvicorn."""
    import uvicorn
    from src.dashboard import app, state

    state.config = config
    state.strategy = __import__('src.strategy', fromlist=['MACrossoverRSI']).MACrossoverRSI(config)
    state.risk_manager = __import__('src.risk', fromlist=['RiskManager']).RiskManager(config=config)
    if config.trading_mode == 'paper':
        state.paper_trader = __import__('src.paper_trading', fromlist=['PaperTrader']).PaperTrader(config)

    print("\n" + "=" * 60)
    print("  TradePulse - Dashboard Web")
    print("=" * 60)
    print(f"  URL:        http://localhost:{port}")
    print(f"  Exchange:   {config.exchange}")
    print(f"  Par:        {config.symbol}")
    print(f"  Modo:       {config.trading_mode.upper()}")
    print("=" * 60)
    print("  Abra o navegador e clique em 'Iniciar' para começar.\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
