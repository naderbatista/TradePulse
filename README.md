# TradePulse - Bot de Trading de Criptomoedas

Bot de trading automatizado para criptomoedas, desenvolvido em Python com foco em **estabilidade**, **gestão de risco** e **lucros consistentes**.

## Funcionalidades

- **Exchanges suportadas**: Binance e Bybit (troca dinâmica)
- **Estratégia**: Cruzamento de Médias Móveis (EMA/SMA) com confirmação RSI
- **Gestão de Risco**: Stop-loss, take-profit, limite diário de perda, controle de posição
- **Paper Trading**: Simulação completa sem dinheiro real
- **Backtesting**: Teste a estratégia em dados históricos
- **Logs estruturados**: Registros em JSON com justificativa de cada decisão
- **CLI completa**: Interface de linha de comando para controle total

## Arquitetura

```
tradepulse/
├── main.py              # Ponto de entrada e CLI
├── config.yaml          # Configuração da estratégia e risco
├── .env.example         # Modelo de variáveis de ambiente
├── requirements.txt     # Dependências Python
├── src/
│   ├── config.py        # Carregamento de configuração
│   ├── logger.py        # Logging estruturado (JSON)
│   ├── exchange.py      # Integração com exchanges (ccxt)
│   ├── strategy.py      # Estratégia de trading (MA + RSI)
│   ├── risk.py          # Gestão de risco
│   ├── trader.py        # Motor de execução de trades
│   ├── paper_trading.py # Simulador de trading
│   └── backtester.py    # Motor de backtesting
└── logs/                # Logs gerados automaticamente
```

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/naderbatista/TradePulse.git
cd TradePulse
```

### 2. Crie o ambiente virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite o arquivo .env com suas chaves de API
```

## Configuração

### Variáveis de Ambiente (`.env`)

| Variável | Descrição |
|---|---|
| `BINANCE_API_KEY` | Chave de API da Binance |
| `BINANCE_SECRET_KEY` | Chave secreta da Binance |
| `BYBIT_API_KEY` | Chave de API da Bybit |
| `BYBIT_SECRET_KEY` | Chave secreta da Bybit |
| `TRADING_MODE` | `paper` (simulação) ou `live` (real) |

### Parâmetros da Estratégia (`config.yaml`)

O arquivo `config.yaml` contém todos os parâmetros configuráveis:

- **Estratégia**: Períodos das médias móveis, tipo (EMA/SMA), RSI
- **Risco**: Stop-loss, take-profit, risco por operação, limite diário
- **Execução**: Tipo de ordem, intervalo entre verificações
- **Backtest**: Saldo inicial, taxa de comissão

## Uso

### Paper Trading (padrão seguro)

```bash
python main.py
```

### Escolher exchange

```bash
python main.py --exchange bybit
python main.py --exchange binance
```

### Trading ao vivo (requer chaves API)

```bash
python main.py --mode live
```

### Backtest

```bash
python main.py --backtest
python main.py --backtest --symbol ETH/USDT --timeframe 4h
```

### Todos os parâmetros

```bash
python main.py --help
```

| Parâmetro | Curto | Descrição |
|---|---|---|
| `--exchange` | `-e` | Exchange: `binance` ou `bybit` |
| `--mode` | `-m` | Modo: `paper` ou `live` |
| `--symbol` | `-s` | Par de negociação (ex: `BTC/USDT`) |
| `--timeframe` | `-t` | Timeframe (ex: `1h`, `15m`, `4h`) |
| `--backtest` | `-b` | Executa backtest |
| `--config` | `-c` | Arquivo de configuração customizado |

## Estratégia: Cruzamento de Médias Móveis + RSI

### Sinal de COMPRA
- Média móvel curta (9) cruza **acima** da média longa (21) → *Golden Cross*
- RSI confirma: valor abaixo de 50 (favorável) ou abaixo de 30 (forte)

### Sinal de VENDA
- Média móvel curta (9) cruza **abaixo** da média longa (21) → *Death Cross*
- RSI confirma: valor acima de 50 (desfavorável) ou acima de 70 (forte)

### Gestão de Risco
- **Risco por operação**: Máximo de 2% do saldo
- **Stop-loss**: 2% abaixo do preço de entrada
- **Take-profit**: 4% acima do preço de entrada
- **Limite diário**: Para de operar após perda de 50 USDT no dia
- **Prevenção de duplicatas**: Apenas uma posição aberta por vez

## Segurança

- Chaves de API **nunca** são hardcoded no código
- Todas as credenciais ficam no arquivo `.env` (não commitado)
- Tratamento de erros em todas as chamadas de API
- Lógica de retry com backoff em falhas de conexão
- Rate limiting habilitado via ccxt

## Stack Tecnológica

- **Python 3.11+**
- **ccxt** - Biblioteca unificada de exchanges
- **pandas** - Manipulação de dados
- **numpy** - Cálculos numéricos
- **ta** - Indicadores de análise técnica
- **asyncio** - Operações assíncronas
- **python-dotenv** - Variáveis de ambiente
- **PyYAML** - Configuração em YAML

## Aviso Legal

Este bot é fornecido apenas para fins educacionais e de pesquisa. Trading de criptomoedas envolve risco significativo de perda financeira. Use por sua conta e risco. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software.

## Licença

MIT License - Veja o arquivo [LICENSE](LICENSE) para detalhes.
