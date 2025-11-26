# polyabi

🧠 Polymarket Single-Side Grid Bot

An automated trading bot for Polymarket CLOB, built around a single-side momentum grid strategy executed within predefined time slots.

🔍 How It Works

Each market has multiple outcome tokens (Up/Down).
For every trading slot, the bot:
Waits a configurable delay period (e.g., 10 minutes) to let early volatility settle.
Selects only one trading side (Up or Down) based on: minimum price increase threshold,
trend filters from adx.json, selecting the strongest (highest-priced) valid leg.
Executes the first buy only if price > baseline + step.
Every minute, performs grid buys only if price continues rising, using a multiplier to increase position size.
Continuously monitors:
Stop-loss per token, Take-profit on the entire pair, removal of inactive tokens.

🧩 Configuration

The bot is controlled via two files:
config.json — global bot parameters (stoploss, grid settings, delays, buy amounts, etc.)
pairs.json — list of markets with individual trading time slots.

⭐ Why This Strategy?

Avoids the typical loss from dual-side hedging (where one side always bleeds).
Concentrates capital on the side showing real momentum.
Expands position only when price confirms strength (grid with trend).
