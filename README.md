# polyabi
Polymarket crypto trading grid bot PolyAbi

An automated trading bot for Polymarket, utilizing the CLOB API and Data API. It operates on markets defined in pairs.json (up/down) for BTC/USDT pairs, 
in specific time slots, with a delay, stop-loss, and take-profit for the entire pair. Instead of classic hedging, it trades only one side of the market, 
with additional purchases in the grid if the price rises, with the size of each subsequent purchase increasing according to buy_grid_multiplier. 
The bot dynamically reloads pairs.json and filters trade direction based on the trend in the adx.json file.
