import json
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, time as dtime, timedelta

import requests

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.exceptions import PolyApiException


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("abi-bot")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("abi.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================
# EXCEPTION FOR INACTIVE TOKENS
# ============================================================

class InactiveTokenError(Exception):
    """Token is inactive – /price endpoint returns 404 or price <= 0."""
    pass


# ============================================================
# MODELS
# ============================================================

@dataclass
class Config:
    private_key: str
    proxy_address: str
    api_key: str
    api_secret: str
    api_passphrase: str

    stoploss_price: float
    add_step_price: float
    buy_amount_usdc: float
    buy_grid_multiplier: float
    pair_take_profit_pct: float

    interval_delay_minutes: int
    poll_interval_sec: float

    gamma_base_url: str
    clob_host: str
    chain_id: int
    signature_type: int
    slippage_percent: float

    pairs_file: str
    adx_file: str
    pairs_reload_interval_sec: float   # how often to reload pairs.json in seconds


@dataclass
class TokenLegState:
    token_id: str
    label: str
    market_key: str
    pair_symbol: Optional[str]

    stoploss_price: float
    add_step_price: float
    buy_amount_usdc: float

    total_buys: int = 0
    total_sells: int = 0
    is_closed: bool = False

    # We keep these for potential future use; current buy logic does not depend on them:
    slot_start_price: Optional[float] = None
    anchor_buy_price: Optional[float] = None

    last_buy_price: Optional[float] = None
    last_buy_minute_index: Optional[int] = None  # to enforce "max 1 BUY per minute"


@dataclass
class PairState:
    market_key: str
    pair_symbol: Optional[str]
    trade_start_str: str    # "HH:MM"
    trade_end_str: str      # "HH:MM"
    legs: Dict[str, TokenLegState] = field(default_factory=dict)

    active_token_id: Optional[str] = None      # which leg is currently "directional" in this window
    closed: bool = False                       # whether the pair is fully disabled (TP/expired/inactive)


# ============================================================
# TIME & TREND HELPERS
# ============================================================

def parse_hhmm(hhmm: str) -> dtime:
    """Parse 'HH:MM' string into a datetime.time object."""
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {hhmm}")
    h = int(parts[0])
    m = int(parts[1])
    return dtime(hour=h, minute=m)


def build_today_window(trade_start_str: str, trade_end_str: str) -> Tuple[datetime, datetime]:
    """
    Return (start_dt, end_dt) for today's date,
    given start/end times in 'HH:MM' format.
    """
    today = date.today()
    start_t = parse_hhmm(trade_start_str)
    end_t = parse_hhmm(trade_end_str)
    start_dt = datetime.combine(today, start_t)
    end_dt = datetime.combine(today, end_t)
    return start_dt, end_dt


def load_trend_map(path: str) -> Dict[str, str]:
    """
    Load adx.json file, expected format:
      { "BTC/USDT": "UP", "ETH/USDT": "DOWN" }

    On error returns empty dict (no trend filtering).
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: str(v).upper() for k, v in data.items()}
        return {}
    except FileNotFoundError:
        logger.debug("Missing file %s – trend not used.", path)
    except Exception as e:
        logger.warning("Read error %s: %s", path, e)
    return {}


def label_to_direction(label: str) -> Optional[str]:
    """
    Map a leg label (Up/Down, Yes/No etc.) to a logical direction:
    - 'UP'    => leg earns when the underlying goes up
    - 'DOWN'  => leg earns when the underlying goes down
    Returns 'UP', 'DOWN' or None if we cannot determine.
    """
    lu = label.upper()
    if "UP" in lu or lu == "YES":
        return "UP"
    if "DOWN" in lu or lu == "NO":
        return "DOWN"
    return None


# ============================================================
# CONFIG, PAIRS, CLIENT
# ============================================================

def load_config(path: str = "config.json") -> Config:
    logger.info("Loading config from file: %s", path)
    with open(path, "r") as f:
        data = json.load(f)

    for key in ["private_key", "proxy_address", "api_key", "api_secret", "api_passphrase"]:
        if not data.get(key):
            logger.error("Brak wartości '%s' w %s", key, path)
            raise RuntimeError(f"Missig data '{key}' in {path}")

    interval_delay_minutes = int(data.get("interval_delay_minutes", 0))
    if interval_delay_minutes < 0:
        interval_delay_minutes = 0

    pairs_reload_interval_sec = float(data.get("pairs_reload_interval_sec", 60.0))
    if pairs_reload_interval_sec < 5.0:
        pairs_reload_interval_sec = 5.0  # minimum, to avoid hammering the file every second

    cfg = Config(
        private_key=data["private_key"],
        proxy_address=data["proxy_address"],
        api_key=data["api_key"],
        api_secret=data["api_secret"],
        api_passphrase=data["api_passphrase"],
        stoploss_price=float(data.get("stoploss_price", 0.3)),
        add_step_price=float(data.get("add_step_price", 0.05)),
        buy_amount_usdc=float(data.get("buy_amount_usdc", 1.1)),
        buy_grid_multiplier=float(data.get("buy_grid_multiplier", 1.0)),
        pair_take_profit_pct=float(data.get("pair_take_profit_pct", 0.2)),
        interval_delay_minutes=interval_delay_minutes,
        poll_interval_sec=float(data.get("poll_interval_sec", 5.0)),
        gamma_base_url=data.get("gamma_base_url", "https://data-api.polymarket.com"),
        clob_host=data.get("clob_host", "https://clob.polymarket.com"),
        chain_id=int(data.get("chain_id", 137)),
        signature_type=int(data.get("signature_type", 2)),
        slippage_percent=float(data.get("slippage_percent", 0.03)),
        pairs_file=data.get("pairs_file", "pairs.json"),
        adx_file=data.get("adx_file", "adx.json"),
        pairs_reload_interval_sec=pairs_reload_interval_sec,
    )

    logger.info(
        "Config: stoploss=%.4f, add_step=%.4f, buy_amount=%.2f, grid_mult=%.3f, "
        "pair_TP=%.2f, delay=%d min, poll=%.1fs, pairs_file=%s, adx_file=%s, pairs_reload=%.1fs",
        cfg.stoploss_price,
        cfg.add_step_price,
        cfg.buy_amount_usdc,
        cfg.buy_grid_multiplier,
        cfg.pair_take_profit_pct,
        cfg.interval_delay_minutes,
        cfg.poll_interval_sec,
        cfg.pairs_file,
        cfg.adx_file,
        cfg.pairs_reload_interval_sec,
    )
    return cfg


def init_clob_client(cfg: Config) -> ClobClient:
    logger.info("Inicjalizuję ClobClient...")
    client = ClobClient(
        cfg.clob_host,
        key=cfg.private_key,
        chain_id=cfg.chain_id,
        signature_type=cfg.signature_type,
        funder=cfg.proxy_address,
    )
    api_creds = ApiCreds(
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        api_passphrase=cfg.api_passphrase,
    )
    client.set_api_creds(api_creds)
    logger.info("ClobClient gotowy (host=%s, chain_id=%d).", cfg.clob_host, cfg.chain_id)
    return client


def load_pairs(cfg: Config) -> Dict[str, PairState]:
    """
    Load pairs_file (pairs.json) with format:
    {
      "pairs": [
        {
          "market_key": "...",
          "pair_symbol": "BTC/USDT",
          "trade_start": "11:30",
          "trade_end": "11:45",
          "legs": [
            { "token_id": "...", "label": "Down" },
            { "token_id": "...", "label": "Up" }
          ]
        },
        ...
      ]
    }

    Returns a dict: (market_key|trade_start|trade_end) -> PairState
    """
    with open(cfg.pairs_file, "r") as f:
        data = json.load(f)

    pairs_raw = data.get("pairs", [])
    pairs: Dict[str, PairState] = {}

    for p in pairs_raw:
        market_key = p.get("market_key")
        pair_symbol = p.get("pair_symbol")
        trade_start = p.get("trade_start")
        trade_end = p.get("trade_end")
        legs_raw = p.get("legs", [])

        if not market_key or not trade_start or not trade_end or not legs_raw:
            logger.warning(
                "A pair was omitted due to missing data. (market_key/trade_start/trade_end/legs). Data: %s", p
            )
            continue

        pair_state = PairState(
            market_key=market_key,
            pair_symbol=pair_symbol,
            trade_start_str=trade_start,
            trade_end_str=trade_end,
            legs={},
        )

        for leg in legs_raw:
            token_id = leg.get("token_id")
            label = leg.get("label") or "LEG"
            if not token_id:
                logger.warning(
                    "A leg was omitted in the pair %s (missing token_id): %s",
                    market_key,
                    leg,
                )
                continue

            tls = TokenLegState(
                token_id=token_id,
                label=label,
                market_key=market_key,
                pair_symbol=pair_symbol,
                stoploss_price=cfg.stoploss_price,
                add_step_price=cfg.add_step_price,
                buy_amount_usdc=cfg.buy_amount_usdc,
            )
            pair_state.legs[token_id] = tls
            logger.info(
                "PAIR %s: added leg token_id=%s, label=%s, pair_symbol=%s, slot=[%s-%s]",
                market_key,
                token_id,
                label,
                pair_symbol,
                trade_start,
                trade_end,
            )

        if len(pair_state.legs) == 0:
            logger.warning("Pair %s there are no active legs – I'm leaving that out.", market_key)
            continue

        pair_key = f"{market_key}|{trade_start}|{trade_end}"
        pairs[pair_key] = pair_state

    logger.info("Loades %d pairs from file %s.", len(pairs), cfg.pairs_file)
    return pairs


# ============================================================
# DATA-API – POSITIONS
# ============================================================

def fetch_positions_simple(cfg: Config) -> List[dict]:
    """
    Fetch positions from Data-API:
      GET /positions?user=0x...&limit=1000
    Returns a list of positions.
    """
    url = f"{cfg.gamma_base_url}/positions"
    params = {"user": cfg.proxy_address, "limit": 1000}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("positions", [])


def get_token_size_from_positions(cfg: Config, token_id: str) -> float:
    """
    Use Data-API to compute how many units of a given token we hold (sum size).
    """
    try:
        positions = fetch_positions_simple(cfg)
    except Exception as e:
        logger.error("Error getting position in get_token_size_from_positions: %s", e)
        return 0.0

    balance = 0.0
    for p in positions:
        asset = p.get("asset")
        if asset != token_id:
            continue
        size = float(p.get("size", 0.0) or 0.0)
        balance += size

    logger.info(
        "Balance (size) from Data-API for token_id=%s totals %.8f",
        token_id,
        balance,
    )
    return balance


# ============================================================
# CLOB – PRICE & ORDERS
# ============================================================

def get_best_price(clob_host: str, token_id: str, side_for_price: str) -> float:
    """
    GET /price?token_id=...&side=BUY/SELL
    If 404 or price <= 0 -> InactiveTokenError.
    """
    url = f"{clob_host}/price"
    params = {"token_id": token_id, "side": side_for_price}
    r = requests.get(url, params=params)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 404:
            logger.warning(
                "Price endpoint returned 404 for token_id=%s, side=%s. I consider the token inactive.",
                token_id,
                side_for_price,
            )
            raise InactiveTokenError(f"Inactive token {token_id}") from e
        raise

    data = r.json()
    price = float(data["price"])
    if price <= 0:
        logger.warning(
            "Brak płynności (cena=%.4f) dla tokena %s, side=%s. I consider the token/pair inactive.",
            price,
            token_id,
            side_for_price,
        )
        raise InactiveTokenError(
            f"No liquidity (price={price}) for token {token_id}, side={side_for_price}"
        )

    return price


def place_buy_usdc(
    client: ClobClient,
    cfg: Config,
    token_id: str,
    amount_usdc: float,
) -> dict:
    """
    Place a BUY order on token_id for a given USDC amount (market-style with limit & slippage).
    """
    transaction_id = f"BUY-{int(time.time())}-{token_id[:6]}"
    logger.info(
        "[%s] I am preparing to BUY for %.2f USDC for token %s",
        transaction_id,
        amount_usdc,
        token_id,
    )

    best_ask_price = get_best_price(cfg.clob_host, token_id, "SELL")
    logger.info("[%s] Best ASK (SELL side) = %.6f", transaction_id, best_ask_price)

    if amount_usdc <= 0:
        raise ValueError("The USDC amount must be > 0")

    TARGET_COST = round(amount_usdc, 2)
    PRICE_LIMIT = best_ask_price * (1 + cfg.slippage_percent)
    if PRICE_LIMIT > 0.99:
        PRICE_LIMIT = 0.99
    PRICE_LIMIT_ROUNDED = round(PRICE_LIMIT, 4)

    CALCULATED_SIZE = TARGET_COST / PRICE_LIMIT_ROUNDED
    FINAL_SIZE = math.ceil(CALCULATED_SIZE * 10000) / 10000.0
    FINAL_PRICE = round(TARGET_COST / FINAL_SIZE, 4)
    if FINAL_PRICE > PRICE_LIMIT_ROUNDED:
        FINAL_PRICE = PRICE_LIMIT_ROUNDED

    logger.info(
        "[%s] BUY params: price=%.4f, size=%.4f, cost~=%.4f",
        transaction_id,
        FINAL_PRICE,
        FINAL_SIZE,
        FINAL_PRICE * FINAL_SIZE,
    )

    args = OrderArgs(
        price=FINAL_PRICE,
        size=FINAL_SIZE,
        side=BUY,
        token_id=token_id,
    )
    signed = client.create_order(args)
    resp = client.post_order(signed, OrderType.GTC)
    logger.info("[%s] BUY posted. CLOB Response: %s", transaction_id, resp)
    return resp


def place_sell_all(
    client: ClobClient,
    cfg: Config,
    token_id: str,
    known_balance: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a SELL ALL order for a given token_id, either using a known_balance,
    or fetching available size from Data-API.
    """
    transaction_id = f"SELL-{int(time.time())}-{token_id[:6]}"
    logger.info("[%s] I am preparing SELL ALL for token %s", transaction_id, token_id)

    if known_balance is not None:
        balance = known_balance
    else:
        balance = get_token_size_from_positions(cfg, token_id)

    if balance <= 0:
        logger.warning(
            "[%s] No tokens %s for sale (saldo=%.8f).",
            transaction_id,
            token_id,
            balance,
        )
        return None

    FINAL_SIZE = math.floor(balance * 10000) / 10000.0
    if FINAL_SIZE <= 0:
        logger.warning(
            "[%s] To small balance (%f) after rounding to FINAL_SIZE=%f.",
            transaction_id,
            balance,
            FINAL_SIZE,
        )
        return None

    best_bid_price = get_best_price(cfg.clob_host, token_id, "BUY")
    PRICE_LIMIT = best_bid_price * (1 - cfg.slippage_percent)
    FINAL_PRICE = round(PRICE_LIMIT, 4)
    if FINAL_PRICE < 0.01:
        FINAL_PRICE = 0.01

    logger.info(
        "[%s] SELL params: price=%.4f, size=%.4f, value~=%.4f",
        transaction_id,
        FINAL_PRICE,
        FINAL_SIZE,
        FINAL_PRICE * FINAL_SIZE,
    )

    args = OrderArgs(
        price=FINAL_PRICE,
        size=FINAL_SIZE,
        side=SELL,
        token_id=token_id,
    )
    signed = client.create_order(args)
    resp = client.post_order(signed, OrderType.GTC)
    logger.info("[%s] Sell submitted. Answer: %s", transaction_id, resp)
    return resp


# ============================================================
# BOT
# ============================================================

class HedgedBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        logger.info("Creating a PolyAbi instance (pairs.json mode)...")
        self.client = init_clob_client(cfg)
        self.pairs: Dict[str, PairState] = load_pairs(cfg)
        self.last_pairs_reload_ts: float = time.time()

    # ---------------------- TREND / GRID ----------------------

    def _is_buy_allowed_by_trend(self, leg: TokenLegState) -> bool:
        """
        Check if a BUY on this leg is allowed based on trend from adx.json.
        If no trend info or no pair_symbol, BUY is allowed.
        """
        if not leg.pair_symbol:
            return True

        trend_map = load_trend_map(self.cfg.adx_file)
        trend = trend_map.get(leg.pair_symbol)
        if not trend:
            return True

        trend = trend.upper()
        leg_dir = label_to_direction(leg.label)
        if leg_dir is None:
            return True

        if trend == "UP" and leg_dir == "DOWN":
            logger.info(
                "TREND-FILTER: %s have trend=UP. Skipped BUY at token %s (label=%s, direction=DOWN).",
                leg.pair_symbol,
                leg.token_id,
                leg.label,
            )
            return False

        if trend == "DOWN" and leg_dir == "UP":
            logger.info(
                "TREND-FILTER: %s have trend=DOWN. Skipped BUY at token %s (label=%s, direction=UP).",
                leg.pair_symbol,
                leg.token_id,
                leg.label,
            )
            return False

        return True

    def _compute_next_buy_amount(self, leg: TokenLegState) -> float:
        """
        Compute the USDC amount for the next BUY on this leg using grid multiplier:
          amount = base * (mult ** total_buys)
        where total_buys is number of BUYs already executed on this leg.
        """
        base = leg.buy_amount_usdc
        mult = self.cfg.buy_grid_multiplier
        if mult <= 0:
            mult = 1.0
        amount = base * (mult ** leg.total_buys)
        logger.info(
            "GRID: token=%s, total_buys=%d -> amount_usdc=%.4f (base=%.4f, mult=%.3f)",
            leg.token_id,
            leg.total_buys,
            amount,
            base,
            mult,
        )
        return amount

    # ---------------------- RELOAD PAIRS.JSON -----------------

    def _reload_pairs_if_needed(self, now_ts: float):
        """
        Every cfg.pairs_reload_interval_sec seconds reload pairs.json
        and:
          - add new pairs (new time windows),
          - add new legs to existing pairs,
          - do NOT remove old pairs (let them naturally finish).
        """
        if now_ts - self.last_pairs_reload_ts < self.cfg.pairs_reload_interval_sec:
            return

        self.last_pairs_reload_ts = now_ts
        logger.info("RELOAD: I'm checking out new pairs in %s...", self.cfg.pairs_file)
        try:
            new_pairs = load_pairs(self.cfg)
        except Exception as e:
            logger.error("RELOAD: error while loading pairs.json: %s", e)
            return

        # Merge new pairs into existing ones
        for pair_key, new_pair in new_pairs.items():
            if pair_key in self.pairs:
                existing = self.pairs[pair_key]

                # Update static fields (slot times, symbol)
                existing.trade_start_str = new_pair.trade_start_str
                existing.trade_end_str = new_pair.trade_end_str
                if new_pair.pair_symbol:
                    existing.pair_symbol = new_pair.pair_symbol

                # Merge legs
                for token_id, new_leg in new_pair.legs.items():
                    if token_id in existing.legs:
                        leg = existing.legs[token_id]
                        # Update config-driven fields, keep runtime fields (buys/sells etc.)
                        leg.label = new_leg.label
                        leg.stoploss_price = new_leg.stoploss_price
                        leg.add_step_price = new_leg.add_step_price
                        leg.buy_amount_usdc = new_leg.buy_amount_usdc
                    else:
                        existing.legs[token_id] = new_leg
                        logger.info(
                            "RELOAD: a new leg was added token_id=%s to an existing pair %s",
                            token_id,
                            pair_key,
                        )
            else:
                self.pairs[pair_key] = new_pair
                logger.info(
                    "RELOAD: a new pair has been added %s (market_key=%s, slot=%s-%s, symbol=%s)",
                    pair_key,
                    new_pair.market_key,
                    new_pair.trade_start_str,
                    new_pair.trade_end_str,
                    new_pair.pair_symbol,
                )

    # ---------------------- PAIR TAKE-PROFIT ------------------

    def _check_market_pair_take_profit(self, now: datetime):
        """
        If total PnL% on a market (market_key) >= pair_take_profit_pct,
        SELL ALL tokens for that market, but only if the time window
        for that pair is still within the trading period.
        """
        try:
            positions = fetch_positions_simple(self.cfg)
        except Exception as e:
            logger.warning("Failed to retrieve items for PnL calculation par: %s", e)
            return

        # Group positions by market_key (conditionId or slug)
        markets_positions: Dict[str, List[dict]] = {}
        for p in positions:
            try:
                asset = p.get("asset")
                if not asset:
                    continue
                condition_id = p.get("conditionId")
                slug = p.get("slug")
                market_key = condition_id or slug or "unknown"
                p["_market_key"] = market_key
                markets_positions.setdefault(market_key, []).append(p)
            except Exception:
                continue

        for mk, pos_list in markets_positions.items():
            # mk is conditionId/slug; match it to PairState by market_key
            related_pairs = [
                pair for pair in self.pairs.values()
                if pair.market_key == mk and not pair.closed
            ]
            if not related_pairs:
                continue

            for pair_state in related_pairs:
                start_dt, end_dt = build_today_window(pair_state.trade_start_str, pair_state.trade_end_str)
                last_trade_dt = end_dt - timedelta(seconds=30)
                if not (start_dt <= now < last_trade_dt):
                    continue

                total_cost = 0.0
                current_value = 0.0
                for p in pos_list:
                    try:
                        size = float(p.get("size", 0.0) or 0.0)
                        avg_price = float(p.get("avgPrice", 0.0) or 0.0)
                        cur_price = float(p.get("curPrice", 0.0) or 0.0)
                        total_cost += size * avg_price
                        current_value += size * cur_price
                    except Exception:
                        continue

                if total_cost <= 0:
                    continue

                pnl_abs = current_value - total_cost
                pnl_pct = pnl_abs / total_cost

                logger.debug(
                    "PnL pary market_key=%s: total_cost=%.4f, current_value=%.4f, pnl=%.4f (%.2f %%)",
                    mk,
                    total_cost,
                    current_value,
                    pnl_abs,
                    pnl_pct * 100.0,
                )

                if pnl_pct >= self.cfg.pair_take_profit_pct:
                    logger.info(
                        "TP pair achieved: market_key=%s, pnl=%.4f (%.2f%%) >= %.2f%%. "
                        "I am closing all the legs of this market/windows (%s-%s).",
                        mk,
                        pnl_abs,
                        pnl_pct * 100.0,
                        self.cfg.pair_take_profit_pct * 100.0,
                        pair_state.trade_start_str,
                        pair_state.trade_end_str,
                    )

                    # SELL ALL for all tokens of this market (which are actually in positions)
                    asset_ids = {p.get("asset") for p in pos_list if p.get("asset")}
                    for token_id in asset_ids:
                        try:
                            resp = place_sell_all(self.client, self.cfg, token_id)
                            if resp:
                                # Update leg stats if present in the pair
                                for leg in pair_state.legs.values():
                                    if leg.token_id == token_id:
                                        leg.total_sells += 1
                                        leg.is_closed = True
                                logger.info(
                                    "TP-PARA: SELL ALL token=%s w rynku %s, resp=%s",
                                    token_id,
                                    mk,
                                    resp,
                                )
                        except InactiveTokenError:
                            logger.warning(
                                "TP-PARA: token %s from market %s not acitve at SELL – skipped.",
                                token_id,
                                mk,
                            )
                        except PolyApiException as e:
                            logger.error(
                                "TP-PARA: PolyApiException przy SELL token=%s, rynek=%s: %s",
                                token_id,
                                mk,
                                e,
                            )
                        except Exception as e:
                            logger.error(
                                "TP-PARA: Unexpected error on SELL token=%s, rynek=%s: %s",
                                token_id,
                                mk,
                                e,
                            )

                    pair_state.closed = True
                    logger.info(
                        "Market / window %s [%s-%s] market as closed by TP.",
                        mk,
                        pair_state.trade_start_str,
                        pair_state.trade_end_str,
                    )

    # ---------------------- STOPLOSS + ENTRY/GRID  ------------

    def _check_stoploss_and_sell(self, leg: TokenLegState, current_price: float):
        """
        STOP LOSS:
        - first check if we actually hold a positive size of this token,
        - only then compare current_price < stoploss_price,
        - this avoids spamming SL when no position is held.
        """
        if leg.is_closed:
            return

        # Check if we hold any position in this token
        size = get_token_size_from_positions(self.cfg, leg.token_id)
        if size <= 0:
            # No tokens – SL does not apply at the moment
            return

        if current_price < leg.stoploss_price:
            logger.info(
                "STOPLOSS hit: token=%s, label=%s, price=%.4f < stoploss=%.4f (market_key=%s, size=%.8f)",
                leg.token_id,
                leg.label,
                current_price,
                leg.stoploss_price,
                leg.market_key,
                size,
            )
            try:
                resp = place_sell_all(self.client, self.cfg, leg.token_id, known_balance=size)
                if resp:
                    leg.total_sells += 1
                    leg.is_closed = True
                    logger.info(
                        "SELL ALL done (SL): token=%s, total_sells=%d, leg closed.",
                        leg.token_id,
                        leg.total_sells,
                    )
            except InactiveTokenError:
                logger.warning(
                    "Token %s inactive when trying to SELL ALL – I mark the leg as closed.",
                    leg.token_id,
                )
                leg.is_closed = True
            except PolyApiException as e:
                logger.error("PolyApiException at SELL ALL %s: %s", leg.token_id, e)
            except Exception as e:
                logger.error("Unexpected error on SELL ALL %s: %s", leg.token_id, e)

    def _maybe_open_first_leg(self, pair: PairState, now: datetime, prices: Dict[str, float]):
        """
        NEW FIRST ENTRY LOGIC:
        - After delay, if we still don't have an active leg:
          * for each leg:
            - check trend (adx.json)
            - first BUY condition:
                current_price >= 0.50 + add_step_price
          * from all legs passing the condition, pick the one with the highest price
          * execute BUY for this leg:
                amount_usdc = base * mult**total_buys
        """
        if pair.active_token_id is not None:
            return  # already have a chosen direction

        candidates = []
        base_level = 0.5  # fixed base as requested: condition is price >= 0.5 + add_step_price

        for token_id, leg in pair.legs.items():
            if leg.is_closed:
                continue
            if token_id not in prices:
                continue

            current_price = prices[token_id]

            # Trend filter
            if not self._is_buy_allowed_by_trend(leg):
                continue

            threshold = base_level + leg.add_step_price

            logger.debug(
                "FIRST-ENTRY CHECK: token=%s, label=%s, price=%.4f, threshold=%.4f",
                token_id,
                leg.label,
                current_price,
                threshold,
            )

            if current_price >= threshold:
                candidates.append((token_id, leg, current_price))

        if not candidates:
            return

        # Choose the leg with the highest current price (strongest signal)
        best_token_id, best_leg, best_price = max(candidates, key=lambda x: x[2])

        amount_usdc = self._compute_next_buy_amount(best_leg)
        try:
            resp = place_buy_usdc(self.client, self.cfg, best_token_id, amount_usdc)
            if resp:
                best_leg.total_buys += 1
                best_leg.anchor_buy_price = best_price
                best_leg.last_buy_price = best_price
                minute_index = int(now.timestamp() // 60)
                best_leg.last_buy_minute_index = minute_index
                pair.active_token_id = best_token_id

                logger.info(
                    "FIRST-ENTRY BUY: market=%s, pair=%s, token=%s, label=%s, "
                    "price=%.4f, total_buys=%d",
                    pair.market_key,
                    pair.pair_symbol,
                    best_token_id,
                    best_leg.label,
                    best_price,
                    best_leg.total_buys,
                )
        except InactiveTokenError:
            logger.warning(
                "FIRST-ENTRY: token %s inactive when trying to BUY – I mark the leg as closed.",
                best_token_id,
            )
            best_leg.is_closed = True
        except PolyApiException as e:
            logger.error("FIRST-ENTRY: PolyApiException at BUY %s: %s", best_token_id, e)
        except Exception as e:
            logger.error("FIRST-ENTRY: Unexpected error on BUY %s: %s", best_token_id, e)

    def _maybe_grid_buy_active_leg(self, pair: PairState, now: datetime, prices: Dict[str, float]):
        """
        NEW GRID LOGIC FOR SUBSEQUENT BUYS:
        - Only for active leg (pair.active_token_id != None)
        - At most 1 BUY per minute per leg
        - Condition:
            current_price > last_buy_price  (price must be rising)
        - USDC amount:
            base * (mult ** total_buys)
        """
        if pair.active_token_id is None:
            return

        token_id = pair.active_token_id
        leg = pair.legs.get(token_id)
        if not leg or leg.is_closed:
            return

        if token_id not in prices:
            return

        current_price = prices[token_id]

        # If last_buy_price is missing (e.g. after restart), set it and skip BUY in this iteration
        if leg.last_buy_price is None:
            leg.last_buy_price = current_price
            logger.info(
                "GRID-FIX: missing last_buy_price for token=%s, set to %.4f and I'm skipping BUY in this iteration.",
                token_id,
                current_price,
            )
            return

        minute_index = int(now.timestamp() // 60)
        if leg.last_buy_minute_index == minute_index:
            # Already executed a BUY for this leg in this minute
            return

        # Price must be strictly higher than the last BUY price
        if current_price <= leg.last_buy_price:
            logger.debug(
                "GRID: token=%s, price=%.4f <= last_buy_price=%.4f -> brak dokupki.",
                token_id,
                current_price,
                leg.last_buy_price,
            )
            return

        # Trend filter
        if not self._is_buy_allowed_by_trend(leg):
            return

        amount_usdc = self._compute_next_buy_amount(leg)
        logger.info(
            "GRID BUY: market=%s, token=%s, label=%s, last_buy=%.4f, current=%.4f, amount=%.4f",
            pair.market_key,
            token_id,
            leg.label,
            leg.last_buy_price,
            current_price,
            amount_usdc,
        )

        try:
            resp = place_buy_usdc(self.client, self.cfg, token_id, amount_usdc)
            if resp:
                leg.total_buys += 1
                leg.last_buy_price = current_price
                leg.last_buy_minute_index = minute_index
                logger.info(
                    "GRID BUY wykonany: token=%s, total_buys=%d, last_buy_price=%.4f",
                    token_id,
                    leg.total_buys,
                    leg.last_buy_price,
                )
        except InactiveTokenError:
            logger.warning(
                "GRID BUY: token %s inactive when trying to BUY – I mark the leg as closed.",
                token_id,
            )
            leg.is_closed = True
        except PolyApiException as e:
            logger.error("GRID BUY: PolyApiException at BUY %s: %s", token_id, e)
        except Exception as e:
            logger.error("GRID BUY: Unexpected error on BUY %s: %s", token_id, e)

    # ---------------------- MAIN LOOP -------------------------

    def run(self):
        logger.info("Starting the main loop of PolyAbi (pairs.json + trade windows + dynamic reload)...")
        try:
            while True:
                loop_start = time.time()
                now = datetime.now()

                # 0) Dynamic reload of pairs.json (adding new pairs/legs)
                self._reload_pairs_if_needed(loop_start)

                # 1) Pair-level take-profit (only for active time windows)
                self._check_market_pair_take_profit(now)

                # 2) Per-pair trading logic
                for pair_key, pair in self.pairs.items():
                    if pair.closed:
                        continue

                    start_dt, end_dt = build_today_window(pair.trade_start_str, pair.trade_end_str)
                    last_trade_dt = end_dt - timedelta(seconds=30)
                    delay_end_dt = start_dt + timedelta(minutes=self.cfg.interval_delay_minutes)

                    if now < start_dt:
                        # Time window has not started yet
                        continue

                    if now >= end_dt:
                        # Window finished – never touch this pair again
                        if not pair.closed:
                            logger.info(
                                "Pair window %s (%s, %s) ended (%s-%s). "
                                "Stop monitoring.",
                                pair_key,
                                pair.market_key,
                                pair.pair_symbol,
                                pair.trade_start_str,
                                pair.trade_end_str,
                            )
                            pair.closed = True
                        continue

                    # Do not place new orders after last_trade_dt
                    trading_allowed = start_dt <= now < last_trade_dt

                    # 2a) Fetch current prices for legs
                    prices: Dict[str, float] = {}
                    inactive = False
                    for token_id, leg in pair.legs.items():
                        if leg.is_closed:
                            continue
                        try:
                            price = get_best_price(self.cfg.clob_host, token_id, "BUY")
                            prices[token_id] = price
                        except InactiveTokenError:
                            logger.warning(
                                "Token %s in pair  %s (pair_key=%s, market_key=%s) is inactive. "
                                "Cloase pair.",
                                token_id,
                                pair.pair_symbol,
                                pair_key,
                                pair.market_key,
                            )
                            inactive = True
                            break
                        except Exception as e:
                            logger.warning(
                                "Error price fetch for token_id=%s (pair_key=%s, market_key=%s): %s",
                                token_id,
                                pair_key,
                                pair.market_key,
                                e,
                            )
                            continue

                    if inactive:
                        for leg in pair.legs.values():
                            leg.is_closed = True
                        pair.closed = True
                        continue

                    if not prices:
                        continue

                    # 2b) STOPLOSS – only if trading is allowed (SL also sends orders)
                    if trading_allowed:
                        for token_id, leg in pair.legs.items():
                            if leg.is_closed:
                                continue
                            if token_id not in prices:
                                continue
                            self._check_stoploss_and_sell(leg, prices[token_id])

                    # 2c) Delay phase – we do not perform first entry or grid here
                    if not trading_allowed:
                        continue

                    if now < delay_end_dt:
                        # Optional: record slot_start_price only for logging / diagnostics
                        for token_id, leg in pair.legs.items():
                            if leg.is_closed:
                                continue
                            if token_id not in prices:
                                continue
                            if leg.slot_start_price is None:
                                leg.slot_start_price = prices[token_id]
                                logger.info(
                                    "DELAY: Set slot_start_price for token=%s to %.4f (pair_key=%s, market_key=%s)",
                                    token_id,
                                    leg.slot_start_price,
                                    pair_key,
                                    leg.market_key,
                                )
                        continue

                    # 2d) After delay – normal trading:
                    #    - first possible entry (first BUY) if none active
                    #    - then grid BUYs only for the active leg

                    if pair.active_token_id is None:
                        self._maybe_open_first_leg(pair, now, prices)
                    else:
                        self._maybe_grid_buy_active_leg(pair, now, prices)

                elapsed = time.time() - loop_start
                sleep_for = max(0.0, self.cfg.poll_interval_sec - elapsed)
                logger.debug("Loop time %.3fs, sleep for %.3fs", elapsed, sleep_for)
                time.sleep(sleep_for)

        except KeyboardInterrupt:
            logger.info("Manual end (Ctrl+C). Closing PolyAbi.")
        except Exception as e:
            logger.exception("Unexpected error in main loop: %s", e)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info("=== Start abi-bot (pairs.json + individual slots + delay + single-side grid + reload + new buy logic) ===")
    cfg = load_config("config.json")
    bot = HedgedBot(cfg)
    bot.run()
