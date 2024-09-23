from enum import Enum


class RunMode(str, Enum):
    """
    Bot running mode (backtest, hyperopt, ...)
    can be "live", "dry-run", "backtest", "edge", "hyperopt".
    """

    LIVE = "live"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"
    EDGE = "edge"
    HYPEROPT = "hyperopt"
    UTIL_EXCHANGE = "util_exchange"
    UTIL_NO_EXCHANGE = "util_no_exchange"
    PLOT = "plot"
    WEBSERVER = "webserver"
    OTHER = "other"


TRADE_MODES = [RunMode.LIVE, RunMode.DRY_RUN]
OPTIMIZE_MODES = [RunMode.BACKTEST, RunMode.EDGE, RunMode.HYPEROPT]
NON_UTIL_MODES = TRADE_MODES + OPTIMIZE_MODES
