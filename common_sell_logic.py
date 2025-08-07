from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class SellStrategyConfig:
    """存放所有卖出策略的参数，便于统一调整。"""
    max_hold_days: int = 15
    ma_breakdown_threshold: float = -0.004  # 跌破MA5卖出阈值: (收盘价 - MA5) / MA5 <= -0.4%

@dataclass
class MarketDataContext:
    # --- 当日或实时数据 ---
    high: float
    close: float
    ma5: float
    up_limit_price: float
    down_limit_price: float

    # --- 前一交易日数据 ---
    prev_close: float
    prev_up_limit_price: float
    prev_down_limit_price: float


def get_sell_decision(
    position_info: Dict[str, Any],
    market_data: MarketDataContext,
    config: SellStrategyConfig = SellStrategyConfig()
) -> Tuple[bool, str]:
    """
    通用的卖出决策函数。
    根据持仓信息和市场数据，判断是否应该卖出。

    :param position_info: 包含持仓信息的字典，必须包含 'hold_days' (int)。
    :param market_data: 包含当前和前一交易日市场数据的上下文对象。
    :param config: 卖出策略的配置参数。
    :return: 一个元组 (should_sell, reason)。
             - should_sell (bool): True表示应该卖出，False则不卖。
             - reason (str): 卖出原因，如果不卖出则为 None。
    """
    # 按优先级顺序检查卖出条件

    # 卖出条件1: 最大持有天数限制 ---
    if position_info['hold_days'] >= config.max_hold_days:
        return True, '持有超限'

    # 卖出条件2: 跌停止损 (基于前一日状态) ---
    # 如果前一天收盘价已经低于或等于前一天的跌停价
    if market_data.prev_close <= market_data.prev_down_limit_price:
        return True, '跌停止损'

    # 卖出条件3: 断板止盈 (基于前一日状态) ---
    # 如果前一天是涨停的
    if market_data.prev_close >= market_data.prev_up_limit_price:
        # 但今天收盘价没有涨停，则卖出
        if market_data.close < market_data.up_limit_price:
            return True, '断板止盈'
        else:
            # 如果今天继续涨停，则继续持有
            return False, ''

    # 卖出条件4: 首板炸板卖出 (基于当日状态) ---
    is_limit_touched = market_data.high >= market_data.up_limit_price
    is_not_closed_at_limit = market_data.close < market_data.up_limit_price
    is_prev_not_limit = market_data.prev_close < market_data.prev_up_limit_price

    if is_limit_touched and is_not_closed_at_limit and is_prev_not_limit:
        return True, '炸板卖出'

    # 卖出条件5:: 跌破五日线卖出 ---
    # 确保 MA5 是有效值
    if market_data.ma5 and market_data.ma5 > 0:
        deviation = (market_data.close - market_data.ma5) / market_data.ma5
        if deviation <= config.ma_breakdown_threshold:
            return True, '跌破五日线'

    # 如果所有条件都不满足，则继续持有
    return False, ''
