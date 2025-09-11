from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Set

from stock_info import StockInfo


@dataclass
class SellStrategyConfig:
    """存放所有卖出策略的参数，便于统一调整。"""
    max_hold_days: int = 15
    ma_breakdown_threshold: float = -0.004  # 跌破MA5卖出阈值: (收盘价 - MA5) / MA5 <= -0.4%
    postpone_sell_lower_bound: float = -0.09  # 延迟卖出的跌幅下限 (例如, -9%)
    postpone_sell_upper_bound: float = -0.08  # 延迟卖出的跌幅上限 (例如, -8%)
    manual_override_stocks: Set[str] = field(default_factory=set)

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
    stock_info: StockInfo,
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
    if stock_info.code in config.manual_override_stocks:
        return False, '手动操作'

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

    # 卖出条件4: 首板炸板卖出，只要股票冲高到达8.7%，收盘没有涨停就卖出 ---
    price_at_9_percent_gain = market_data.prev_close * 1.087
    is_limit_touched = market_data.high >= price_at_9_percent_gain
    is_not_closed_at_limit = market_data.close < market_data.up_limit_price
    is_prev_not_limit = market_data.prev_close < market_data.prev_up_limit_price

    if is_limit_touched and is_not_closed_at_limit and is_prev_not_limit:
        return True, '炸板卖出'

    # 卖出条件5:: 跌破五日线卖出 ---
    # 确保 MA5 是有效值
    if market_data.ma5 and market_data.ma5 > 0:
        deviation = (market_data.close - market_data.ma5) / market_data.ma5
        if deviation <= config.ma_breakdown_threshold:
            # 检查当天是否出现大幅下跌
            daily_change = (market_data.close - market_data.prev_close) / market_data.prev_close
            if config.postpone_sell_lower_bound < daily_change <= config.postpone_sell_upper_bound:
                # 核心逻辑：当日跌幅在8%到9%之间，则延迟一天卖出
                position_info['force_sell_next_day'] = True
                return False, ''  # 今天不卖
            else:
                return True, '跌破五日线'

    # --- 卖出条件6: 延迟后的强制卖出 ---
    if position_info.get('force_sell_next_day', False):
        return True, '大跌后次日卖出'

    # 如果所有条件都不满足，则继续持有
    return False, ''
