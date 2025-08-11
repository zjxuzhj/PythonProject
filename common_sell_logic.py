from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

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
    volume: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma55: float
    up_limit_price: float
    down_limit_price: float

    # --- 前一交易日数据 ---
    prev_close: float
    prev_up_limit_price: float
    prev_down_limit_price: float

@dataclass
class PositionContext:
    """存放与特定持仓相关的信息"""
    hold_days: int
    # 新增：首板涨停日的数据，在买入时就应确定
    limit_up_day_data: Dict[str, float]
    # 新增：最近几日的成交量，用于判断是否缩量
    recent_volumes: List[float]


def get_sell_decision_refined(
        position_ctx: PositionContext,
        market_ctx: MarketDataContext,
        is_ma10_hold_mode: bool,  # 新增：接收当前是否为10日线持股模式
        config: SellStrategyConfig = SellStrategyConfig()
) ->Tuple[bool, str, bool]:
    """
    经过优化的卖出决策函数，增加了对'健康回调'形态的识别。
    """
    # 按优先级顺序检查卖出条件

    # 卖出条件1: 最大持有天数限制 ---
    if position_ctx.hold_days >= config.max_hold_days:
        return True, '持有超限', is_ma10_hold_mode

    # 卖出条件2: 跌停止损 (基于前一日状态) ---
    # 如果前一天收盘价已经低于或等于前一天的跌停价
    if market_ctx.prev_close <= market_ctx.prev_down_limit_price:
        return True, '跌停止损', is_ma10_hold_mode

    # 卖出条件3: 断板止盈 (基于前一日状态) ---
    # 如果前一天是涨停的
    if market_ctx.prev_close >= market_ctx.prev_up_limit_price:
        # 但今天收盘价没有涨停，则卖出
        if market_ctx.close < market_ctx.up_limit_price:
            return True, '断板止盈', is_ma10_hold_mode
        else:
            # 如果今天继续涨停，则继续持有
            return False, '', is_ma10_hold_mode

    # 卖出条件4: 首板炸板卖出，只要股票冲高到达8.7%，收盘没有涨停就卖出 ---
    price_at_9_percent_gain = market_ctx.prev_close * 1.087
    is_limit_touched = market_ctx.high >= price_at_9_percent_gain
    is_not_closed_at_limit = market_ctx.close < market_ctx.up_limit_price
    is_prev_not_limit = market_ctx.prev_close < market_ctx.prev_up_limit_price
    if is_limit_touched and is_not_closed_at_limit and is_prev_not_limit:
        return True, '炸板卖出', is_ma10_hold_mode

    # --- 2. 检查跌破MA5的条件 ---
    if is_ma10_hold_mode:
        # --- 已经是10日线持股模式 ---
        # 在此模式下，我们只关心是否跌破10日线
        if market_ctx.ma10 and market_ctx.close < market_ctx.ma10:
            # 跌破10日线，卖出
            return True, '跌破十日线(切换后)', True  # 模式状态返回True
        else:
            # 未跌破10日线，继续持有
            return False, '', True  # 模式状态保持True

    else:
        # --- 仍是标准的5日线持股模式 ---
        if market_ctx.ma5 and market_ctx.ma5 > 0:
            deviation = (market_ctx.close - market_ctx.ma5) / market_ctx.ma5
            if deviation <= config.ma_breakdown_threshold:
                # 触发了“跌破五日线”，进入“豁免审查”
                is_healthy, reason = check_healthy_pullback_exception(position_ctx, market_ctx)

                if is_healthy:
                    # **模式切换点！**
                    # 是健康回调，不卖出，并且从今天起，切换到10日线持股模式
                    return False, f'持有({reason},转10日线持股)', True  # 返回新的模式状态 True
                else:
                    # 不是健康回调，按原计划卖出
                    return True, '跌破五日线', False  # 模式状态保持 False

        # 如果所有条件都不满足，则继续持有，并保持当前模式状态不变
    return False, '', is_ma10_hold_mode


def check_healthy_pullback_exception(position_ctx: PositionContext, market_ctx: MarketDataContext) -> Tuple[bool, str]:
    """
    检查当前跌破MA5的情况是否满足“健康回调”的豁免条件。
    """
    # --- 条件1: 必须是缩量回调 ---
    # recent_volumes 包含当天的成交量，所以比较时要排除最后一个
    if len(position_ctx.recent_volumes) > 1 and market_ctx.volume > min(position_ctx.recent_volumes[:-1]):
        return False, "非缩量回调"  # 如果今天量不是近期最低，则不满足豁免条件

    # --- 条件2: 价格必须守住首板50%支撑位 ---
    limit_up_data = position_ctx.limit_up_day_data
    support_level = (limit_up_data['prev_close'] + limit_up_data['close']) / 2
    if market_ctx.close < support_level:
        return False, "跌破首板50%支撑"

    # --- 条件3: 均线系统必须提供支撑 ---
    short_term_mas = [market_ctx.ma10, market_ctx.ma20, market_ctx.ma30]
    # 检查均线是否有效（非0或None）
    if not all(short_term_mas):
        return False, "均线数据不全"

    # 3.1 均线必须在价格下方形成支撑
    # if not all(ma < market_ctx.close for ma in short_term_mas):
    #     return False, "均线未形成下方支撑"

    # 3.2 中短期均线必须粘合
    # 计算均线的极差率，小于2%认为粘合
    ma_spread = max(short_term_mas) - min(short_term_mas)
    if (ma_spread / market_ctx.close) > 0.02:  # 粘合度放宽到2.5%
        return False, "中短期均线不粘合"

    # 3.3 55日线趋势必须健康
    # MA55要么参与粘合，要么在更下方提供远期支撑
    ma55 = market_ctx.ma55
    is_ma55_in_cluster = abs(ma55 - market_ctx.ma30) / market_ctx.ma30 < 0.02  # 55线和30线差距2%内算粘合
    is_ma55_far_below = ma55 < min(short_term_mas) * 0.98  # 55线比均线簇最低点还低3%
    if not (is_ma55_in_cluster or is_ma55_far_below):
        return False, "55日线趋势不佳"

    # --- 所有豁免条件均满足 ---
    return True, "缩量回调,均线强支撑"
