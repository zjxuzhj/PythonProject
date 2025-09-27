from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Set

from stock_info import StockInfo
import pandas as pd  # <--- 引入 pandas
from typing import Optional  # <--- 引入 Optional


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
    low: float
    open: float
    close: float
    ma5: float
    up_limit_price: float
    down_limit_price: float

    # --- 前一交易日数据 ---
    prev_close: float
    prev_up_limit_price: float
    prev_down_limit_price: float

    # --- 存放包含今日实时数据的完整DataFrame ---
    today_df: Optional[pd.DataFrame] = None


def get_sell_decision(
        stock_info: StockInfo,
        position_info: Dict[str, Any],
        market_data: MarketDataContext,
        config: SellStrategyConfig = SellStrategyConfig(),
        use_optimized_logic: bool = False
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

    # --- 卖出条件6: 防双头(M头)形态卖出 ---
    # 此条件仅在T+2买入後的次日(即T+3, 持有1天)生效
    # 对受影响交易的平均收益影响: +3.81 %，正向: 负向 6: 0，优化对总收益率的净影响: +23.55 %
    if position_info.get('hold_days') == 1 and market_data.today_df is not None:
        df = market_data.today_df
        if len(df) > 24:
            # 1. 定位T+0(首板日), T+1, T+2, T+3(今天)
            # T+3是今天，即df的最后一行
            today_data = df.iloc[-1]
            # T+2是昨天，即df的倒数第二行
            p2_data = df.iloc[-2]
            # 从昨天(T+2)往前找最近的涨停日(T+0)
            historical_df = df.iloc[:-2]
            limit_up_days = historical_df[historical_df['is_limit']]
            if not limit_up_days.empty:
                last_limit_up_date = limit_up_days.index[-1]
                limit_up_day_idx = df.index.get_loc(last_limit_up_date)
                # 确保T+0, T+1, T+2, T+3是连续的 (今天的索引 = T+0索引+3)
                if df.index.get_loc(today_data.name) == limit_up_day_idx + 3:
                    p1_data = df.iloc[limit_up_day_idx + 1]
                    # 3. 获取涨停前20日的最高点及当日量能
                    lookback_window = df.iloc[max(0, limit_up_day_idx - 20): limit_up_day_idx]
                    if not lookback_window.empty:
                        peak_day_loc = lookback_window['high'].idxmax()
                        prev_peak_high = df.loc[peak_day_loc, 'high']
                        prev_peak_volume = df.loc[peak_day_loc, 'volume']
                        prev_peak_low = df.loc[peak_day_loc, 'low']
                        midpoint_price = (prev_peak_high + prev_peak_low) / 2
                        # 条件A: T+1是阴线, 且最高价和量能均未超过前期高点
                        cond_p1_is_weak = (p1_data['close'] < p1_data['open'] and
                                           p1_data['high'] < prev_peak_high and
                                           p1_data['volume'] < prev_peak_volume and
                                           p1_data['high'] > midpoint_price)
                        # 条件B: T+3相对于T+2放量、价升，但都低于T+1
                        cond_p3_attack_failed = (today_data['volume'] > p2_data['volume'] and
                                                 today_data['high'] > p2_data['high'] and
                                                 today_data['volume'] < p1_data['volume'] and
                                                 today_data['high'] < p1_data['high'])
                        if cond_p1_is_weak and cond_p3_attack_failed:
                            return True, '疑似双头卖出'

    # 卖出条件5:: 跌破五日线卖出 ---
    # 确保 MA5 是有效值
    if market_data.ma5 and market_data.ma5 > 0:
        deviation = (market_data.close - market_data.ma5) / market_data.ma5
        if deviation <= config.ma_breakdown_threshold:
            first_limit_up_price = position_info.get('first_limit_up_price')
            # if use_optimized_logic and position_info['hold_days'] < 5:
            #     df = market_data.today_df
            #     # 确保df有效且有足够数据
            #     if df is not None and not df.empty:
            #         # 从当前模拟日(最后一行)往前寻找最近的涨停
            #         historical_df = df.iloc[:-1]  # 排除当前这一天
            #         limit_up_days = historical_df[historical_df['is_limit']]
            #
            #         if not limit_up_days.empty:
            #             # 找到了涨停日，取最近的一个
            #             last_limit_up_day_index = limit_up_days.index[-1]
            #             limit_up_day_idx = df.index.get_loc(last_limit_up_day_index)
            #
            #             # 检查是否有足够的数据（涨停前需要有3天）
            #             if limit_up_day_idx >= 3:
            #                 ma30_support_found = True
            #                 # 循环检查涨停前的3天
            #                 for i in range(1, 4):
            #                     prev_day_data = df.iloc[limit_up_day_idx - i]
            #                     low_price = prev_day_data['low']
            #                     close_price = prev_day_data['close']
            #                     ma30 = prev_day_data['ma30']
            #
            #                     # 如果MA30无效，或不满足支撑条件，则豁免失败
            #                     if pd.isna(ma30) or not (low_price < ma30 and close_price > ma30):
            #                         ma30_support_found = False
            #                         break
            #
            #                 # 如果连续三天的MA30支撑都得到确认
            #                 if ma30_support_found:
            #                     position_info['force_sell_next_day'] = True
            #                     position_info['postponed_reason'] = '首板价支撑次日卖'
            #                     # 给予二次机会，本次不卖出
            #                     return False, 'MA30支撑二次机会'

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
