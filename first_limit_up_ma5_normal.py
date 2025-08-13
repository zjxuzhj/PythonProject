import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import getAllStockCsv
from common_sell_logic import get_sell_decision, MarketDataContext


@dataclass
class StrategyConfig:
    """集中存放所有策略参数，便于统一调整。"""
    # --- 数据设置 ---
    USE_2019_DATA: bool = False  # False 使用2024年数据, True 使用2019年数据

    # --- 首板涨停识别参数 ---
    MARKET_LIMIT_RATES = {'主板': 0.10, '创业板': 0.20, '科创板': 0.20}
    MAX_MARKET_CAP_BILLIONS = 250  # 条件7: 最大市值过滤（单位：亿）

    # --- 买入参数 ---
    PREDICT_PRICE_INCREASE_RATIO = 1.04  # 用于预测MA5的价格涨幅

    # --- 双头形态检测参数 ---
    DOUBLE_TOP_CHECK_DAYS = 40  # 检测最近40日的双头形态
    DOUBLE_TOP_PRICE_TOLERANCE = 0.04  # 两个高点价格差异容忍度（3%）
    DOUBLE_TOP_VOLUME_DECREASE_THRESHOLD = 0.8  # 第二头部成交量需小于第一头部的阈值

    # ---  涨停日成交量校验参数 ---
    LIMIT_UP_VOLUME_LOOKBACK_DAYS: int = 100  # 寻找前期高点以校验成交量的回看天数
    LIMIT_UP_VOLUME_THRESHOLD: float = 0.9  # 涨停日成交量不得超过前高成交量的阈值


def simulate_ma5_order_prices(df, current_day, config: StrategyConfig, lookback_days=5):
    """模拟预测买入日MA5值，然后计算挂单价格"""
    current_idx = df.index.get_loc(current_day)

    if current_idx < lookback_days:
        return None

    prev_data = df.iloc[current_idx - lookback_days: current_idx]
    predict_ratio = config.PREDICT_PRICE_INCREASE_RATIO

    try:
        price_ma5 = modify_last_days_and_calc_ma5(prev_data, predict_ratio)['MA5'].iloc[-1]
        return price_ma5
    except Exception as e:
        print(f"预测MA5失败: {e}")
        return None


def get_stock_data(symbol, config: StrategyConfig):
    """带本地缓存的数据获取"""
    date_str = "20190101" if config.USE_2019_DATA else "20240201"
    cache_dir = "back_test_data_cache" if config.USE_2019_DATA else "data_cache"
    file_path = os.path.join(cache_dir, f"stock_{symbol}_{date_str}.parquet")

    if not os.path.exists(file_path):
        print(f"数据文件未找到: {symbol} at {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        print(f"已从缓存加载数据: {symbol}")
        return df, True
    except Exception as e:
        print(f"读取缓存文件失败 {file_path}: {e}")
        return None


def get_market_type(symbol: str) -> str:
    """根据股票代码判断所属板块。"""
    if symbol.startswith(("688", "689")):
        return "科创板"
    if symbol.startswith(("300", "301")):
        return "创业板"
    return "主板"


def is_valid_first_limit_up_day(df: pd.DataFrame, day: pd.Timestamp, code: str, config: StrategyConfig,
                                query_tool) -> bool:
    """
    检查给定的某一天是否是符合所有条件的首板涨停日。默认获得涨停后一日的数据
    :return: 如果通过所有检查, 返回True; 如果任何一个检查失败, 返回False。
    """
    limit_up_day_date = day  # 涨停日日期
    limit_up_day_idx = df.index.get_loc(limit_up_day_date)  # 涨停日行号
    limit_up_day_data = df.loc[limit_up_day_date]  # 涨停日数据
    limit_up_day_price = limit_up_day_data['close']  # 涨停日收盘价，重要支撑位
    limit_up_day_volume = limit_up_day_data['volume']
    day_minus_1_idx = limit_up_day_idx - 1  # 涨停前一日
    day_minus_2_idx = limit_up_day_idx - 2  # 涨停前二日
    day_plus_1_idx = limit_up_day_idx + 1  # 涨停后一日
    day_plus_2_idx = limit_up_day_idx + 2  # 涨停后第二日
    day_plus_3_idx = limit_up_day_idx + 3

    if day_plus_1_idx >= len(df):
        return False

    day_plus_1_day_date = df.index[day_plus_1_idx]
    day_plus_1_data = df.iloc[day_plus_1_idx]

    # 条件0：排除前一日涨停
    if day_minus_1_idx >= 0:
        prev_day_data = df.iloc[day_minus_1_idx]
        if prev_day_data['is_limit']:
            return False

    # 条件1：排除后一日涨停 (连板)
    if day_plus_1_data['is_limit']:
        return False

    # 条件3：涨停后第一天量能过滤条件
    if (day_plus_1_data['volume'] >= limit_up_day_volume * 3.6) and (
            day_plus_1_data['close'] < day_plus_1_data['open']):
        return False

    # 条件4：排除涨停和涨停后一天的交易量是前120天中最大交易量四倍以上的股票
    lookback_days = 120
    if day_plus_1_idx >= lookback_days:
        lookback_window = df.iloc[limit_up_day_idx - lookback_days: limit_up_day_idx]
        if not lookback_window.empty:
            prev_high_volume_day_loc = lookback_window['volume'].idxmax()
            prev_high_volume = df.loc[prev_high_volume_day_loc, 'volume']
            max_volume = limit_up_day_volume
            if day_plus_1_data['volume'] > limit_up_day_volume:
                max_volume = day_plus_1_data['volume']
            if max_volume > prev_high_volume * 4:
                # print("排除涨停和涨停后一天的交易量是前120天中最大交易量四倍以上的股票")
                return False

    # 条件6：排除涨停后第一天放量阳线+涨停后第二天低开未收复前日实体中点的情况
    if day_plus_2_idx < len(df):
        second_day = df.index[day_plus_2_idx]
        second_day_data = df.loc[second_day]
        # 条件6-1：涨停后第一天为放量实体阳线（成交量>首板日且实体占比在总的价格范围的>50%）
        volume_condition = (day_plus_1_data['volume'] > limit_up_day_volume * 1.5)  # 放量1.5倍
        price_range = day_plus_1_data['high'] - day_plus_1_data['low']
        if abs(price_range) < 1e-5:  # 若最高价=最低价（一字线），实体占比无法计算，直接排除
            candle_condition = False
        else:
            body_ratio = (day_plus_1_data['close'] - day_plus_1_data['open']) / price_range
            candle_condition = (body_ratio > 0.5) and (day_plus_1_data['close'] > day_plus_1_data['open'])
        # 条件6-2：涨停后第二天低开且未收复前日实体中点
        midpoint = (day_plus_1_data['close'] + day_plus_1_data['open']) / 2  # 前日阳线实体中点
        low_open_condition = (second_day_data['open'] < day_plus_1_data['close'])  # 低开
        recover_condition = (second_day_data['close'] < midpoint)  # 盘中最高点未达中点

        if volume_condition and candle_condition and low_open_condition and recover_condition:
            return False

    # 条件12: 排除前期高位连板后，出现的缩量反抽形态，如果回调期最低点高于起涨的最低点，排除说明的确是在反抽，不过是缩量反抽，不需要参与
    lookback_period_12 = 10
    if limit_up_day_idx > lookback_period_12 and day_plus_1_idx < len(df):
        lookback_df = df.iloc[limit_up_day_idx - lookback_period_12: limit_up_day_idx]
        limit_up_days = lookback_df[lookback_df['is_limit']]

        if not limit_up_days.empty:
            # 使用np.diff()和cumsum()来分组连续的涨停天数
            limit_up_indices = [df.index.get_loc(d) for d in limit_up_days.index]
            groups = (np.diff(limit_up_indices, prepend=np.nan) != 1).cumsum()

            # 遍历所有连板分组
            for group_id in np.unique(groups):
                streak_indices = [idx for i, idx in enumerate(limit_up_indices) if groups[i] == group_id]

                # 子条件1: 必须是大于2天的连板
                if len(streak_indices) > 2:
                    streak_df = df.iloc[streak_indices]
                    peak_day_timestamp = streak_df['high'].idxmax()
                    peak_day_data = df.loc[peak_day_timestamp]
                    peak_day_idx = df.index.get_loc(peak_day_timestamp)
                    limit_up_day_data = df.iloc[limit_up_day_idx]
                    day_plus_1_data = df.iloc[day_plus_1_idx]
                    # 子条件3: 涨停日和次日成交量均小于峰值日成交量
                    vol_condition = (limit_up_day_data['volume'] < peak_day_data['volume'] and
                                     day_plus_1_data['volume'] < peak_day_data['volume'])
                    # 子条件4: 次日最高价低于峰值日或后一日的最高价
                    price_condition = day_plus_1_data['high'] < peak_day_data['high']
                    pullback_is_shallow = False
                    preceding_lookback_days = 10
                    streak_start_idx = streak_indices[0]
                    if streak_start_idx > preceding_lookback_days:
                        # 找到连板行情启动前的最低点 (A点)
                        preceding_window = df.iloc[streak_start_idx - preceding_lookback_days: streak_start_idx]
                        preceding_low = preceding_window['low'].min()
                        # 获取连板行情的最高点 (B点)
                        peak_high = peak_day_data['high']
                        # 找到回调期的最低点 (D点)
                        peak_day_idx = df.index.get_loc(peak_day_timestamp)
                        adjustment_window = df.iloc[peak_day_idx + 1: limit_up_day_idx]
                        if not adjustment_window.empty:
                            lowest_low_in_between = adjustment_window['low'].min()
                            # 计算价格支撑位 (E点) 并进行判断
                            total_rise = peak_high - preceding_low
                            if total_rise > 0 and lowest_low_in_between > preceding_low + total_rise * 0.1:
                                pullback_is_shallow = True

                    if vol_condition and price_condition and pullback_is_shallow:
                        print(
                            f"[{code}] 在 {day.date()} 的涨停被条件12排除：疑似对 {peak_day_timestamp.date()} 的高位炸板进行缩量反抽。")
                        return False

    # 条件7：前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
    lookback_period_5 = 5
    if limit_up_day_idx >= lookback_period_5:
        pre5_start = df.index[df.index.get_loc(day) - 5]
        pre5_close = df.loc[pre5_start, 'close']
        if pre5_close != 0:
            total_change = (limit_up_day_price - pre5_close) / pre5_close * 100
            if total_change >= 15:
                return False

    # 条件8：排除10日内涨停次数过多的股票
    lookback_period = 10
    if limit_up_day_idx >= lookback_period:
        lookback_data = df.iloc[limit_up_day_idx - lookback_period: limit_up_day_idx]
        limit_up_count = (lookback_data['close'] >= lookback_data['limit_price']).sum()
        if limit_up_count >= 4:
            return False

    # 条件9：排除10日内存在跌破一半的涨停
    lookback_period = 4
    if limit_up_day_idx >= lookback_period:
        lookback_data = df.iloc[limit_up_day_idx - lookback_period: limit_up_day_idx]
        recent_limit_ups = lookback_data[lookback_data['close'] >= lookback_data['limit_price']]
        if not recent_limit_ups.empty:
            last_limit_up_day = recent_limit_ups.index[-1]
            # 获取最近涨停日的前一日收盘价
            prev_close_of_last_limit_up = df.loc[last_limit_up_day, 'prev_close']
            price_floor = prev_close_of_last_limit_up * 1.035
            intermediate_days_loc = slice(df.index.get_loc(last_limit_up_day) + 1, limit_up_day_idx)
            intermediate_days = df.iloc[intermediate_days_loc]

            if not intermediate_days.empty:
                min_low_in_between = intermediate_days['low'].min()
                if min_low_in_between < price_floor:
                    return False

    # 条件10：排除市值大于250亿的股票
    market_value = query_tool.get_stock_market_value(code)
    if market_value > config.MAX_MARKET_CAP_BILLIONS:
        return False

    # 条件11：排除特定题材
    theme = query_tool.get_theme_by_code(code)
    name = query_tool.get_name_by_code(code)
    if "证券" in name or "金融" in name or "证券" in theme or "金融" in theme:  # 牛市旗手，跟不上，不参与
        return False
    if "外贸" in theme:
        return False

    # 条件13: 排除涨停前10日内，先出现至少2连板后又出现至少2连跌停的极端走势
    lookback_period_13 = 10
    if limit_up_day_idx >= lookback_period_13:
        window = df.iloc[limit_up_day_idx - lookback_period_13: limit_up_day_idx]
        # 检查窗口内是否存在2连板,找出所有2连板结束的位置
        up_streaks = window['is_limit'].rolling(window=2).sum()
        up_streak_end_indices = up_streaks[up_streaks >= 2].index

        if not up_streak_end_indices.empty:
            for up_streak_end_date in up_streak_end_indices:
                up_streak_end_pos = window.index.get_loc(up_streak_end_date)
                search_for_down_streak_window = window.iloc[up_streak_end_pos + 1:]
                if len(search_for_down_streak_window) >= 2:
                    down_streaks = search_for_down_streak_window['is_limit_down'].rolling(window=2).sum()
                    if (down_streaks >= 2).any():
                        # 如果在2连板之后，确实找到了2连跌停，则排除
                        print(f"[{code}] 在 {day.date()} 排除：涨停前10日内出现先2连板后2连跌的极端走势。")
                        return False

    # 条件14: 排除涨停前长期横盘，波动极小的股票
    low_volatility_lookback = 40
    if day_plus_1_idx < len(df):
        day_plus_1_day_date = df.index[day_plus_1_idx]
        if limit_up_day_idx >= low_volatility_lookback:
            lookback_window = df.iloc[limit_up_day_idx - low_volatility_lookback: limit_up_day_idx]
            valid_prev_close = lookback_window['prev_close'].replace(0, np.nan).dropna()
            if not valid_prev_close.empty:
                daily_amplitude = (lookback_window['high'].loc[valid_prev_close.index] - lookback_window['low'].loc[
                    valid_prev_close.index]) / valid_prev_close
                volatile_days_count = (daily_amplitude < 0.035).sum()
                # 如果在过去40天里，振幅超过3.5%的天数少于3天，则认为波动过小，排除,同时要求第一天的量小于涨停日的二点五倍
                first_day_volume = df.loc[day_plus_1_day_date, 'volume']
                if volatile_days_count > 37 and first_day_volume < limit_up_day_volume * 2.5:
                    # print(f"[{code}] 在 {day.date()} 的涨停被排除：前{low_volatility_lookback}日波动过小({volatile_days_count}天振幅 > 3.5%)。")
                    return False

    return True


def is_valid_buy_opportunity(df: pd.DataFrame, limit_up_day_idx: int, offset: int, code: str,
                             config: StrategyConfig) -> bool:
    """
    检查从首板日到潜在买入日之间，是否满足所有的买入前置条件。

    :param df: 包含所有数据的DataFrame
    :param limit_up_day_idx: 首板涨停日的整数索引
    :param offset: 潜在买入日的偏移整数索引
    :return: 如果所有条件都满足，返回 True，否则返回 False
    """
    limit_up_day_date = df.index[limit_up_day_idx]
    limit_up_day_price = df.iloc[limit_up_day_idx]['close']  # 涨停日收盘价，重要支撑位
    limit_up_day_volume = df.iloc[limit_up_day_idx]['volume']
    day_minus_1_idx = limit_up_day_idx - 1  # 涨停前一日
    day_minus_1_data = df.iloc[limit_up_day_idx - 1]
    day_minus_2_idx = limit_up_day_idx - 2  # 涨停前二日
    day_plus_1_idx = limit_up_day_idx + 1  # 涨停后一日
    day_plus_2_idx = limit_up_day_idx + 2  # 涨停后第二日
    day_plus_3_idx = limit_up_day_idx + 3
    day_plus_4_idx = limit_up_day_idx + 4
    day_plus_1_day_date = df.index[day_plus_1_idx]

    potential_buy_day_idx = limit_up_day_idx + offset

    # 在第四天时，排除已经在第二天买入的股票
    if offset == 4:
        day_2_idx = limit_up_day_idx + 2
        if day_2_idx < len(df):
            day_2_timestamp = df.index[day_2_idx]
            day_2_data = df.iloc[day_2_idx]
            hypothetical_price_on_day_2 = simulate_ma5_order_prices(df, day_2_timestamp, config)
            if hypothetical_price_on_day_2 is not None:
                if day_2_data['low'] <= hypothetical_price_on_day_2:
                    return False

    # 新增买前条件: 排除T+2日相对T+1日收盘价大幅低开(>3%)的情况
    # if offset >= 2:
    #     # 必须确保T+2和T+1的数据都存在，防止索引越界
    #     if day_plus_2_idx < len(df) and day_plus_1_idx < len(df):
    #         day_plus_1_close = df.iloc[day_plus_1_idx]['close']
    #         day_plus_2_open = df.iloc[day_plus_2_idx]['open']
    #
    #         # 如果T+2开盘价 < T+1收盘价的97%，则认为低开幅度过大，排除
    #         if day_plus_2_open > (day_plus_1_close * 1.05):
    #             return True

    # 买前条件1: 检查在首板日和买入日之间，是否出现了新的涨停
    for i in range(1, offset):
        check_day = df.index[limit_up_day_idx + i]
        if df.loc[check_day, 'is_limit']:
            return False

    # 买前条件2: 检查在首板日和买入日之间，收盘价是否始终高于首板收盘价
    for i in range(1, offset):
        check_day = df.index[limit_up_day_idx + i]
        if df.loc[check_day, 'close'] < limit_up_day_price:
            return False

    # 买前条件3: 检查到买入日为止，MA5数据是否有效（非空值）
    ma5_data = df['ma5'].iloc[limit_up_day_idx: potential_buy_day_idx + 1]
    if ma5_data.isnull().any():
        return False

    # 买前条件4: 排除买入前日收盘价>80的股票
    latest_close = df.iloc[potential_buy_day_idx - 1]['close']
    if latest_close > 80:
        return False

    if offset > 1:
        # 买前条件5：排除一字板，且排除最近20天内在20日线下超过一次的票，筛选出那些在涨停前趋势保持良好、没有经历深度或反复调整的股票
        lookback_days = 20
        if limit_up_day_idx < lookback_days:  # 数据不足，无法判断，直接排除
            return False

        lookback_window = df.iloc[limit_up_day_idx - lookback_days: limit_up_day_idx]
        breakdown_count = (lookback_window['close'] < lookback_window['ma20']).sum()

        day_minus_2_data = df.iloc[limit_up_day_idx - 2]
        is_prev_day_below_ma5 = day_minus_1_data['close'] < day_minus_1_data['ma5']
        is_day_minus_2_below_ma5 = day_minus_2_data['close'] < day_minus_2_data['ma5']

        if df.loc[limit_up_day_date, 'is_limit'] and df.loc[limit_up_day_date, 'low'] == df.loc[
            limit_up_day_date, 'close'] and breakdown_count > 1 and is_prev_day_below_ma5 and is_day_minus_2_below_ma5:
            print("排除首板是一字板的")
            return False

    # 买前条件6: 排除涨停后三天的最高价低于40天内最高价且距离小于1%，并且第三天冲高回落超过4%
    if offset > 3:
        pressure_test_met = False
        lookback_days_6b = 40
        lookback_days_6b_2 = 5
        if limit_up_day_idx >= lookback_days_6b:
            hist_window = df.iloc[limit_up_day_idx - lookback_days_6b: limit_up_day_idx - lookback_days_6b_2]
            prev_40d_high = hist_window['high'].max()
            high_t1 = df.iloc[day_plus_1_idx]['high']
            high_t2 = df.iloc[day_plus_2_idx]['high']
            high_t3 = df.iloc[day_plus_3_idx]['high']
            peak_T1_T2_T3 = max(high_t1, high_t2, high_t3)
            if peak_T1_T2_T3 < prev_40d_high and (prev_40d_high - peak_T1_T2_T3) / prev_40d_high < 0.01:
                pressure_test_met = True
        weakness_confirmed = False
        if pressure_test_met:
            day_plus_3_data = df.iloc[day_plus_3_idx]
            high_t3 = day_plus_3_data['high']
            close_t3 = day_plus_3_data['close']
            if high_t3 > 0:
                fallback_ratio = (high_t3 - close_t3) / high_t3
                if fallback_ratio > 0.04:
                    weakness_confirmed = True

        if pressure_test_met and weakness_confirmed:
            return False

        # 买前条件7： 检查涨停后第一天，第二天，第三天是否受到前期双头颈线压制
        for i in range(1, offset):
            check_day_idx = limit_up_day_idx + i
            if check_day_idx >= len(df):
                continue
            is_resisted, neckline = check_double_top_neckline_resistance(df, check_day_idx, config)
            if is_resisted:
                check_day_date = df.index[check_day_idx].date()
                print(f"[{code}] 在涨停后第{i}天({check_day_date})触及双头颈线 {neckline:.2f} 回落，放弃买入机会。")
                return False

        # 买前条件8： 排除T+1,T+2连续创40日新高但巨量回落的“双顶出货”形态
        lookback_days = 40
        # 必须有足够的回看周期和未来两天的数据
        if limit_up_day_idx >= lookback_days and day_plus_2_idx < len(df):
            hist_window = df.iloc[limit_up_day_idx - lookback_days: limit_up_day_idx]
            first_day_data = df.iloc[day_plus_1_idx]
            high_t1 = first_day_data['high']
            close_t1 = first_day_data['close']
            volume_t1 = first_day_data['volume']
            second_day_data = df.iloc[day_plus_2_idx]
            high_t2 = second_day_data['high']
            close_t2 = second_day_data['close']
            volume_t2 = second_day_data['volume']
            # 条件A: T+1和T+2是否双双创出40日新高
            prev_40d_high = hist_window['high'].max()
            condition_A_met = (high_t1 > prev_40d_high) and (high_t2 > prev_40d_high)
            # 条件B: 两个高点是否足够接近 (相差<0.5%)
            if max(high_t1, high_t2) > 0:
                proximity = abs(high_t1 - high_t2) / max(high_t1, high_t2)
                condition_B_met = (proximity <= 0.001)
            else:
                condition_B_met = False
            # 条件C: 两天是否都有大于3%的回撤
            fallback_t1_ok = ((high_t1 - close_t1) / high_t1 > 0.03) if high_t1 > 0 else False
            fallback_t2_ok = ((high_t2 - close_t2) / high_t2 > 0.03) if high_t2 > 0 else False
            condition_C_met = fallback_t1_ok and fallback_t2_ok
            # 条件D: 两天成交量之和是否属于巨量,计算历史窗口中，任意连续两天的最大成交量之和
            rolling_vol_sum = hist_window['volume'].rolling(window=2).sum()
            max_rolling_vol = rolling_vol_sum.max()
            condition_D_met = False
            if pd.notna(max_rolling_vol):
                current_vol_sum = volume_t1 + volume_t2
                condition_D_met = (current_vol_sum > max_rolling_vol * 0.95)
            if condition_A_met and condition_B_met and condition_C_met and condition_D_met:
                print(f"[{code}] 在 {df.index[limit_up_day_idx].date()} 后触发T+1,T+2双顶巨量回落形态，排除。")
                return False

        if day_plus_3_idx < len(df):
            # 买前条件9：如果涨停后第一天最高价大于第三天大于第二天，并且都有长上影线，则排除
            data_t1 = df.iloc[day_plus_1_idx]
            data_t2 = df.iloc[day_plus_2_idx]
            data_t3 = df.iloc[day_plus_3_idx]
            is_t1_peak = data_t1['high'] > data_t3['high'] > data_t2['high']
            s_t1 = (data_t1['high'] - data_t1['close']) / data_t1['close'] > 0.04 if data_t1['close'] > 0 else False
            s_t2 = (data_t2['high'] - data_t2['close']) / data_t2['close'] > 0.03 if data_t2['close'] > 0 else False
            s_t3 = (data_t3['high'] - data_t3['close']) / data_t3['close'] > 0.03 if data_t3['close'] > 0 else False
            both_have_long_shadows = s_t2 and s_t3 and s_t1
            if is_t1_peak and both_have_long_shadows:
                return False

            # 买前条件10：如果涨停后第二天和第三天均出现高开大于2%的情况，排除
            day_plus_1_close = df.iloc[day_plus_1_idx]['close']
            day_plus_2_close = df.iloc[day_plus_2_idx]['close']
            day_plus_2_open = df.iloc[day_plus_2_idx]['open']
            day_plus_3_open = df.iloc[day_plus_3_idx]['open']

            if day_plus_2_open > (day_plus_1_close * 1.02) and \
                    day_plus_3_open > (day_plus_2_close * 1.02):
                return False

    return True


def find_first_limit_up(symbol, df, config: StrategyConfig):
    """识别首板涨停日并排除连板"""
    limit_rate = config.MARKET_LIMIT_RATES[get_market_type(symbol)]

    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    df['is_limit'] = df['close'] >= df['limit_price']
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)
    df['is_limit_down'] = df['close'] <= df['down_limit_price']
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    valid_days = []
    for day in limit_days:
        # 日期过滤条件（方便回测）
        if day < pd.Timestamp('2024-03-01') and not config.USE_2019_DATA:
            continue

        if is_valid_first_limit_up_day(df, day, symbol, config, query_tool):
            valid_days.append(day)

    return valid_days


def modify_last_days_and_calc_ma5(df, predict_ratio=1.04):
    """模拟预测MA5的核心方法（优化版，使用.loc替代concat）"""
    if df.empty or len(df) < 2:
        raise ValueError("数据不足，至少需要2个交易日数据")
    modified_df = df.copy()
    last_row = modified_df.iloc[-1]
    new_index_label = last_row.name + pd.Timedelta(days=1)
    new_data = last_row.copy()
    new_data['close'] = last_row['close'] * predict_ratio
    modified_df.loc[new_index_label] = new_data
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5, min_periods=1
    ).mean().round(2)
    return modified_df


def calculate_actual_fill_price(open_price, order_price):
    """计算实际成交价：如果开盘价低于挂单价，以开盘价成交"""
    return min(open_price, order_price) if order_price is not None else None


def check_double_top_neckline_resistance(df, check_day_idx, config: StrategyConfig):
    """检查指定日期是否受到前期双头形态颈线位的压制。"""
    # 1. 定义检测周期
    lookback_days = config.DOUBLE_TOP_CHECK_DAYS
    if check_day_idx < lookback_days:
        return False, None  # 数据不足
    # 2. 获取检测区间的数据.切片到 check_day_idx + 1 是为了包含 check_day_idx 当天。
    hist_data = df.iloc[check_day_idx - lookback_days: check_day_idx + 1]
    # 3. 使用find_peaks寻找显著高点
    # prominence(振幅)参数是关键，用于过滤小波动，值越大，找到的波峰越显著
    peaks, properties = find_peaks(hist_data['high'], prominence=hist_data['high'].std() * 0.95)
    if len(peaks) < 2:
        return False, None  # 未找到足够的显著高点
    # 4. 筛选出最近的两个高点作为潜在的双头
    peak_indices_in_hist = hist_data.index[peaks[-2:]]
    peak_prices = hist_data.loc[peak_indices_in_hist, 'high']
    # 5. 验证双头形态的条件
    p1_date, p2_date = peak_indices_in_hist[0], peak_indices_in_hist[1]
    p1_price, p2_price = peak_prices.iloc[0], peak_prices.iloc[1]
    # 条件a: 两个高点价格接近
    if abs(p1_price - p2_price) / max(p1_price, p2_price) > config.DOUBLE_TOP_PRICE_TOLERANCE:
        return False, None
    # 6. 找到两个高点之间的最低点，作为颈线
    trough_start_idx = df.index.get_loc(p1_date)
    trough_end_idx = df.index.get_loc(p2_date)
    trough_low_price = df.iloc[trough_start_idx:trough_end_idx]['low'].min()
    trough_close_price = df.iloc[trough_start_idx:trough_end_idx]['close'].min()
    neckline_low_price = trough_low_price
    neckline_close_price = trough_close_price
    # 8. 判断'check_day'是否触及颈线位并回落
    check_day_data = df.iloc[check_day_idx]
    check_day_high = check_day_data['high']
    check_day_close = check_day_data['close']
    touch_close = check_day_high >= neckline_close_price and check_day_close < neckline_close_price
    touch_low = check_day_high >= neckline_low_price and check_day_close < neckline_low_price
    if touch_close and not (touch_close and touch_low):
        # 核心判断：当日最高价超过颈线，但收盘价低于颈线，视为受阻回落
        return True, neckline_close_price

    return False, None


def generate_signals(df, first_limit_day, stock_code, stock_name, config: StrategyConfig):
    """生成买卖信号"""
    signals = []
    first_limit_timestamp = pd.Timestamp(first_limit_day)
    limit_rate = config.MARKET_LIMIT_RATES[get_market_type(stock_code)]
    start_idx = df.index.get_loc(first_limit_day)

    # --- 数据准备 ---
    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma10'] = df['close'].rolling(10, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma30'] = df['close'].rolling(30, min_periods=1).mean()
    df['ma55'] = df['close'].rolling(55, min_periods=1).mean()

    next_day_2_pct = None
    if start_idx + 2 < len(df):
        next_day_1 = df.index[start_idx + 1]
        next_day_2 = df.index[start_idx + 2]
        next_day_1_close = df.loc[next_day_1, 'close']
        next_day_2_close = df.loc[next_day_2, 'low']
        if abs(next_day_1_close) > 1e-5:
            next_day_2_pct = (next_day_2_close - next_day_1_close) / next_day_1_close * 100

    start_idx = df.index.get_loc(first_limit_day)
    if (start_idx + 1) >= len(df):
        return signals

    def _create_signal_dict(buy_data, sell_day, sell_price, sell_reason_str, hold_days_val, actual_price):
        return {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '首板日': first_limit_day.strftime('%Y-%m-%d'),
            '买入日': buy_data.name.strftime('%Y-%m-%d'),
            '卖出日': sell_day.strftime('%Y-%m-%d'),
            '涨停后天数': (buy_data.name - first_limit_timestamp).days,
            '持有天数': hold_days_val,
            '买入价': round(actual_price, 2),
            '卖出价': round(sell_price, 2),
            '触碰类型': 'MA5支撑反弹' if buy_data['close'] > buy_data['ma5'] else 'MA5破位回升',
            '收益率(%)': round((sell_price - actual_price) / actual_price * 100, 2),
            '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
            '卖出原因': sell_reason_str,
        }

    # for offset in range(2,5):  # 检查涨停后第2、3、4、5天
    for offset in [2, 4]:  # 检查涨停后第2、3、4、5天
        if start_idx + offset >= len(df):
            break

        current_day_idx = start_idx + offset
        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        if not is_valid_buy_opportunity(df, start_idx, offset, code, config):
            continue

        # 计算挂单价格
        price_ma5 = simulate_ma5_order_prices(df, current_day, config)
        if price_ma5 is None:
            continue

        # 获取当日价格数据
        day_open = current_data['open']
        day_low = current_data['low']
        day_high = current_data['high']

        # 检查挂单价是否在当日价格范围内
        order_valid = day_low <= price_ma5
        if not order_valid:
            continue
        if price_ma5 >= day_open:
            order_valid = True
        actual_price = calculate_actual_fill_price(day_open, price_ma5) if order_valid else None

        # 持有期卖出逻辑
        hold_days = 0
        for sell_offset in range(1, 20):
            sell_day_idx = start_idx + offset + sell_offset

            if sell_day_idx < len(df):
                sell_day = df.index[start_idx + offset + sell_offset]
                current_sell_data = df.loc[sell_day]
                prev_sell_data = df.iloc[sell_day_idx - 1]
                hold_days += 1

                position_info = {'hold_days': hold_days}
                market_data = MarketDataContext(
                    high=current_sell_data['high'],
                    close=current_sell_data['close'],
                    ma5=current_sell_data['ma5'],
                    up_limit_price=current_sell_data['limit_price'],
                    down_limit_price=current_sell_data['down_limit_price'],
                    prev_close=prev_sell_data['close'],
                    prev_up_limit_price=prev_sell_data['limit_price'],
                    prev_down_limit_price=prev_sell_data['down_limit_price']
                )
                should_sell, reason = get_sell_decision(position_info, market_data)
                if should_sell:
                    sell_price = current_sell_data['close']
                    signal = _create_signal_dict(current_data, sell_day, sell_price, reason, position_info['hold_days'],
                                                 actual_price)
                    signals.append(signal)
                    break
            else:
                last_day_idx = len(df) - 1
                sell_day = df.index[last_day_idx]
                sell_data = df.loc[sell_day]
                sell_price = df.iloc[last_day_idx]['close']
                sell_reason = '持有中'
                signal = _create_signal_dict(current_data, sell_day, sell_price, sell_reason, hold_days,
                                             actual_price)
                signals.append(signal)
                break

        if signals:  # 只要已经生成了信号，就终止，不然会重复统计一个涨停信号的不同时间段买入
            break
    return signals


def create_daily_holdings(result_df):
    # 转换日期格式
    result_df['买入日'] = pd.to_datetime(result_df['买入日'])
    result_df['卖出日'] = pd.to_datetime(result_df['卖出日'])

    # 生成日期范围
    min_date = result_df['买入日'].min()
    max_date = result_df['卖出日'].max()
    date_range = pd.date_range(min_date, max_date)

    # 创建持仓字典
    holdings = {}
    for date in date_range:
        # 筛选当日持仓的股票
        held_stocks = result_df[
            (result_df['买入日'] <= date) &
            (result_df['卖出日'] >= date)
            ]['股票代码'].tolist()

        # 格式化为字符串
        holdings[date] = {
            '日期': date.strftime('%Y-%m-%d'),
            '持有股票': ','.join(held_stocks),
            '数量': len(held_stocks)
        }

    return pd.DataFrame(holdings.values())


def save_trades_excel(result_df):
    column_order = ['股票代码', '股票名称', '首板日', '买入日', '卖出日', '涨停后天数',
                    '持有天数', '买入价', '卖出价', '涨停后第二日涨幅(%)', '收益率(%)', '卖出原因']
    # 按买入日降序排序
    result_df = result_df.sort_values(by='买入日', ascending=False)
    result_df = result_df[column_order]
    """专业级Excel导出函数"""
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_name = f"首板交易记录_{timestamp}.xlsx"

    # 创建带格式的Excel写入器
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # 写入原始数据
        result_df.to_excel(writer, sheet_name='交易明细', index=False)

        # 获取工作表对象
        workbook = writer.book
        worksheet = writer.sheets['交易明细']

        # 1. 列宽自适应
        for idx, col in enumerate(result_df.columns):
            max_len = max(result_df[col].astype(str).map(len).max(), len(col)) + 2
            if col == '股票名称':
                max_len += 3
            worksheet.set_column(idx, idx, max_len)

        # 2. 条件格式(收益率红涨绿跌)
        format_green = workbook.add_format({'font_color': '#00B050', 'num_format': '0.00%'})
        format_red = workbook.add_format({'font_color': '#FF0000', 'num_format': '0.00%'})

        # 将收益率转换为小数格式
        for row in range(1, len(result_df) + 1):
            cell_value = result_df.iloc[row - 1]['收益率(%)'] / 100
            if cell_value >= 0:
                worksheet.write(row, result_df.columns.get_loc('收益率(%)'), cell_value, format_green)
            else:
                worksheet.write(row, result_df.columns.get_loc('收益率(%)'), cell_value, format_red)

        # 3. 冻结首行
        worksheet.freeze_panes(1, 0)
        # 4. 自动筛选
        worksheet.autofilter(0, 0, len(result_df), len(result_df.columns) - 1)

        # 2. 添加单支股票表现统计表
        # 确保有交易数据
        if not result_df.empty:
            # 创建分组统计
            stock_stats = result_df.groupby(['股票代码', '股票名称']).agg(
                交易次数=('收益率(%)', 'size'),
                胜率=('收益率(%)', lambda x: (x > 0).sum() / len(x)),
                平均收益=('收益率(%)', lambda x: x[x > 0].mean() if any(x > 0) else 0),
                平均亏损=('收益率(%)', lambda x: x[x <= 0].mean() if any(x <= 0) else 0),
                总收益=('收益率(%)', lambda x: x.sum()),
                总盈利=('收益率(%)', lambda x: x[x > 0].sum()),
                总亏损=('收益率(%)', lambda x: x[x <= 0].sum()),
                胜利次数=('收益率(%)', lambda x: (x > 0).sum()),
                失败次数=('收益率(%)', lambda x: (x <= 0).sum())
            ).reset_index()

            # 格式化胜率为百分比
            stock_stats['胜率'] = stock_stats['胜率'].apply(lambda x: f"{x * 100:.2f}%")

            # 添加列：盈亏比
            stock_stats['盈亏比'] = stock_stats.apply(
                lambda row: abs(row['平均收益'] / row['平均亏损']) if row['平均亏损'] != 0 else np.inf,
                axis=1
            )

            # 对盈亏比进行格式化
            def format_profit_ratio(value):
                if value == np.inf:
                    return "无穷大"
                elif value > 100:
                    return f"{value:.0f}:1"
                else:
                    return f"{value:.2f}:1"

            stock_stats['盈亏比'] = stock_stats['盈亏比'].apply(format_profit_ratio)

            # 重命名列名
            stock_stats = stock_stats.rename(columns={
                '平均收益': '平均收益(%)',
                '平均亏损': '平均亏损(%)',
                '总收益': '总收益(%)',
                '总盈利': '总盈利(%)',
                '总亏损': '总亏损(%)'
            })

            # 重新排列列顺序
            stock_stats = stock_stats[['股票代码', '股票名称', '交易次数', '胜利次数', '失败次数',
                                       '胜率', '平均收益(%)', '平均亏损(%)', '盈亏比',
                                       '总收益(%)', '总盈利(%)', '总亏损(%)']]

            # 写入新工作表
            stock_stats.to_excel(writer, sheet_name='个股表现统计', index=False)

            # 获取股票统计表工作表
            stock_worksheet = writer.sheets['个股表现统计']

            # 设置股票统计表的格式
            # 列宽自适应
            for idx, col in enumerate(stock_stats.columns):
                max_len = max(stock_stats[col].astype(str).map(len).max(), len(col)) + 2
                stock_worksheet.set_column(idx, idx, max_len)

            # 添加条件格式（正收益绿色，负收益红色）
            # 获取需要格式化的列索引
            num_cols = [stock_stats.columns.get_loc(col) for col in
                        ['平均收益(%)', '平均亏损(%)', '总收益(%)', '总盈利(%)', '总亏损(%)']]

            for col_idx in num_cols:
                # 为总亏损应用红色格式（因为它本身是负值）
                if stock_stats.columns[col_idx] == '总亏损(%)':
                    stock_worksheet.conditional_format(
                        1, col_idx, len(stock_stats), col_idx,
                        {
                            'type': 'data_bar',
                            'bar_color': '#FF0000',
                            'bar_border_color': '#FF0000'
                        }
                    )
                else:
                    stock_worksheet.conditional_format(
                        1, col_idx, len(stock_stats), col_idx,
                        {
                            'type': 'data_bar',
                            'bar_color': '#00B050',
                            'bar_border_color': '#00B050'
                        }
                    )

            # 冻结首行
            stock_worksheet.freeze_panes(1, 0)
            # 自动筛选
            stock_worksheet.autofilter(0, 0, len(stock_stats), len(stock_stats.columns) - 1)

        holdings_df = create_daily_holdings(result_df)
        holdings_df.to_excel(writer, sheet_name='每日持仓情况', index=False)
        holdings_worksheet = writer.sheets['每日持仓情况']

        # 设置持仓表格式
        for idx, col in enumerate(holdings_df.columns):
            max_len = max(holdings_df[col].astype(str).map(len).max(), len(col)) + 2
            holdings_worksheet.set_column(idx, idx, max_len)

        holdings_worksheet.freeze_panes(1, 0)
        holdings_worksheet.autofilter(0, 0, len(holdings_df), len(holdings_df.columns) - 1)

        # 5. 添加统计页
        stats_df = pd.DataFrame({
            '统计指标': ['总交易次数', '胜率', '平均盈利', '平均亏损', '盈亏比'],
            '数值': [
                len(result_df),
                f"{result_df['收益率(%)'].gt(0).mean() * 100:.1f}%",
                f"{result_df[result_df['收益率(%)'] > 0]['收益率(%)'].mean():.1f}%",
                f"{result_df[result_df['收益率(%)'] <= 0]['收益率(%)'].mean():.1f}%",
                f"{result_df['收益率(%)'].gt(0).sum() / result_df['收益率(%)'].le(0).sum():.1f}:1"
            ]
        })
        stats_df.to_excel(writer, sheet_name='策略统计', index=False)
    print(f"\033[32m交易记录已保存至 {excel_name}\033[0m")


if __name__ == '__main__':
    total_start = time.perf_counter()  # 记录程序开始时间

    config = StrategyConfig()
    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values

    all_signals = []
    stock_process_start = time.perf_counter()

    for idx, (code, name) in enumerate(stock_list, 1):
        df, _ = get_stock_data(code, config)
        if df.empty:
            continue

        first_limit_days = find_first_limit_up(code, df, config)
        for day in first_limit_days:
            signals = generate_signals(df, day, code, name, config)
            all_signals.extend(signals)

    stock_process_duration = time.perf_counter() - stock_process_start

    # 生成统计报表
    result_df = pd.DataFrame(all_signals)
    if not result_df.empty:
        # 把原有的买入日从年月日类型转为年月日时分秒，确保买入日为日期类型
        result_df['买入日'] = pd.to_datetime(result_df['买入日'])
        # 获取最近三个月的截止日期（含当月）
        current_date = datetime.now()
        three_months_ago = current_date - pd.DateOffset(months=17)

        # 过滤最近三个月的交易记录
        recent_trades = result_df[result_df['买入日'] >= three_months_ago]

        # 按月分组计算收益总和
        monthly_returns = recent_trades.resample('ME', on='买入日')['收益率(%)'].sum()
        monthly_returns = monthly_returns.round(2)  # 保留两位小数
        monthly_summary = [f"{month.month}月 {ret}%" for month, ret in monthly_returns.items()]
        monthly_str = "，".join(monthly_summary)  # 中文逗号分隔

        win_rate = len(result_df[result_df['收益率(%)'] > 0]) / len(result_df) * 100
        avg_win = result_df[result_df['收益率(%)'] > 0]['收益率(%)'].mean()
        avg_loss = abs(result_df[result_df['收益率(%)'] <= 0]['收益率(%)'].mean())
        profit_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        avg_hold_days = result_df['持有天数'].mean()

        get_money = len(result_df[result_df['收益率(%)'] > 0]) / len(result_df) * avg_win - (
                1 - len(result_df[result_df['收益率(%)'] > 0]) / len(result_df)) * avg_loss

        print(f"\n\033[1m=== 策略表现汇总 ===\033[0m")
        print(
            f"总交易数: {len(result_df)}，胜率: {win_rate:.2f}%，均盈: {avg_win:.2f}% | 均亏: {avg_loss:.2f}% | 均持: {avg_hold_days:.2f}，盈亏比: {profit_ratio:.2f}:1，期望: {get_money:.3f}")
        print(f"近三月收益：{monthly_str}")

        # 示例输出
        # print("\n\033[1m最近5笔交易记录:\033[0m")
        # print(result_df.tail(5).to_string(index=False))
    else:
        print("未产生有效交易信号")

    if not result_df.empty:
        result_df['买入日'] = result_df['买入日'].dt.strftime('%Y-%m-%d')
        save_start = time.perf_counter()  # 记录Excel保存开始时间
        save_trades_excel(result_df)
        save_duration = time.perf_counter() - save_start
        print(f"Excel保存耗时: {save_duration:.4f}秒")

    # 程序总耗时统计
    total_duration = time.perf_counter() - total_start
    print(f"\n\033[1m=== 性能统计 ===\033[0m")
    print(f"总运行时间: {total_duration:.2f}秒")
    print(f"股票数据处理时间: {stock_process_duration:.2f}秒")
    print(f"Excel保存时间: {save_duration:.4f}秒")
    print(f"平均每支股票处理时间: {stock_process_duration / len(stock_list) * 1000:.2f}毫秒")
