import concurrent.futures
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import getAllStockCsv
from common_sell_logic import get_sell_decision, MarketDataContext
from scipy.signal import find_peaks
from scipy import stats

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


def simulate_ma5_order_prices(df, current_day, predict_ratio, lookback_days=5):
    """模拟预测买入日MA5值，然后计算挂单价格"""
    current_idx = df.index.get_loc(current_day)
    if current_idx < lookback_days:
        return None
    prev_data = df.iloc[current_idx - lookback_days: current_idx]
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


def prepare_data(df: pd.DataFrame, symbol: str, config: StrategyConfig) -> pd.DataFrame:
    """
    一次性计算所有需要的指标，避免在循环中重复计算。
    """
    if df.empty:
        return df

    # 1. 计算所有均线
    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma10'] = df['close'].rolling(10, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma30'] = df['close'].rolling(30, min_periods=1).mean()
    df['ma55'] = df['close'].rolling(55, min_periods=55).mean()
    df['ma60'] = df['close'].rolling(60, min_periods=60).mean()
    df['ma120'] = df['close'].rolling(120, min_periods=120).mean()

    df['boll_mid'] = df['ma20']
    df['boll_std'] = df['close'].rolling(20, min_periods=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['prev_close_for_tr'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(row['high'] - row['low'],
                                        abs(row['high'] - row['prev_close_for_tr']),
                                        abs(row['low'] - row['prev_close_for_tr'])), axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    # 计算归一化ATR，使其在不同价格的股票间可比
    df['natr'] = (df['atr'] / df['close']) * 100
    df.drop(columns=['prev_close_for_tr', 'tr'], inplace=True)

    df['rsi_slope'] = df['rsi'].diff(2)

    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv

    vwma_12 = (df['close'] * df['volume']).rolling(12).sum() / df['volume'].rolling(12).sum()
    vwma_26 = (df['close'] * df['volume']).rolling(26).sum() / df['volume'].rolling(26).sum()

    range_val = df['high'] - df['low']
    # 避免除以零
    clv = np.where(range_val > 0, ((df['close'] - df['low']) - (df['high'] - df['close'])) / range_val, 0)

    # Volume-Weighted CLV，并计算其10日移动平均值以观察趋势
    df['vwcl'] = clv * df['volume']
    df['vwcl_ma10'] = df['vwcl'].rolling(10).mean()

    # Volume-Weighted MACD
    df['v_macd'] = vwma_12 - vwma_26
    df['v_signal'] = df['v_macd'].rolling(9).mean()

    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)  # 处理 high == low 的情况
    # Money Flow Volume
    mfv = mfm * df['volume']
    # 20-day CMF
    df['cmf_20d'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()

    # 计算5日VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tpv = typical_price * df['volume']
    df['vwap_5d'] = tpv.rolling(5).sum() / df['volume'].rolling(5).sum()

    # 2. 计算涨跌停价和是否触及
    limit_rate = config.MARKET_LIMIT_RATES[get_market_type(symbol)]
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    df['is_limit'] = df['close'] >= df['limit_price']
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)
    df['is_limit_down'] = df['close'] <= df['down_limit_price']

    return df


def is_valid_first_limit_up_day(df: pd.DataFrame, day: pd.Timestamp, code: str, config: StrategyConfig,
                                market_value: float, theme: str, name: str) -> bool:
    """
    检查给定的某一天是否是符合所有条件的首板涨停日。默认获得涨停后一日的数据
    :return: 如果通过所有检查, 返回True; 如果任何一个检查失败, 返回False。
    """
    # 没有涨停后一天数据的直接排除
    if df.index.get_loc(day) + 1 >= len(df):
        return False

    # 涨停前数据少于20的直接排除
    lookback_days = 20
    if df.index.get_loc(day) < lookback_days:
        return False

    limit_up_day_date = day  # 涨停日日期
    limit_up_day_idx = df.index.get_loc(limit_up_day_date)  # 涨停日行号
    limit_up_day_data = df.loc[limit_up_day_date]  # 涨停日数据
    limit_up_day_price = limit_up_day_data['close']  # 涨停日收盘价，重要支撑位
    limit_up_day_open = limit_up_day_data['open']
    limit_up_day_high = limit_up_day_data['high']
    limit_up_day_volume = limit_up_day_data['volume']
    day_minus_1_idx = limit_up_day_idx - 1  # 涨停前一日
    day_minus_1_day_date = df.index[day_minus_1_idx]
    day_minus_1_data = df.iloc[day_minus_1_idx]
    open_m1 = day_minus_1_data['open']
    high_m1 = day_minus_1_data['high']
    close_m1 = day_minus_1_data['close']
    volume_m1 = day_minus_1_data['volume']
    day_plus_1_idx = limit_up_day_idx + 1  # 涨停后一日
    day_plus_1_day_date = df.index[day_plus_1_idx]
    day_plus_1_data = df.iloc[day_plus_1_idx]
    open_p1 = day_plus_1_data['open']
    high_p1 = day_plus_1_data['high']
    close_p1 = day_plus_1_data['close']
    volume_p1 = day_plus_1_data['volume']
    is_red_candle_p1 = close_p1 >= open_p1

    # 条件0：排除市值大于250亿的股票
    if market_value > config.MAX_MARKET_CAP_BILLIONS:
        return False

    # 条件1：排除特定题材
    if "证券" in name or "金融" in name or "证券" in theme or "金融" in theme:  # 牛市旗手，跟不上，不参与
        return False
    if "外贸" in theme:
        return False

    # 条件2：排除前一日涨停和后一日涨停 (连板)
    if day_minus_1_data['is_limit'] or day_plus_1_data['is_limit']:
        return False

    # 条件3：涨停后第一天量能过滤条件。副条件：并且第一天是阴线
    if (volume_p1 >= limit_up_day_volume * 3.6) and not is_red_candle_p1:
        return False

    # 条件4：排除10日内涨停超过4次的股票
    lookback_data = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]
    limit_up_count = (lookback_data['close'] >= lookback_data['limit_price']).sum()
    if limit_up_count >= 4:
        return False

    # 条件5：前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
    pre5_start = df.index[df.index.get_loc(day) - 5]
    pre5_close = df.loc[pre5_start, 'close']
    if pre5_close != 0:
        total_change = (limit_up_day_price - pre5_close) / pre5_close * 100
        if total_change >= 15:
            return False

    # 条件6：排除涨停和涨停后一天的最大量是前120天中最大量四倍以上的股票
    lookback_days_120 = 120
    if day_plus_1_idx >= lookback_days_120:
        lookback_window = df.iloc[limit_up_day_idx - lookback_days_120: limit_up_day_idx]
        if not lookback_window.empty:
            prev_high_volume = df.loc[lookback_window['volume'].idxmax(), 'volume']
            max_volume = max(limit_up_day_volume, volume_p1)
            if max_volume > prev_high_volume * 4:
                # print("排除涨停和涨停后一天的交易量是前120天中最大交易量四倍以上的股票")
                return False

    # 条件12: 排除前期高位连板后，出现的缩量反抽形态，如果回调期最低点高于起涨的最低点，排除说明的确是在反抽，不过是缩量反抽，不需要参与
    lookback_df = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]
    limit_up_days = lookback_df[lookback_df['is_limit']]
    if not limit_up_days.empty:
        limit_up_indices = [df.index.get_loc(d) for d in limit_up_days.index]
        groups = (np.diff(limit_up_indices, prepend=np.nan) != 1).cumsum()
        for group_id in np.unique(groups):
            streak_indices = [idx for i, idx in enumerate(limit_up_indices) if groups[i] == group_id]
            # 子条件1: 必须是大于2天的连板
            if len(streak_indices) > 2:
                streak_df = df.iloc[streak_indices]
                peak_day_timestamp = streak_df['high'].idxmax()
                peak_day_data = df.loc[peak_day_timestamp]
                peak_day_idx = df.index.get_loc(peak_day_timestamp)
                # 子条件3: 涨停日和次日成交量均小于峰值日成交量
                vol_condition = (limit_up_day_volume < peak_day_data['volume'] and
                                 volume_p1 < peak_day_data['volume'])
                # 子条件4: 次日最高价低于峰值日或后一日的最高价
                price_condition = high_p1 < peak_day_data['high']
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

    # 条件9：排除4日内存在跌破一半的涨停
    lookback_data = df.iloc[limit_up_day_idx - 4: limit_up_day_idx]
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

    # 条件13: 排除涨停前10日内，先出现至少2连板后又出现至少2连跌停的极端走势
    window = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]
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

    lookback_days_40 = 40
    if limit_up_day_idx >= lookback_days_40:
        pre_40d_window = df.iloc[limit_up_day_idx - lookback_days_40: limit_up_day_idx]
        pre_6d_window = df.iloc[limit_up_day_idx - 6: limit_up_day_idx]

        # 条件14: 排除过去40天里，振幅超过3.5%的天数少于3天的股票。副条件：第一天的量小于涨停日的2.5倍
        valid_prev_close = pre_40d_window['prev_close'].replace(0, np.nan).dropna()
        if not valid_prev_close.empty:
            daily_amplitude = (pre_40d_window['high'].loc[valid_prev_close.index] - pre_40d_window['low'].loc[
                valid_prev_close.index]) / valid_prev_close
            volatile_days_count = (daily_amplitude < 0.035).sum()
            if volatile_days_count > 37 and volume_p1 < limit_up_day_volume * 2.5:
                # print(f"[{code}] 在 {day.date()} 的涨停被排除：前{lookback_days_40}日波动过小({volatile_days_count}天振幅 > 3.5%)。")
                return False

        # 条件15:如果涨停前6天的最高点也是涨停前40天里面的最高点。p1最高价高于最高点并且上影线超过4%，并且收盘价低于最高点。并且最高点的量大于涨停后第一天量能的1.2倍，排除
        if not pre_40d_window.empty and not pre_6d_window.empty:
            peak_40d_price = pre_40d_window['high'].max()
            peak_6d_price = pre_6d_window['high'].max()
            # 条件A: 涨停前6天的最高点也是前40天的最高点
            if abs(peak_6d_price - peak_40d_price) < 1e-5:
                peak_day_loc = pre_40d_window['high'].idxmax()
                peak_day_data = df.loc[peak_day_loc]
                peak_price = peak_day_data['high']
                peak_volume = peak_day_data['volume']
                # 条件C: T+1最高价高于关键高点 (盘中尝试突破)
                cond_c_p1_breaks_peak = high_p1 > peak_price
                # 条件D: T+1日从最高点回撤超过4% (形成长上影线)
                cond_d_p1_long_wick = (high_p1 - close_p1) / high_p1 > 0.04 if high_p1 > 0 else False
                # 条件E: T+1收盘价低于关键高点 (收盘时突破失败)
                cond_e_p1_close_fails = close_p1 < peak_price
                # 条件F: 关键高点日的成交量 > T+1成交量的1.2倍 (缩量假突破)
                cond_f_volume_confirms = peak_volume > (volume_p1 * 1.2)
                # 如果所有条件都满足，则构成一个经典的假突破形态，应予以排除
                if (cond_c_p1_breaks_peak and
                        cond_d_p1_long_wick and
                        cond_e_p1_close_fails and
                        cond_f_volume_confirms):
                    return False

    return True


def is_valid_buy_opportunity(df: pd.DataFrame, limit_up_day_idx: int, offset: int, code: str,
                             config: StrategyConfig) -> bool:
    """
    检查从首板日到潜在买入日之间，是否满足所有的买入前置条件。
    """
    limit_up_day_date = df.index[limit_up_day_idx]
    limit_up_day_data = df.loc[limit_up_day_date]  # 涨停日数据
    limit_up_day_price = limit_up_day_data['close']  # 涨停日收盘价，重要支撑位
    limit_up_day_open = limit_up_day_data['open']
    limit_up_day_high = limit_up_day_data['high']
    limit_up_day_low = limit_up_day_data['low']
    limit_up_day_volume = limit_up_day_data['volume']
    day_minus_1_idx = limit_up_day_idx - 1
    day_minus_1_day_date = df.index[day_minus_1_idx]
    day_minus_1_data = df.iloc[limit_up_day_idx - 1]
    open_m1 = day_minus_1_data['open']
    high_m1 = day_minus_1_data['high']
    close_m1 = day_minus_1_data['close']
    volume_m1 = day_minus_1_data['volume']
    day_minus_2_idx = limit_up_day_idx - 2
    day_minus_2_day_date = df.index[day_minus_2_idx]
    day_minus_2_data = df.iloc[limit_up_day_idx - 2]
    open_m2 = day_minus_2_data['open']
    high_m2 = day_minus_2_data['high']
    close_m2 = day_minus_2_data['close']
    volume_m2 = day_minus_2_data['volume']
    day_minus_3_idx = limit_up_day_idx - 3
    day_minus_3_day_date = df.index[day_minus_3_idx]
    day_minus_3_data = df.iloc[limit_up_day_idx - 3]
    day_plus_1_idx = limit_up_day_idx + 1
    day_plus_1_day_date = df.index[day_plus_1_idx]
    day_plus_1_data = df.iloc[day_plus_1_idx]

    open_p1 = day_plus_1_data['open']
    high_p1 = day_plus_1_data['high']
    close_p1 = day_plus_1_data['close']
    low_p1 = day_plus_1_data['low']
    volume_p1 = day_plus_1_data['volume']
    is_red_candle_p1 = close_p1 >= open_p1
    upper_body_p1 = max(open_p1, close_p1)
    upper_shadow_ratio_p1 = (high_p1 - upper_body_p1) / limit_up_day_price if limit_up_day_price > 0 else 0  # P1上影线比例

    potential_buy_day_idx = limit_up_day_idx + offset

    # 在第四天时，排除已经在第二天买入的股票
    if offset == 4:
        day_2_idx = limit_up_day_idx + 2
        if day_2_idx < len(df):
            day_2_timestamp = df.index[day_2_idx]
            day_2_data = df.iloc[day_2_idx]
            hypothetical_price_on_day_2 = simulate_ma5_order_prices(df, day_2_timestamp,
                                                                    config.PREDICT_PRICE_INCREASE_RATIO)
            if hypothetical_price_on_day_2 is not None:
                if day_2_data['low'] <= hypothetical_price_on_day_2:
                    return False

    # 买前条件1: 排除买入前日收盘价>80的股票
    latest_close = df.iloc[potential_buy_day_idx - 1]['close']
    if latest_close > 80:
        return False

    # 买前条件2: 排除在首板日和买入日之间，有新的涨停的和收盘价低于首板收盘价
    for i in range(1, offset):
        check_day = df.index[limit_up_day_idx + i]
        if df.loc[check_day, 'is_limit'] or df.loc[check_day, 'close'] < limit_up_day_price:
            return False

    # 买前条件3：排除一字板，且排除最近20天内在20日线下超过一次的票，筛选出那些在涨停前趋势保持良好、没有经历深度或反复调整的股票。副条件：排除涨停前第二天低于五日线的票
    lookback_window = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
    breakdown_count = (lookback_window['close'] < lookback_window['ma20']).sum()
    is_m2_day_below_ma5 = close_m2 < day_minus_2_data['ma5']
    if limit_up_day_low == limit_up_day_price and breakdown_count > 1 and is_m2_day_below_ma5:
        print("排除首板是一字板的")
        return False

    if offset > 3:
        day_plus_2_idx = limit_up_day_idx + 2
        day_plus_2_day_date = df.index[day_plus_2_idx]
        day_plus_2_data = df.loc[day_plus_2_day_date]
        open_p2 = day_plus_2_data['open']
        high_p2 = day_plus_2_data['high']
        close_p2 = day_plus_2_data['close']
        low_p2 = day_plus_2_data['low']
        volume_p2 = day_plus_2_data['volume']
        is_red_candle_p2 = close_p2 >= open_p2  # P2 (T+2) 是否为阳线
        upper_body_p2 = max(open_p2, close_p2)
        upper_shadow_ratio_p2 = (high_p2 - upper_body_p2) / close_p1 if close_p1 > 0 else 0  # P2上影线比例

        day_plus_3_idx = limit_up_day_idx + 3
        day_plus_3_day_date = df.index[day_plus_3_idx]
        day_plus_3_data = df.loc[day_plus_3_day_date]
        open_p3 = day_plus_3_data['open']
        high_p3 = day_plus_3_data['high']
        close_p3 = day_plus_3_data['close']
        low_p3 = day_plus_3_data['low']
        volume_p3 = day_plus_3_data['volume']
        is_red_candle_p3 = close_p3 >= open_p3
        upper_body_p3 = max(open_p3, close_p3)
        upper_shadow_ratio_p3 = (high_p2 - upper_body_p3) / close_p2 if close_p2 > 0 else 0

        # 买前条件6：排除p1是放量阳线+p2低开未收复前日实体中点
        volume_condition = (volume_p1 > limit_up_day_volume * 1.5)
        price_range = high_p1 - low_p1
        if abs(price_range) < 1e-5:
            candle_condition = False
        else:
            body_ratio = (day_plus_1_data['close'] - day_plus_1_data['open']) / price_range
            candle_condition = (body_ratio > 0.5) and (day_plus_1_data['close'] > day_plus_1_data['open'])
        midpoint = (close_p1 + open_p1) / 2
        low_open_recover_condition = (open_p2 < close_p1) and (close_p2 < midpoint)
        if volume_condition and candle_condition and low_open_recover_condition:
            return False

        lookback_days_40 = 40
        lookback_days_5 = 5
        if limit_up_day_idx >= lookback_days_40:
            # 买前条件7: 排除涨停后三天的最高价低于40天内最高价且距离小于1%。副条件：p3冲高回落超过4%
            hist_window = df.iloc[limit_up_day_idx - lookback_days_40: limit_up_day_idx - lookback_days_5]
            prev_40d_high = hist_window['high'].max()
            max_high = max(high_p1, high_p2, high_p3)
            if max_high < prev_40d_high and (prev_40d_high - max_high) / prev_40d_high < 0.01:
                if upper_shadow_ratio_p3 > 0.04:
                    return False

            # 买前条件9： 排除p1,p2连续创40日新高但巨量回落的“双顶出货”形态
            pre_40d_window = df.iloc[limit_up_day_idx - lookback_days_40: limit_up_day_idx]
            prev_40d_high = pre_40d_window['high'].max()
            condition_A_met = (high_p1 > prev_40d_high) and (high_p2 > prev_40d_high)
            # 条件B: 两个高点是否足够接近 (相差<0.5%)
            if max(high_p1, high_p2) > 0:
                proximity = abs(high_p1 - high_p2) / max(high_p1, high_p2)
                condition_B_met = (proximity <= 0.001)
            else:
                condition_B_met = False
            # 条件C: 两天是否都有大于3%的回撤
            condition_C_met = upper_shadow_ratio_p1 > 0.03 and upper_shadow_ratio_p2 > 0.03
            # 条件D: 两天成交量之和是否属于巨量,计算历史窗口中，任意连续两天的最大成交量之和
            rolling_vol_sum = pre_40d_window['volume'].rolling(window=2).sum()
            max_rolling_vol = rolling_vol_sum.max()
            condition_D_met = False
            if pd.notna(max_rolling_vol):
                current_vol_sum = volume_p1 + volume_p2
                condition_D_met = (current_vol_sum > max_rolling_vol * 0.95)
            if condition_A_met and condition_B_met and condition_C_met and condition_D_met:
                print(f"[{code}] 在 {df.index[limit_up_day_idx].date()} 后触发T+1,T+2双顶巨量回落形态，排除。")
                return False

        # 买前条件8：排除p1,p2,p3受到前期双头颈线压制
        for i in range(1, offset):
            check_day_idx = limit_up_day_idx + i
            if check_day_idx >= len(df):
                continue
            is_resisted, neckline = check_double_top_neckline_resistance(df, check_day_idx, config)
            if is_resisted:
                check_day_date = df.index[check_day_idx].date()
                print(f"[{code}] 在涨停后第{i}天({check_day_date})触及双头颈线 {neckline:.2f} 回落，放弃买入机会。")
                return False

        # 买前条件9：如果涨停后第一天最高价大于第三天大于第二天，并且都有长上影线，则排除
        is_t1_peak = day_plus_1_data['high'] > day_plus_3_data['high'] > day_plus_2_data['high']
        s_t1 = upper_shadow_ratio_p1 > 0.04
        s_t2 = upper_shadow_ratio_p2 > 0.03
        s_t3 = upper_shadow_ratio_p3 > 0.03
        both_have_long_shadows = s_t2 and s_t3 and s_t1
        if is_t1_peak and both_have_long_shadows:
            return False

        # 买前条件10：如果p2和p3均出现高开大于2%的情况，排除
        if open_p2 > (close_p1 * 1.02) and open_p3 > (close_p2 * 1.02):
            return False
        if open_p2 > (close_p1 * 1.03):
            return False

        # 买前条件11：如果涨停后p2和p3都收阴线，并且p1大幅回落小于2.5%大于1%就排除
        if not is_red_candle_p2 and not is_red_candle_p3 and 0.025 > upper_shadow_ratio_p1 > 0.01:
            return False

        # 买前条件12：涨停后第一天被各条均线压制，并且第二天收盘价被均线压制，并且前三日不被30日线支撑,第二天是阳线的不排除
        support_mas_to_check = ['ma30']
        days_to_check_indices = [day_plus_1_idx, day_plus_2_idx, day_plus_3_idx]

        long_term_mas = ['ma20', 'ma30', 'ma55', 'ma60']
        # long_term_mas = ['ma120']
        for ma_col in long_term_mas:
            ma_value = day_plus_1_data[ma_col]
            is_persistently_supported = True  # 先假设被该均线持续支撑
            for day_idx in days_to_check_indices:
                day_data = df.iloc[day_idx]
                day_close = day_data['close']
                day_open = day_data['open']
                day_ma_value = day_data['ma30']
                if not (pd.notna(day_ma_value) and day_ma_value > 0):
                    is_persistently_supported = False
                is_above_ma = day_open >= day_ma_value
                is_close_to_ma = (abs(day_close - day_ma_value) / day_ma_value) <= 0.02
                if not (is_above_ma and is_close_to_ma):
                    is_persistently_supported = False

            # 确保均线值有效
            if pd.notna(ma_value) and ma_value > 0:
                # 条件1: T+1的最高价 > 均线值 (触碰或上穿)
                tested_ma = high_p1 > ma_value
                # 条件2: T+1的收盘价与均线值差距在千分之五以内
                is_rejected = (abs(close_p1 - ma_value) / ma_value) <= 0.02
                if tested_ma and is_rejected and close_p2 < ma_value and not is_persistently_supported and (
                        close_p2 < open_p2):
                    # print(f"[{code}] 在 {day_plus_1_data.name.date()} 触及 {ma_col} ({ma_value:.2f}) 后被压制，排除。")
                    return False

        # 买前条件13：挑选出涨停后第一天跳空高开，尾盘不回落的票，排除第二天和第三天不跌第一天最低点的票，排除涨停前一天粘合的票
        strong_support_level = limit_up_day_price * 1.03
        t1_low = day_plus_1_data['low']
        # 条件A: T+1最低价 > 涨停价的1.05倍 (跳空高开且不回补)
        is_strong_gap_and_hold = t1_low >= strong_support_level
        # 条件B: T+2收盘价守住T+1的最低点
        t2_holds_support = day_plus_2_data['low'] >= t1_low
        # 条件C: T+3收盘价也守住T+1的最低点
        t3_holds_support = day_plus_3_data['low'] >= t1_low
        ma5_m1 = day_minus_1_data['ma5']
        ma10_m1 = day_minus_1_data['ma10']
        ma20_m1 = day_minus_1_data['ma20']
        is_nian_he = False
        if pd.notna([ma5_m1, ma10_m1, ma20_m1]).all():
            ma_list = [ma5_m1, ma10_m1, ma20_m1]
            max_ma = max(ma_list)
            min_ma = min(ma_list)
            avg_ma = sum(ma_list) / 3
            if avg_ma > 0:
                spread_ratio = (max_ma - min_ma) / avg_ma
                if spread_ratio < 0.005:
                    is_nian_he = True
        if is_strong_gap_and_hold and not t2_holds_support and not t3_holds_support and not is_nian_he:
            return False

        # 买前条件10：p1最高价大于55日线，p2和p3收盘价低于55日线。副条件：三天都不被20日线和30日线支撑
        ma55_t0 = limit_up_day_data['ma55']
        ma55_t1 = day_plus_1_data['ma55']
        ma55_t2 = day_plus_2_data['ma55']
        ma55_t3 = day_plus_3_data['ma55']
        # 确保所有需要用到的MA55值都存在且有效
        if pd.notna([ma55_t0, ma55_t1, ma55_t2, ma55_t3]).all():
            # 条件1: T+1日收盘价 < T+1日ma55
            t1_above_ma55 = day_plus_1_data['high'] > ma55_t1
            # 条件2: T+2日收盘价 > T+2日ma55
            t2_above_ma55 = day_plus_2_data['close'] < ma55_t2
            # 条件3: T+3日收盘价 < T+3日ma55
            t3_below_ma55 = day_plus_3_data['close'] < ma55_t3
            is_persistently_30_supported = True
            is_persistently_20_supported = True
            days_to_check_indices = [day_plus_1_idx, day_plus_2_idx, day_plus_3_idx]
            for day_idx in days_to_check_indices:
                day_data = df.iloc[day_idx]
                day_close = day_data['close']
                day_low = day_data['low']
                day_open = day_data['open']
                day_ma_value = day_data['ma30']
                day_20_ma_value = day_data['ma20']
                is_above_ma = day_open >= day_ma_value
                is_close_to_ma = (abs(day_close - day_ma_value) / day_ma_value) <= 0.02
                if not (is_above_ma and is_close_to_ma):
                    is_persistently_30_supported = False
                is_above_20_ma = day_open >= day_20_ma_value
                is_close_20_to_ma = (abs(day_low - day_20_ma_value) / day_20_ma_value) <= 0.02
                if not (is_above_20_ma and is_close_20_to_ma):
                    is_persistently_20_supported = False
            if t1_above_ma55 and t3_below_ma55 and t2_above_ma55 and not is_persistently_30_supported and not is_persistently_20_supported:
                # print(f"[{code}] 触发MA55假突破回落形态，排除。")
                return False

        # 买前条件11：p1阴线，p2红柱，p3十字星并且p3收盘价小于p2实体的一半。副条件：三天都不被十日线支撑，并且五日十日二十日不粘和
        if not is_red_candle_p1:
            midpoint_p1_body = (open_p1 + close_p1) / 2
            if close_p2 <= midpoint_p1_body:
                # print(f"[{code}] 触发弱反弹形态 (T+2未能收复T+1阴线一半)，排除。")
                if is_red_candle_p2:
                    midpoint_p2_body = (open_p2 + close_p2) / 2
                    ma5_m1 = day_minus_1_data['ma5']
                    ma10_m1 = day_minus_1_data['ma10']
                    ma20_m1 = day_minus_1_data['ma30']
                    is_nian_he = False
                    if pd.notna([ma5_m1, ma10_m1, ma20_m1]).all():
                        ma_list = [ma5_m1, ma10_m1, ma20_m1]
                        max_ma = max(ma_list)
                        min_ma = min(ma_list)
                        avg_ma = sum(ma_list) / 3
                        if avg_ma > 0:
                            spread_ratio = (max_ma - min_ma) / avg_ma
                            if spread_ratio < 0.012:
                                is_nian_he = True
                    days_to_check = [day_plus_1_data, day_plus_2_data, day_plus_3_data]
                    is_persistently_testing_ma10 = True

                    for day_data in days_to_check:
                        day_low = day_data['low']
                        day_close = day_data['close']
                        ma10_value = day_data['ma10']

                        if not (pd.notna(ma10_value) and ma10_value > 0):
                            is_persistently_testing_ma10 = False
                            break

                        # 条件1: 最低价触碰或跌破MA10
                        touched_ma10 = day_low <= ma10_value
                        # 条件2: 收盘价强势反弹至MA10上方1%以上
                        bounced_strongly = (day_close - ma10_value) / ma10_value > 0.01
                        if not (touched_ma10 and bounced_strongly):
                            is_persistently_testing_ma10 = False
                            break
                    is_shi_zi = True
                    if close_p3 > 0:
                        real_body_ratio_p3 = abs(open_p3 - close_p3) / close_p3
                        if real_body_ratio_p3 < 0.004:
                            is_shi_zi = False
                    if close_p3 < midpoint_p2_body and not is_persistently_testing_ma10 and is_shi_zi and not is_nian_he:
                        # print(f"[{code}] 触发上涨乏力形态 (T+3跌破T+2阳线一半)，排除。")
                        return False

        # 买前条件12：涨停后三天的最高值差不多相等，p1p2存在上影线长度占K线总长度的20%以上
        highs = [day_plus_1_data['high'], day_plus_2_data['high'], day_plus_3_data['high']]
        max_h, min_h = max(highs), min(highs)
        if max_h > 0 and (max_h - min_h) / max_h < 0.01:
            all_have_long_wicks = True  # 先假设都满足长上影线条件
            days_data = [day_plus_1_data, day_plus_2_data]
            for day_data in days_data:
                d_high, d_low = day_data['high'], day_data['low']
                d_open, d_close = day_data['open'], day_data['close']
                full_range = d_high - d_low
                if full_range > 1e-5:
                    upper_body = max(d_open, d_close)
                    upper_wick_length = d_high - upper_body
                    # 条件2: 定义长上影线为上影线长度占K线总长度的35%以上
                    if (upper_wick_length / full_range) < 0.2:
                        all_have_long_wicks = False
                        break
                else:
                    all_have_long_wicks = False
                    break
            ma5_m1 = day_minus_1_data['ma5']
            ma10_m1 = day_minus_1_data['ma10']
            ma20_m1 = day_minus_1_data['ma20']
            is_nian_he = False
            if pd.notna([ma5_m1, ma10_m1, ma20_m1]).all():
                ma_list = [ma5_m1, ma10_m1, ma20_m1]
                max_ma = max(ma_list)
                min_ma = min(ma_list)
                avg_ma = sum(ma_list) / 3
                if avg_ma > 0:
                    spread_ratio = (max_ma - min_ma) / avg_ma
                    if spread_ratio < 0.005:
                        is_nian_he = True
            low_support = low_p3 > low_p1 and close_p3 > close_p1
            is_bald_p3 = abs(close_p3 - high_p3) / high_p3 <= 0.01
            # 如果三天最高价相近，并且每天都有长上影线，则排除
            if all_have_long_wicks and not low_support and not is_nian_he and not is_bald_p3:
                return False

        # 买前条件15：如果是第四天买入，排除第三天在预计1.04时可能买入的股票，同时保留0.98时候可能买入的股票。最后的结果再排除被20日线和30日线支撑的股票
        day_3_104_price = simulate_ma5_order_prices(df, day_plus_3_day_date, 1.04)
        day_3_098_price = simulate_ma5_order_prices(df, day_plus_3_day_date, 0.98)
        if day_3_104_price is not None and day_3_098_price is not None:
            if day_3_098_price < low_p3 <= day_3_104_price:
                is_persistently_30_supported = True
                is_persistently_20_supported = True
                days_to_check_indices = [day_plus_1_idx, day_plus_2_idx, day_plus_3_idx]
                for day_idx in days_to_check_indices:
                    day_data = df.iloc[day_idx]
                    day_close = day_data['close']
                    day_low = day_data['low']
                    day_open = day_data['open']
                    day_ma_value = day_data['ma30']
                    day_20_ma_value = day_data['ma20']
                    is_above_ma = day_open >= day_ma_value
                    is_close_to_ma = (abs(day_close - day_ma_value) / day_ma_value) <= 0.02
                    if not (is_above_ma and is_close_to_ma):
                        is_persistently_30_supported = False
                    is_above_20_ma = day_open >= day_20_ma_value
                    is_close_20_to_ma = (abs(day_low - day_20_ma_value) / day_20_ma_value) <= 0.02
                    if not (is_above_20_ma and is_close_20_to_ma):
                        is_persistently_20_supported = False
                if not is_persistently_20_supported and not is_persistently_30_supported:
                    return False


    gap_down_over_5pct = False
    limit_up_close = limit_up_day_data['close']
    if limit_up_close > 0:
        # 计算低开幅度
        if (limit_up_close - open_p1) / limit_up_close > 0.03:
            gap_down_over_5pct = True
    if gap_down_over_5pct and upper_shadow_ratio_p1 > 0.02:
        # print(f"[{code}] 在 {day.date()} 排除：涨停次日低开超5%且有长上影线。") # 如果需要调试，可以取消此行注释
        is_persistently_20_supported = True
        days_to_check_indices = [limit_up_day_idx, day_minus_1_idx, day_minus_2_idx]
        for day_idx in days_to_check_indices:
            day_data = df.iloc[day_idx]
            day_close = day_data['close']
            day_low = day_data['low']
            day_open = day_data['open']
            day_20_ma_value = day_data['ma20']
            is_above_20_ma = day_open >= day_20_ma_value * 0.98
            is_open_20_to_ma = (abs(day_open - day_20_ma_value) / day_20_ma_value) <= 0.02
            is_low_20_to_ma = (abs(day_low - day_20_ma_value) / day_20_ma_value) <= 0.02
            is_close_20_to_ma = is_open_20_to_ma or is_low_20_to_ma
            if not (is_above_20_ma and is_close_20_to_ma):
                is_persistently_20_supported = False
        ma5_m1 = day_minus_1_data['ma5']
        ma10_m1 = day_minus_1_data['ma10']
        ma20_m1 = day_minus_1_data['ma30']
        is_nian_he = False
        if pd.notna([ma5_m1, ma10_m1, ma20_m1]).all():
            ma_list = [ma5_m1, ma10_m1, ma20_m1]
            max_ma = max(ma_list)
            min_ma = min(ma_list)
            avg_ma = sum(ma_list) / 3
            if avg_ma > 0:
                spread_ratio = (max_ma - min_ma) / avg_ma
                if spread_ratio < 0.02:
                    is_nian_he = True

        cond_all_closes_above_ma120 = False
        lookback_days_ma120 = 10
        if limit_up_day_idx > lookback_days_ma120:
            window = df.iloc[limit_up_day_idx - lookback_days_ma120: limit_up_day_idx]
            # 检查1: 确保整个窗口内MA120数据有效，否则跳过此检查
            if not window['ma120'].isnull().any():
                # 条件A: 10天内，每日收盘价都在120日线或其上方
                cond_A_all_closes_above_ma120 = (window['close'] >= window['ma120']).all()
                # 条件B: 10天内，有3天或更多的最低价触及或跌破120日线
                cond_B_lows_test_ma120 = (window['low'] <= window['ma120']).sum() >= 3
                # 条件C: 10天内，有超过6天(即>=7天)的最低价离120日线距离小于2%
                cond_C_lows_hug_ma120 = ((window['low'] - window['ma120']) / window['ma120'] < 0.02).sum() > 6
                # 如果所有条件都满足，则构成“纠缠”形态，予以排除
                if cond_A_all_closes_above_ma120 and cond_B_lows_test_ma120 and cond_C_lows_hug_ma120:
                    # print(f"[{code}] 在 {day.date()} 排除：涨停前10日在MA120附近过度纠缠。") # 调试时可开启
                    cond_all_closes_above_ma120 = True
        is_persistently_30_supported = True
        days_to_check_indices = [day_minus_1_idx, day_minus_2_idx, day_minus_3_idx]
        for day_idx in days_to_check_indices:
            day_data = df.iloc[day_idx]
            day_close = day_data['close']
            day_low = day_data['low']
            day_open = day_data['open']
            day_ma_value = day_data['ma30']
            is_above_ma = day_close >= day_ma_value
            is_open_to_ma = (abs(day_open - day_ma_value) / day_ma_value) <= 0.02
            is_close_to_ma = (abs(day_low - day_ma_value) / day_ma_value) <= 0.02
            all_is_close_to_ma = is_open_to_ma or is_close_to_ma
            if not all_is_close_to_ma:
                is_persistently_30_supported = False
        if not is_persistently_20_supported and not is_nian_he and not cond_all_closes_above_ma120 and not is_persistently_30_supported:
            return False

    if offset == 2:
        lookback_days_60 = 60
        if limit_up_day_idx > lookback_days_60:
            # 定义回看窗口：涨停前的60个交易日
            hist_window = df.iloc[limit_up_day_idx - lookback_days_60: limit_up_day_idx]

            # --- 条件1：近期高点压力 ---
            # 逻辑：如果T+1日的最高价已经非常接近前60日高点，但收盘价却未能突破，说明压力显现，应排除。
            # prev_high = hist_window['high'].max()
            # # T+1最高价摸到或超过前期高点的98%，但收盘价低于前期高点
            # if high_p1 >= prev_high * 0.98 and close_p1 < prev_high:
            #     # print(f"[{code}] T+2买入排除(条件1)：T+1触及近期高点 {prev_high:.2f} 回落。")
            #     return True

            # --- 条件2：前期放量高点压力 ---
            # 逻辑：前期成交量最大的那一天往往是多空分歧最激烈的地方，其最高价是重要心理关口。
            # 如果T+1日尝试突破这个高点但失败，且成交量未能超越当时的量，说明买方力量不足。
            # peak_volume_day_idx = hist_window['volume'].idxmax()
            # peak_volume_day_data = df.loc[peak_volume_day_idx]
            # high_at_peak_volume = peak_volume_day_data['high']
            # volume_at_peak = peak_volume_day_data['volume']
            #
            # # T+1最高价触及放量日的最高价，但收盘价低于它，且T+1的成交量更小
            # if high_p1 >= high_at_peak_volume and close_p1 < high_at_peak_volume and volume_p1 < volume_at_peak:
            #     # print(f"[{code}] T+2买入排除(条件2)：T+1在前期放量高点 {high_at_peak_volume:.2f} 处缩量回落。")
            #     return True
            #
            # # --- 条件3：前期密集峰/平台压力 ---
            # # 逻辑：如果前期存在一个由多个小高点组成的密集平台区，这个区域将构成一个强大的阻力带。
            # # 我们用算法找出这些小高点，如果它们密集分布，就形成一个平台。
            # peaks, _ = find_peaks(hist_window['high'], prominence=hist_window['high'].std() * 0.5, width=2)
            # if len(peaks) >= 3:  # 如果能找到3个以上的显著小高点
            #     peak_highs = hist_window.iloc[peaks]['high']
            #     # 计算这些高点的价格波动范围
            #     price_spread = (peak_highs.max() - peak_highs.min()) / peak_highs.mean()
            #     # 如果这些高点的价格差异在5%以内，我们认为它们构成了一个密集平台
            #     if price_spread < 0.05:
            #         resistance_zone_top = peak_highs.max()
            #         # T+1的最高价进入了这个平台区域，但收盘价未能站稳在平台之上
            #         if high_p1 >= resistance_zone_top * 0.98 and close_p1 < resistance_zone_top:
            #             # print(f"[{code}] T+2买入排除(条件3)：T+1受阻于前期平台 {resistance_zone_top:.2f}。")
            #             return True

            # lookback_days_90 = 90
            # if limit_up_day_idx > lookback_days_90:
            #     hist_window_90d = df.iloc[limit_up_day_idx - lookback_days_90: limit_up_day_idx]

            #     # --- 条件11：涨停“撞线”前期关键高点 ---
            #     prev_major_high = hist_window_90d['high'].max()
            #     if limit_up_day_price < prev_major_high and limit_up_day_price >= prev_major_high * 0.98:
            #         return True

            # if high_p1 > limit_up_day_high and volume_p1 < limit_up_day_volume:
            #     return True
            lookback_days_90 = 90
            if limit_up_day_idx > lookback_days_90:
                hist_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]
                hist_window_90d = df.iloc[limit_up_day_idx - 90: limit_up_day_idx]
                # # --- 条件16：突破K线的“虚假繁荣”压力 ---
                # peak_volume_day_idx = hist_window_60d['volume'].idxmax()
                # high_at_peak_volume = df.loc[peak_volume_day_idx, 'high']
                # avg_price_p1 = (open_p1 + high_p1 + low_p1 + close_p1) / 4
                # if high_p1 > high_at_peak_volume and avg_price_p1 < high_at_peak_volume:
                #     return True

                # --- 条件20：高位巨幅“避雷针”反转压力 ---
                # range_p1 = high_p1 - low_p1
                # if range_p1 > 0:
                #     body_p1_size = abs(open_p1 - close_p1)
                #     upper_wick_p1 = high_p1 - max(open_p1, close_p1)
                #
                #     # 上影线长度是实体的3倍以上，且实体被压缩在总振幅的下三分之一
                #     if body_p1_size > 0 and (upper_wick_p1 / body_p1_size > 3.0) and \
                #             (max(open_p1, close_p1) < low_p1 + range_p1 * 0.33):
                #         return True

                # --- 条件24：涨停前的“高波动/长影线”混乱形态 ---
                # 检查T-1
                # high_m1, low_m1 = day_minus_1_data['high'], day_minus_1_data['low']
                # range_m1 = high_m1 - low_m1
                # is_chaotic_m1 = False
                # if range_m1 > 0 and (range_m1 / open_m1 > 0.05):
                #     if abs(open_m1 - close_m1) / range_m1 < 0.3:
                #         is_chaotic_m1 = True
                # # 检查T-2
                # high_m2, low_m2 = day_minus_2_data['high'], day_minus_2_data['low']
                # range_m2 = high_m2 - low_m2
                # is_chaotic_m2 = False
                # if range_m2 > 0 and (range_m2 / open_m2 > 0.05):
                #     if abs(open_m2 - close_m2) / range_m2 < 0.3:
                #         is_chaotic_m2 = True
                # # 如果连续两天混乱
                # if is_chaotic_m1 and is_chaotic_m2:
                #     return True

                # 重点关注 --- 条件26：中短期均线“趋势冲突” --- 好评
                # if limit_up_day_idx > 65:  # 确保有足够的MA60数据
                #     ma10_p1 = day_plus_1_data['ma10']
                #     ma60_p1 = day_plus_1_data['ma60']
                #
                #     # 获取5天前的均线值用于计算斜率方向
                #     ma10_m4 = df.iloc[limit_up_day_idx - 4]['ma10']
                #     ma60_m4 = df.iloc[limit_up_day_idx - 4]['ma60']
                #
                #     if all(pd.notna([ma10_p1, ma60_p1, ma10_m4, ma60_m4])):
                #         # 短期趋势向上，而中期趋势向下
                #         if ma10_p1 > ma10_m4 and ma60_p1 < ma60_m4:
                #             return True

                # --- 条件44：跌破“心理锚点”后的“垂死挣扎” ---
                # 用find_peaks找波谷（显著低点）
                # troughs, _ = find_peaks(-hist_window_90d['low'], prominence=hist_window_90d['low'].std() * 0.8)
                # if len(troughs) > 0:
                #     # 取最近的一个显著低点作为心理锚点
                #     recent_swing_low_val = hist_window_90d.iloc[troughs[-1]]['low']
                #     # 检查T-5到T-1是否已跌破该锚点
                #     pre_5d_window = df.iloc[limit_up_day_idx - 5:limit_up_day_idx]
                #     if not pre_5d_window.empty and pre_5d_window['low'].min() < recent_swing_low_val:
                #         # 检查T+1是否无力收复
                #         if high_p1 < recent_swing_low_val:
                #             return True

                # --- 条件55：“力不从心”——OBV顶背离 ---
                # if limit_up_day_idx > 60:
                #     hist_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]
                #     peaks, _ = find_peaks(hist_window_60d['high'], prominence=hist_window_60d['high'].std())
                #     if len(peaks) > 0:
                #         # 取最近的前高点
                #         prev_peak_idx_in_window = peaks[-1]
                #         prev_peak_day = hist_window_60d.iloc[prev_peak_idx_in_window]
                #         prev_peak_price = prev_peak_day['high']
                #         obv_at_prev_peak = prev_peak_day['obv']
                #
                #         # 检查T+1是否形成背离
                #         obv_p1 = day_plus_1_data['obv']
                #         if high_p1 > prev_peak_price and obv_p1 < obv_at_prev_peak:
                #             return True

                # lookback_days_20 = 20
                # if limit_up_day_idx > lookback_days_20:
                #     hist_window_20d = df.iloc[limit_up_day_idx - lookback_days_20: limit_up_day_idx].copy()
                #     avg_vol_20d = hist_window_20d['volume'].mean()
                #
                #     #--- 条件56：“潜行派发”——上涨过程中的“量能污点” ---
                #     hist_window_20d['is_up_day'] = hist_window_20d['close'] > hist_window_20d['prev_close']
                #     hist_window_20d['upper_wick_ratio'] = (hist_window_20d['high'] - hist_window_20d[
                #         ['open', 'close']].max(axis=1)) / (hist_window_20d['high'] - hist_window_20d['low'])
                #
                #     stain_condition_A = (~hist_window_20d['is_up_day']) & (
                #                 hist_window_20d['volume'] > avg_vol_20d * 1.5)
                #     stain_condition_B = (hist_window_20d['is_up_day']) & (hist_window_20d['upper_wick_ratio'] > 0.4) & (
                #                 hist_window_20d['volume'] > avg_vol_20d * 1.5)
                #
                #     stain_days_count = (stain_condition_A | stain_condition_B).sum()
                #     if stain_days_count >= 3:
                #         return True


                # lookback_days_60 = 60
                # if limit_up_day_idx > lookback_days_60:
                #     hist_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]
                #
                #     # --- 条件68：“下降趋势线”的精准“狙击” ---
                #     peaks, _ = find_peaks(hist_window_60d['high'], prominence=hist_window_60d['high'].std() * 0.8,
                #                           distance=5)
                #     if len(peaks) >= 2:
                #         # 取最近的两个波段高点
                #         peak_b_idx, peak_a_idx = peaks[-1], peaks[-2]
                #         peak_b, peak_a = hist_window_60d.iloc[peak_b_idx], hist_window_60d.iloc[peak_a_idx]
                #
                #         # 确认是下降趋势
                #         if peak_b['high'] < peak_a['high']:
                #             # 计算趋势线斜率和截距
                #             x_coords = np.array([peak_a_idx, peak_b_idx])
                #             y_coords = np.array([peak_a['high'], peak_b['high']])
                #             slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
                #             # 延伸趋势线到T+0当天
                #             trendline_price_t0 = peak_b['high'] + slope * (len(hist_window_60d) - 1 - peak_b_idx)
                #
                #             if abs(limit_up_day_high - trendline_price_t0) / trendline_price_t0 < 0.02:
                #                 return True

                    # --- 条件70：深陷“巨额套牢区”的无力反弹 ---
                    # high_60d = hist_window_60d['high'].max()
                    # depth = (high_60d - limit_up_day_price) / high_60d if high_60d > 0 else 0
                    # if depth > 0.30:
                    #     return True

                # # --- 条件71：MA20的“持续性压制” ---
                # if limit_up_day_idx > 20:
                #     hist_window_20d = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
                #     days_below_ma20 = (hist_window_20d['close'] < hist_window_20d['ma20']).sum()
                #     if days_below_ma20 >= 15:
                #         return True

                # if limit_up_day_idx > 60:
                #     hist_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]
                #
                #     # --- 条件83：“大缺口”后“修复平台”的“逃生墙” ---
                #     gaps_down = hist_window_60d[hist_window_60d['open'] < hist_window_60d['prev_close'] * 0.97]
                #     if not gaps_down.empty:
                #         last_gap_day_idx = df.index.get_loc(gaps_down.index[-1])
                #         if last_gap_day_idx + 5 < limit_up_day_idx:
                #             ledge_window = df.iloc[last_gap_day_idx + 1: last_gap_day_idx + 6]
                #             ledge_high = ledge_window['high'].max()
                #             if high_p1 >= ledge_high and close_p1 < ledge_high:
                #                 return True


    return True


def second_chance_check(df: pd.DataFrame, limit_up_day_idx: int, offset: int, code: str,
                        config: StrategyConfig) -> bool:
    """
    二次机会检查的公有方法。
    如果常规检查失败，则进入此函数进行最终裁决。
    :return: 如果满足强力看涨条件，返回True，即可“复活”该股；否则返回False。
    """

    return False


def find_first_limit_up(symbol, df, config: StrategyConfig, query_tool, market_value, theme, name):
    """识别首板涨停日并排除连板"""
    limit_days = df[df['is_limit']].index.tolist()

    valid_days = []
    for day in limit_days:
        # 日期过滤条件（方便回测）
        if day < pd.Timestamp('2024-03-01') and not config.USE_2019_DATA:
            continue

        if is_valid_first_limit_up_day(df, day, symbol, config, market_value, theme, name):
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
    start_idx = df.index.get_loc(first_limit_day)

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

    # for offset in range(2, 3):  # 检查涨停后第2、3、4、5天
    for offset in [2, 4]:  # 检查涨停后第2、3、4、5天
        if start_idx + offset >= len(df):
            break

        current_day_idx = start_idx + offset
        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        if not is_valid_buy_opportunity(df, start_idx, offset, stock_code, config):
            if not second_chance_check(df, start_idx, offset, stock_code, config):
                continue

        # 计算挂单价格
        price_ma5 = simulate_ma5_order_prices(df, current_day, config.PREDICT_PRICE_INCREASE_RATIO)
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


def process_single_stock(stock_info):
    """
    处理单只股票的回测逻辑，用于并行计算。
    """
    code, name, query_tool, config = stock_info  # 接收包含所有必需信息的元组

    # 1. 加载数据
    df, _ = get_stock_data(code, config)
    if df is None or df.empty:
        return []  # 如果没有数据，返回空列表

    # 2. 准备数据（计算指标）
    df = prepare_data(df, code, config)

    # 3. 查询静态数据
    market_value = query_tool.get_stock_market_value(code)
    theme = query_tool.get_theme_by_code(code)

    # 4. 寻找信号并生成交易记录
    first_limit_days = find_first_limit_up(code, df, config, query_tool, market_value, theme, name)

    all_signals = []
    for day in first_limit_days:
        signals = generate_signals(df, day, code, name, config)
        all_signals.extend(signals)

    return all_signals


if __name__ == '__main__':
    total_start = time.perf_counter()  # 记录程序开始时间

    config = StrategyConfig()
    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values

    all_signals = []
    stock_process_start = time.perf_counter()
    tasks = [(code, name, query_tool, config) for code, name in stock_list]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map 会自动将tasks列表中的每一项分配给一个进程，并收集结果
        results = list(executor.map(process_single_stock, tasks))

    for result in results:
        all_signals.extend(result)

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
