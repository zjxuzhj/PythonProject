import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

import getAllStockCsv


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
    POSITION_ALLOCATION: Tuple[float, float, float] = (0.4, 0.35, 0.25)  # 三个挂单的仓位分配

    # --- 卖出参数 ---
    SELL_ON_MA_BREAKDOWN_THRESHOLD = -0.004  # 跌破MA5卖出阈值: (收盘价 - MA5) / MA5 <= -0.4%

    # --- 双头形态检测参数 ---
    DOUBLE_TOP_CHECK_DAYS = 40  # 检测最近40日的双头形态
    DOUBLE_TOP_PRICE_TOLERANCE = 0.04  # 两个高点价格差异容忍度（3%）
    DOUBLE_TOP_VOLUME_DECREASE_THRESHOLD = 0.8  # 第二头部成交量需小于第一头部的阈值


def simulate_ma5_order_prices(df, current_day, config: StrategyConfig, lookback_days=5):
    """模拟预测买入日MA5值，然后计算三个挂单价格"""
    current_idx = df.index.get_loc(current_day)

    if current_idx < lookback_days:
        return None, None, None, None

    prev_data = df.iloc[current_idx - lookback_days: current_idx]
    predict_ratio = config.PREDICT_PRICE_INCREASE_RATIO

    try:
        price1 = modify_last_days_and_calc_ma5(prev_data, predict_ratio)['MA5'].iloc[-1]
        price2 = modify_last_days_and_calc_ma5(prev_data, predict_ratio)['MA5'].iloc[-1]
        price3 = modify_last_days_and_calc_ma5(prev_data, predict_ratio)['MA5'].iloc[-1]

        return price1, price2, price3
    except Exception as e:
        print(f"预测MA5失败: {e}")
        return None, None, None, None


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


def _get_market_type(symbol: str) -> str:
    """根据股票代码判断所属板块。"""
    if symbol.startswith(("688", "689")):
        return "科创板"
    if symbol.startswith(("300", "301")):
        return "创业板"
    return "主板"


def find_first_limit_up(symbol, df, config: StrategyConfig):
    """识别首板涨停日并排除连板"""
    market_type = _get_market_type(symbol)
    limit_rate = config.MARKET_LIMIT_RATES[market_type]

    # 计算涨停价
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    # 识别涨停日
    limit_days = df[df['close'] >= df['limit_price']].index.tolist()

    valid_days = []
    for day in limit_days:
        # 日期过滤条件（方便回测）
        if day < pd.Timestamp('2024-03-01') and not config.USE_2019_DATA:
            continue

        day_idx = df.index.get_loc(day)

        # 条件1：排除后一日涨停
        next_day = df.index[df.index.get_loc(day) + 1] if (df.index.get_loc(day) + 1) < len(df) else None
        if next_day and df.loc[next_day, 'close'] >= df.loc[next_day, 'limit_price']:
            continue

        # 条件2：涨停后第一天涨幅>8%的排除
        next_day_idx = df.index.get_loc(day) + 1
        if next_day_idx < len(df):
            next_day = df.index[next_day_idx]
            base_price = df.loc[day, 'close']
            if abs(base_price) < 1e-5:
                continue
            next_day_change = (df.loc[next_day, 'close'] - base_price) / base_price * 100
            if next_day_change >= 8:
                continue

        #  条件3：涨停后第一天量能过滤条件（放量存在出货可能）
        if next_day_idx < len(df):
            next_day = df.index[next_day_idx]
            limit_day_volume = df.loc[day, 'volume']
            next_day_volume = df.loc[next_day, 'volume']
            next_day_open = df.loc[next_day, 'open']
            next_day_close = df.loc[next_day, 'close']
            if (next_day_volume >= limit_day_volume * 3.6) and (next_day_close < next_day_open):
                continue

        # 条件4：前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
        if df.index.get_loc(day) >= 5:
            pre5_start = df.index[df.index.get_loc(day) - 5]
            pre5_close = df.loc[pre5_start, 'close']
            if pre5_close != 0:
                total_change = (df.loc[day, 'close'] - pre5_close) / pre5_close * 100
                if total_change >= 15:
                    continue

        # 条件5：前高压制条件
        if day_idx >= 20:  # 确保20日历史数据
            # 计算前高（20日最高价）
            historical_high = df.iloc[day_idx - 20:day_idx]['high'].max()
            # 检查涨停前3日最高价是否触及前高的95%，获取涨停日前4个交易日（包括涨停日前3天、前2天、前1天，即索引位置day_idx-3到day_idx-1）
            recent_4day_high = df.iloc[day_idx - 4:day_idx]['high'].max()
            if historical_high * 0.95 <= recent_4day_high < historical_high:
                continue  # 触发排除除条件

        # 条件6：排除首板次日放量阳线+第三日低开未收复前日实体中点的情况
        if next_day_idx + 1 < len(df):  # 确保有第三日数据
            # 获取首板次日（第一天）和第三日（第二天）数据
            first_day = df.index[next_day_idx]
            first_day_data = df.loc[first_day]
            second_day = df.index[next_day_idx + 1]
            second_day_data = df.loc[second_day]

            # 条件6-1：首板次日为放量实体阳线（成交量>首板日且实体占比在总的价格范围的>50%）
            volume_condition = (first_day_data['volume'] > df.loc[day, 'volume'] * 1.5)  # 放量1.5倍
            price_range = first_day_data['high'] - first_day_data['low']
            if abs(price_range) < 1e-5:  # 若最高价=最低价（一字线），实体占比无法计算，直接排除
                candle_condition = False
            else:
                body_ratio = (first_day_data['close'] - first_day_data['open']) / price_range
                candle_condition = (body_ratio > 0.5) and (first_day_data['close'] > first_day_data['open'])

            # 条件6-2：第三日低开且未收复前日实体中点
            midpoint = (first_day_data['open'] + first_day_data['close']) / 2  # 前日阳线实体中点
            low_open_condition = (second_day_data['open'] < first_day_data['close'])  # 低开
            recover_condition = (second_day_data['close'] < midpoint)  # 盘中最高点未达中点

            if volume_condition and candle_condition and low_open_condition and recover_condition:
                continue  # 触发排除

        # 条件7：排除市值大于250亿的股票
        market_value = query_tool.get_stock_market_value(symbol)
        if market_value > config.MAX_MARKET_CAP_BILLIONS:
            continue

        # 条件8 - 排除10日内涨停次数过多的股票
        lookback_period_9 = 10
        if day_idx >= lookback_period_9:
            lookback_data_9 = df.iloc[day_idx - lookback_period_9: day_idx]
            limit_up_count = (lookback_data_9['close'] >= lookback_data_9['limit_price']).sum()
            if limit_up_count >= 4:
                continue

        # 条件9 排除特定题材
        theme = query_tool.get_theme_by_code(code)
        name = query_tool.get_name_by_code(code)
        if "证券" in name or "金融" in name or "证券" in theme or "金融" in theme:  # 牛市旗手，跟不上，不参与
            continue
        if "石油" in name or "油气" in name or "石油" in theme:  # 受海外消息影响过于严重，不参与
            continue

        # 条件10：排除10日内存在跌破一半的涨停
        lookback_period_10 = 4
        if day_idx >= lookback_period_10:
            lookback_data_10 = df.iloc[day_idx - lookback_period_10: day_idx]
            recent_limit_ups = lookback_data_10[lookback_data_10['close'] >= lookback_data_10['limit_price']]
            if not recent_limit_ups.empty:
                last_limit_up_day = recent_limit_ups.index[-1]
                # 获取最近涨停日的前一日收盘价
                prev_close_of_last_limit_up = df.loc[last_limit_up_day, 'prev_close']
                price_floor = prev_close_of_last_limit_up * 1.035
                intermediate_days_loc = slice(df.index.get_loc(last_limit_up_day) + 1, day_idx)
                intermediate_days = df.iloc[intermediate_days_loc]

                if not intermediate_days.empty:
                    min_low_in_between = intermediate_days['low'].min()
                    if min_low_in_between < price_floor:
                        continue

        valid_days.append(day)
    return valid_days


def modify_last_days_and_calc_ma5(df, predict_ratio=1.04):
    """模拟预测MA5的核心方法"""
    if df.empty or len(df) < 2:
        raise ValueError("数据不足，至少需要2个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)
    modified_df['adjusted_close'] = modified_df['close']

    new_row = modified_df.iloc[-1].copy()
    new_row['close'] *= predict_ratio
    new_row.name = new_row.name + pd.Timedelta(days=1)
    modified_df = pd.concat([modified_df, new_row.to_frame().T], axis=0)

    modified_df['MA5'] = modified_df['close'].rolling(
        window=5, min_periods=1
    ).mean().round(2)
    return modified_df


def calculate_actual_fill_price(open_price, order_price):
    """计算实际成交价：如果开盘价低于挂单价，以开盘价成交"""
    return min(open_price, order_price) if order_price is not None else None


def check_double_top(df, current_idx, config):
    """
    检测指定位置前40日内是否存在双头形态
    :return: 存在双头形态返回True，否则False
    """
    if current_idx < config.DOUBLE_TOP_CHECK_DAYS:
        return False  # 数据不足时不检测

    # 获取检测窗口数据（排除当前日）
    window_data = df.iloc[current_idx - config.DOUBLE_TOP_CHECK_DAYS: current_idx]

    # 1. 寻找两个有效高点
    high_points = []
    window_len = len(window_data)

    # 遍历窗口内的每一个数据点，不再忽略首尾
    for i in range(window_len):
        current_high = window_data.iloc[i]['high']
        neighbor_offsets = [-2, -1, 1, 2]
        valid_neighbor_indices = [i + offset for offset in neighbor_offsets if 0 <= i + offset < window_len]

        if not valid_neighbor_indices:
            continue

        is_peak = all(current_high > window_data.iloc[n_idx]['high'] for n_idx in valid_neighbor_indices)

        if is_peak:
            high_points.append((i, window_data.iloc[i]['high'], window_data.iloc[i]['volume']))

    # 如果找到的高点少于2个，不可能形成双头
    if len(high_points) < 2:
        return False

    # 2. 检测符合条件的双头
    for i in range(len(high_points) - 1):
        first_top_idx, first_top_price, first_top_vol = high_points[i]
        second_top_idx, second_top_price, second_top_vol = high_points[i + 1]

        # 价格条件：两个高点差异在3%以内
        price_diff = abs(first_top_price - second_top_price) / min(first_top_price, second_top_price)
        if price_diff < config.DOUBLE_TOP_PRICE_TOLERANCE:
            return True
            # continue

        # 成交量条件：第二头部成交量需小于第一头部（至少减少20%）
        # if second_top_vol > first_top_vol * config.DOUBLE_TOP_VOLUME_DECREASE_THRESHOLD:
        #     continue

        # 颈线检测：两个高点之间的最低点
        neckline_idx = window_data.iloc[first_top_idx:second_top_idx]['low'].idxmin()
        neckline_slice = window_data.iloc[first_top_idx + 1: second_top_idx]
        if neckline_slice.empty:
            continue  # 两个高点相邻，没有中间区域，无法形成颈线

        neckline_price = neckline_slice['low'].min()

        # 突破确认：第二头部后需有收盘价跌破颈线
        # 检查从第二个高点之后到窗口结尾的数据
        breakdown_slice = window_data.iloc[second_top_idx + 1:]
        if not breakdown_slice.empty:
            if (breakdown_slice['close'] < neckline_price).any():
                print(
                    f"检测到双头形态: Top1 at index {first_top_idx}, Top2 at index {second_top_idx}, Neckline: {neckline_price:.2f}")
                return True  # 检测到有效双头形态

    return False


def generate_signals(df, first_limit_day, stock_code, stock_name, config: StrategyConfig):
    """生成买卖信号"""
    signals = []
    first_limit_timestamp = pd.Timestamp(first_limit_day)
    market_type = "科创板" if stock_code.startswith(("688", "689")) else "创业板" if stock_code.startswith(
        ("300", "301")) else "主板"
    limit_rate = 0.20 if market_type in ["创业板", "科创板"] else 0.10

    base_price = df.loc[first_limit_day, 'close']  # 首板收盘价，最重要的位置，表示主力的支撑度

    next_day_2_pct = None
    start_idx = df.index.get_loc(first_limit_day)
    if start_idx + 2 < len(df):  # 确保有第二个交易日数据
        next_day_1 = df.index[start_idx + 1]
        next_day_2 = df.index[start_idx + 2]
        base_price = df.loc[first_limit_day, 'close']
        next_day_1_close = df.loc[next_day_1, 'close']
        next_day_2_close = df.loc[next_day_2, 'low']
        if abs(next_day_1_close) > 1e-5:  # 防止除零错误
            next_day_2_pct = (next_day_2_close - next_day_1_close) / next_day_1_close * 100

    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)

    start_idx = df.index.get_loc(first_limit_day)
    if (start_idx + 1) >= len(df):  # 边界检查，跳过无数据的情况
        return signals

    df['ma5'] = df['close'].rolling(5).mean()
    # df['ma10'] = df['close'].rolling(10).mean()
    # df['ma20'] = df['close'].rolling(20).mean()
    # df['ma30'] = df['close'].rolling(30).mean()
    # df['ma55'] = df['close'].rolling(55).mean()

    # for offset in range(2,3):  # 检查涨停后第2、3、4、5天
    for offset in [2, 4]:  # 检查涨停后第2、3、4、5天
        if start_idx + offset >= len(df):
            break

        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        # 买前条件1 检查中间是否有新的涨停
        has_new_limit = False
        for i in range(1, offset):
            check_day = df.index[start_idx + i]
            if df.loc[check_day, 'close'] >= df.loc[check_day, 'limit_price']:
                has_new_limit = True
                break
        if has_new_limit:
            continue

        # 买前条件2 买入前收盘价不低于首板收盘价
        price_condition_met = True
        for check_offset in range(1, offset):
            check_day = df.index[start_idx + check_offset]
            if df.loc[check_day, 'close'] < base_price:
                price_condition_met = False
                break
        if not price_condition_met:
            continue

        # 买前条件3 获取最近5日MA5数据（防止空值）
        ma5_data = df['ma5'].iloc[start_idx:start_idx + offset + 1]
        if ma5_data.isnull().any():
            continue

        # 核心条件：当日最低价触碰五日均线
        touch_condition = (current_data['low'] <= current_data['ma5']) & \
                          (current_data['high'] >= current_data['ma5'])

        # 计算三个挂单价格
        price1, price2, price3 = simulate_ma5_order_prices(df, current_day, config)
        if price1 is None:
            continue

        # 获取当日价格数据
        day_open = current_data['open']
        day_low = current_data['low']
        day_high = current_data['high']

        # 检查挂单价是否在当日价格范围内
        order1_valid = day_low <= price1 <= day_high
        order2_valid = day_low <= price2 <= day_high
        order3_valid = day_low <= price3 <= day_high

        # 如果没有一个价格有效，则跳过
        if not (order1_valid or order2_valid or order3_valid):
            continue

        if price1 >= day_open:
            order1_valid = True

        # 计算实际成交价（考虑开盘价）
        actual_price1 = calculate_actual_fill_price(day_open, price1) if order1_valid else None
        actual_price2 = calculate_actual_fill_price(day_open, price2) if order2_valid else None
        actual_price3 = calculate_actual_fill_price(day_open, price3) if order3_valid else None

        # 仓位分配比例（修正为正确的比例）
        position_percentage1 = config.POSITION_ALLOCATION[0] if actual_price1 else 0.0
        position_percentage2 = config.POSITION_ALLOCATION[1] if actual_price2 else 0.0
        position_percentage3 = config.POSITION_ALLOCATION[2] if actual_price3 else 0.0

        # 确保总仓位不超过100%
        total_percentage = position_percentage1 + position_percentage2 + position_percentage3
        if abs(total_percentage) < 1e-5:
            continue

        # 计算加权平均买入价
        weighted_avg_price = (
                                     (position_percentage1 * actual_price1 if actual_price1 else 0) +
                                     (position_percentage2 * actual_price2 if actual_price2 else 0) +
                                     (position_percentage3 * actual_price3 if actual_price3 else 0)
                             ) / total_percentage

        buy_day_timestamp = pd.Timestamp(current_day)
        days_after_limit = (buy_day_timestamp - first_limit_timestamp).days
        history_window = df.iloc[start_idx + 1: start_idx + offset]
        history_condition = (history_window['close'] > history_window['ma5']).all()

        if not history_condition:
            continue

        # 持有期卖出逻辑
        hold_days = 0
        for sell_offset in range(1, 20):
            if start_idx + offset + sell_offset >= len(df):
                break

            # 存在双头形态，跳过买入
            # if check_double_top(df, start_idx + offset + sell_offset, config):
            #     continue

            sell_day = df.index[start_idx + offset + sell_offset]
            sell_data = df.loc[sell_day]
            hold_days += 1

            # # 卖出时检查，如果是前一天买入，并且当前价格距离20日线，30日线，55日线任意两条的距离偏差小于1%，暂时不卖出
            # if hold_days == 1 or hold_days == 2:
            #     price_gap = 0.01
            #     current_price = sell_data['close']
            #     ma10 = sell_data.get('ma10')
            #     ma20 = sell_data.get('ma20')
            #     ma30 = sell_data.get('ma30')
            #     ma55 = sell_data.get('ma55')
            #
            #     buy_day_timestamp = pd.Timestamp(current_day)
            #     days_after_limit = (buy_day_timestamp - first_limit_timestamp).days
            #     history_window = df.iloc[start_idx + 1: start_idx + offset]
            #     history_condition = (history_window['close'] > history_window['ma5']).all()
            #     if not history_condition:
            #         continue
            #
            #     buy_day_idx = df.index.get_loc(current_day)
            #     if buy_day_idx < 4:  # 确保有足够的数据进行5日回看
            #         continue
            #
            #     lookback_window = df.iloc[buy_day_idx - 4: buy_day_idx + 1]  # 获取5日窗口
            #     successful_days = 0
            #     for i in range(len(lookback_window)):
            #         day_data = lookback_window.iloc[i]
            #         close_price = day_data['close']
            #         ma10 = day_data.get('ma10')
            #         ma20 = day_data.get('ma20')
            #         ma30 = day_data.get('ma30')
            #         ma55 = day_data.get('ma55')
            #
            #         # 确保均线值有效
            #         if not (pd.notna(ma20) and pd.notna(ma30) and pd.notna(ma55)):
            #             continue
            #
            #         ma_above_count = 0
            #         if close_price > ma10:
            #             ma_above_count += 1
            #         if close_price > ma20:
            #             ma_above_count += 1
            #         if close_price > ma30:
            #             ma_above_count += 1
            #         if close_price > ma55:
            #             ma_above_count += 1
            #
            #         if ma_above_count >= 2:  # 如果收盘价在至少两条均线之上
            #             successful_days += 1
            #
            #     # Ensure MAs are valid numbers before calculation to avoid errors
            #     if pd.notna(ma20) and pd.notna(ma30) and pd.notna(ma55) and ma20 > 0 and ma30 > 0 and ma55 > 0:
            #         dist_ma10 = abs(current_price - ma10) / ma10
            #         dist_ma20 = abs(current_price - ma20) / ma20
            #         dist_ma30 = abs(current_price - ma30) / ma30
            #         dist_ma55 = abs(current_price - ma55) / ma55
            #
            #         # Count how many MAs are within the 1% threshold
            #         ma_proximity_count = 0
            #         if dist_ma10 < price_gap:
            #             ma_proximity_count += 1
            #         if dist_ma20 < price_gap:
            #             ma_proximity_count += 1
            #         if dist_ma30 < price_gap:
            #             ma_proximity_count += 1
            #         if dist_ma55 < price_gap:
            #             ma_proximity_count += 1
            #
            #         # If close to 2 or more MAs, skip selling for this day and hold
            #         if ma_proximity_count >= 2 and successful_days >= 3:
            #             print(
            #                 f"[{sell_day.strftime('%Y-%m-%d')}] 触发持有条件: 股票 {stock_code} ({stock_name}) 接近多条均线，暂时持有。首板日: {first_limit_day.strftime('%Y-%m-%d')}")
            #             continue  # Defer sell and check again next day

            # 1. 跌停第二天卖出
            if df.index.get_loc(sell_day) >= 1:
                prev_day = df.index[df.index.get_loc(sell_day) - 1]
                prev_day_data = df.loc[prev_day]
                # 判断前一天是否跌停
                if prev_day_data['close'] <= prev_day_data['down_limit_price']:
                    # 第二天收盘价卖出
                    sell_price = sell_data['close']
                    profit_pct = (sell_price - weighted_avg_price) / weighted_avg_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '实际均价': round(weighted_avg_price, 2),
                        '卖出价': round(sell_price, 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        # '收益率(%)': round(profit_pct, 2)*round(total_percentage, 2),
                        '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
                        '卖出原因': '跌停止损',
                        '挂单价1': round(price1, 2) if price1 else None,
                        '挂单价2': round(price2, 2) if price2 else None,
                        '挂单价3': round(price3, 2) if price3 else None,
                        '实际成交价1': round(actual_price1, 2) if actual_price1 else None,
                        '实际成交价2': round(actual_price2, 2) if actual_price2 else None,
                        '实际成交价3': round(actual_price3, 2) if actual_price3 else None,
                        '是否成交1': '是' if actual_price1 else '否',
                        '是否成交2': '是' if actual_price2 else '否',
                        '是否成交3': '是' if actual_price3 else '否',
                        '买入比例': round(total_percentage * 100, 2)
                    })
                    break

                if sell_data['close'] >= sell_data['limit_price']:
                    continue  # 涨停日继续持有

                # 2. 涨停后断板日卖出
                prev_day = df.index[df.index.get_loc(sell_day) - 1]
                if df.loc[prev_day, 'close'] >= df.loc[prev_day, 'limit_price']:
                    sell_price = sell_data['close']
                    profit_pct = (sell_price - weighted_avg_price) / weighted_avg_price * 100
                    signals.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '首板日': first_limit_day.strftime('%Y-%m-%d'),
                        '买入日': current_day.strftime('%Y-%m-%d'),
                        '卖出日': sell_day.strftime('%Y-%m-%d'),
                        '涨停后天数': days_after_limit,
                        '持有天数': hold_days,
                        '实际均价': round(weighted_avg_price, 2),
                        '卖出价': round(sell_price, 2),
                        '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                        '收益率(%)': round(profit_pct, 2),
                        # '收益率(%)': round(profit_pct, 2)*round(total_percentage, 2),
                        '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
                        '卖出原因': '断板止盈',
                        '挂单价1': round(price1, 2) if price1 else None,
                        '挂单价2': round(price2, 2) if price2 else None,
                        '挂单价3': round(price3, 2) if price3 else None,
                        '实际成交价1': round(actual_price1, 2) if actual_price1 else None,
                        '实际成交价2': round(actual_price2, 2) if actual_price2 else None,
                        '实际成交价3': round(actual_price3, 2) if actual_price3 else None,
                        '是否成交1': '是' if actual_price1 else '否',
                        '是否成交2': '是' if actual_price2 else '否',
                        '是否成交3': '是' if actual_price3 else '否',
                        '买入比例': round(total_percentage * 100, 2)
                    })
                    break

            # 3. 首板炸板卖出条件
            # 条件3-1：当日最高价达到涨停价但未封板（收盘价<涨停价
            is_limit_touched = (sell_data['high'] >= sell_data['limit_price'])
            is_limit_closed = (sell_data['close'] < sell_data['limit_price'])
            # 条件3-2：前一日未涨停，也就是首板
            prev_day = df.index[df.index.get_loc(sell_day) - 1]
            prev_limit_price = (df.loc[prev_day, 'prev_close'] * (1 + limit_rate)).round(2)
            is_prev_limit = (df.loc[prev_day, 'close'] >= prev_limit_price)

            if is_limit_touched and is_limit_closed and not is_prev_limit:
                sell_price = sell_data['close']
                profit_pct = (sell_price - weighted_avg_price) / weighted_avg_price * 100
                signals.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '首板日': first_limit_day.strftime('%Y-%m-%d'),
                    '买入日': current_day.strftime('%Y-%m-%d'),
                    '卖出日': sell_day.strftime('%Y-%m-%d'),
                    '涨停后天数': days_after_limit,
                    '持有天数': hold_days,
                    '实际均价': round(weighted_avg_price, 2),
                    '卖出价': round(sell_price, 2),
                    '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                    '收益率(%)': round(profit_pct, 2),
                    # '收益率(%)': round(profit_pct, 2)*round(total_percentage, 2),
                    '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
                    '卖出原因': '炸板卖出',
                    '挂单价1': round(price1, 2) if price1 else None,
                    '挂单价2': round(price2, 2) if price2 else None,
                    '挂单价3': round(price3, 2) if price3 else None,
                    '实际成交价1': round(actual_price1, 2) if actual_price1 else None,
                    '实际成交价2': round(actual_price2, 2) if actual_price2 else None,
                    '实际成交价3': round(actual_price3, 2) if actual_price3 else None,
                    '是否成交1': '是' if actual_price1 else '否',
                    '是否成交2': '是' if actual_price2 else '否',
                    '是否成交3': '是' if actual_price3 else '否',
                    '买入比例': round(total_percentage * 100, 2)
                })
                break

            # 3. 跌破五日线卖出(改为跌破五日线千分之三卖出)
            # if sell_data['close'] < sell_data['ma5']:
            if (sell_data['close'] - sell_data['ma5']) / sell_data['ma5'] <= config.SELL_ON_MA_BREAKDOWN_THRESHOLD:
                sell_price = sell_data['close']
                # shadow_up_ratio = (sell_data['high'] - sell_data['close']) / sell_data['close']
                # # 长下影线判断（收盘价-最低价）/最低价 > 3%
                # shadow_down_ratio = (sell_data['close'] - sell_data['low']) / sell_data['low']
                # if shadow_down_ratio > 0.03 and shadow_up_ratio < 0.02:  # 长下影线超过3%，且上影线小于1.5%
                #     print(f"[{sell_day.strftime('%Y-%m-%d')}] 长下影线拉回({shadow_down_ratio:.2%})，继续持有")
                #     continue  # 跳过卖出，继续持有观察
                profit_pct = (sell_price - weighted_avg_price) / weighted_avg_price * 100
                signals.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '首板日': first_limit_day.strftime('%Y-%m-%d'),
                    '买入日': current_day.strftime('%Y-%m-%d'),
                    '卖出日': sell_day.strftime('%Y-%m-%d'),
                    '涨停后天数': days_after_limit,
                    '持有天数': hold_days,
                    '实际均价': round(weighted_avg_price, 2),
                    '卖出价': round(sell_price, 2),
                    '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                    '收益率(%)': round(profit_pct, 2),
                    # '收益率(%)': round(profit_pct, 2)*round(total_percentage, 2),
                    '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
                    '卖出原因': '跌破五日线',
                    '挂单价1': round(price1, 2) if price1 else None,
                    '挂单价2': round(price2, 2) if price2 else None,
                    '挂单价3': round(price3, 2) if price3 else None,
                    '实际成交价1': round(actual_price1, 2) if actual_price1 else None,
                    '实际成交价2': round(actual_price2, 2) if actual_price2 else None,
                    '实际成交价3': round(actual_price3, 2) if actual_price3 else None,
                    '是否成交1': '是' if actual_price1 else '否',
                    '是否成交2': '是' if actual_price2 else '否',
                    '是否成交3': '是' if actual_price3 else '否',
                    '买入比例': round(total_percentage * 100, 2)
                })
                break

            # 4. 最大持有天数限制（15天）
            if hold_days >= 15:
                sell_price = sell_data['close']
                profit_pct = (sell_price - weighted_avg_price) / weighted_avg_price * 100
                signals.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '首板日': first_limit_day.strftime('%Y-%m-%d'),
                    '买入日': current_day.strftime('%Y-%m-%d'),
                    '卖出日': sell_day.strftime('%Y-%m-%d'),
                    '涨停后天数': days_after_limit,
                    '持有天数': hold_days,
                    '实际均价': round(weighted_avg_price, 2),
                    '卖出价': round(sell_price, 2),
                    '触碰类型': 'MA5支撑反弹' if current_data['close'] > current_data['ma5'] else 'MA5破位回升',
                    '收益率(%)': round(profit_pct, 2),
                    # '收益率(%)': round(profit_pct, 2)*round(total_percentage, 2),
                    '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
                    '卖出原因': '持有超限',
                    '挂单价1': round(price1, 2) if price1 else None,
                    '挂单价2': round(price2, 2) if price2 else None,
                    '挂单价3': round(price3, 2) if price3 else None,
                    '实际成交价1': round(actual_price1, 2) if actual_price1 else None,
                    '实际成交价2': round(actual_price2, 2) if actual_price2 else None,
                    '实际成交价3': round(actual_price3, 2) if actual_price3 else None,
                    '是否成交1': '是' if actual_price1 else '否',
                    '是否成交2': '是' if actual_price2 else '否',
                    '是否成交3': '是' if actual_price3 else '否',
                    '买入比例': round(total_percentage * 100, 2)
                })
                break
        break  # 只处理第一个符合条件的买入点
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
                    '持有天数', '实际均价', '卖出价', '涨停后第二日涨幅(%)', '收益率(%)', '卖出原因',
                    '挂单价1', '挂单价2', '挂单价3', '实际成交价1', '实际成交价2', '实际成交价3',
                    '是否成交1', '是否成交2', '是否成交3', '买入比例']
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
        three_months_ago = current_date - pd.DateOffset(months=4)

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
        print("\n\033[1m最近5笔交易记录:\033[0m")
        print(result_df.tail(5).to_string(index=False))
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
