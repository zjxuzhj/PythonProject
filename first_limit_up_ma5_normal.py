import concurrent.futures
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import getAllStockCsv
from ConditionChecker import ConditionChecker
from common_sell_logic import get_sell_decision, MarketDataContext
from stock_info import StockInfo
from strategy_rules import RuleEnum


@dataclass
class StrategyConfig:
    """集中存放所有策略参数，便于统一调整。"""
    # --- 数据设置 ---
    USE_2019_DATA: bool = False  # False 使用2024年数据, True 使用2019年数据

    USE_SELL_LOGIC: bool = False  # False 回测买入时的排除条件, True 回测卖出条件

    USE_SINGLE_RULE: bool = False  # False 进入正常回测所有排除条件， True 单独回测一个条件

    # <<<<<<<<<<<<<<<< 买入日偏移量配置 <<<<<<<<<<<<<<<<
    # BUY_OFFSETS: list[int] = field(default_factory=lambda: [2])
    # BUY_OFFSETS: list[int] = field(default_factory=lambda: [4])
    BUY_OFFSETS: list[int] = field(default_factory=lambda: [2, 4])

    # --- 首板涨停识别参数 ---
    MARKET_LIMIT_RATES = {'主板': 0.10, '创业板': 0.20, '科创板': 0.20}
    MAX_MARKET_CAP_BILLIONS = 250  # 条件7: 最大市值过滤（单位：亿）

    # --- 买入参数 ---
    PREDICT_PRICE_INCREASE_RATIO = 1.04  # 用于预测MA5的价格涨幅
    SCORE_THRESHOLD: int = 75  # <<<<<<<<<<<<<<<< 新增：买入的最低质量分门槛

    # --- 双头形态检测参数 ---
    DOUBLE_TOP_CHECK_DAYS = 40  # 检测最近40日的双头形态
    DOUBLE_TOP_PRICE_TOLERANCE = 0.04  # 两个高点价格差异容忍度（3%）
    DOUBLE_TOP_VOLUME_DECREASE_THRESHOLD = 0.8  # 第二头部成交量需小于第一头部的阈值


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


def calculate_quality_score(stock_info: StockInfo, df: pd.DataFrame, limit_up_day_idx: int, offset: int,
                            config: StrategyConfig) -> tuple[int, list[str]]:
    """
    计算给定买入机会的质量得分。
    - 基准分100分。
    - 根据积极信号加分，根据消极信号减分。
    - 返回最终得分和评分原因列表。
    """
    score = 20
    reasons = []
    checker = ConditionChecker(df, limit_up_day_idx=limit_up_day_idx, offset=offset, stock_info=stock_info)

    # --- 提取常用数据变量，保持代码清晰 ---
    limit_up_day_data = df.iloc[limit_up_day_idx]
    limit_up_day_price = limit_up_day_data['close']
    limit_up_day_volume = limit_up_day_data['volume']
    limit_up_day_low = limit_up_day_data['low']
    limit_up_close = limit_up_day_data['close']
    day_minus_1_data = df.iloc[limit_up_day_idx - 1]
    day_plus_1_data = df.iloc[limit_up_day_idx + 1]
    day_minus_1_idx = limit_up_day_idx - 1
    day_minus_1_day_date = df.index[day_minus_1_idx]
    day_minus_1_data = df.iloc[day_minus_1_idx]
    open_m1 = day_minus_1_data['open']
    high_m1 = day_minus_1_data['high']
    low_m1 = day_minus_1_data['low']
    close_m1 = day_minus_1_data['close']
    volume_m1 = day_minus_1_data['volume']
    volume_p1 = day_plus_1_data['volume']
    close_m1 = df.iloc[limit_up_day_idx - 1]['close']
    ma30_m1 = df.iloc[limit_up_day_idx - 1]['ma30']
    ma60_m1 = df.iloc[limit_up_day_idx - 1]['ma60']
    day_minus_2_data = df.iloc[limit_up_day_idx - 2]
    open_m2 = day_minus_2_data['open']
    high_m2 = day_minus_2_data['high']
    low_m2 = day_minus_2_data['low']
    close_m2 = day_minus_2_data['close']
    high_p1 = day_plus_1_data['high']
    open_p1 = day_plus_1_data['open']
    close_p1 = day_plus_1_data['close']
    low_p1 = day_plus_1_data['low']

    # 和60日线距离
    if pd.notna(ma60_m1) and ma60_m1 > 0:
        bias_ratio_60 = (close_m1 - ma60_m1) / ma60_m1
        # 总交易数: 161 ，胜率: 51.55%，均盈: 14.28% | 均亏: 4.13% | 均持: 2.16，盈亏比: 3.46:1，期望: 5.359
        # 总交易数: 440 ，胜率: 43.64%，均盈: 11.34% | 均亏: 4.59% | 均持: 1.89，盈亏比: 2.47:1，期望: 2.363
        if 0.15 < bias_ratio_60:
            score += 10
        # 总交易数: 118 ，胜率: 38.98%，均盈: 8.44% | 均亏: 2.74% | 均持: 2.24，盈亏比: 3.08:1，期望: 1.621
        # 总交易数: 364 ，胜率: 34.34%，均盈: 8.12% | 均亏: 3.63% | 均持: 1.82，盈亏比: 2.24:1，期望: 0.404
        if 0.1 < bias_ratio_60 <= 0.15:
            score -= 10
        # 总交易数: 152 ，胜率: 45.39%，均盈: 5.34% | 均亏: 2.98% | 均持: 2.12，盈亏比: 1.79:1，期望: 0.796
        # 总交易数: 707 ，胜率: 41.44%，均盈: 5.74% | 均亏: 2.64% | 均持: 2.22，盈亏比: 2.17:1，期望: 0.832
        if -0.09 < bias_ratio_60 <= -0.04:
            score -= 10

    # 和30日线距离
    bias_ratio = (close_m1 - ma30_m1) / ma30_m1
    # 总交易数: 129 ，胜率: 48.84%，均盈: 12.94% | 均亏: 3.79% | 均持: 2.02，盈亏比: 3.41:1，期望: 4.379
    # 总交易数: 357 ，胜率: 43.42%，均盈: 10.88% | 均亏: 4.81% | 均持: 1.84，盈亏比: 2.26:1，期望: 2.004
    if 0.1 < bias_ratio:
        score += 10
    # 总交易数: 116 ，胜率: 49.14%，均盈: 10.01% | 均亏: 2.85% | 均持: 3.82，盈亏比: 3.51:1，期望: 3.469
    # 总交易数: 512 ，胜率: 59.77%，均盈: 8.28% | 均亏: 3.00% | 均持: 3.77，盈亏比: 2.76:1，期望: 3.745
    if bias_ratio <= -0.14:
        score += 10
    # 总交易数: 164 ，胜率: 45.12%，均盈: 7.17% | 均亏: 2.92% | 均持: 2.30，盈亏比: 2.46:1，期望: 1.636
    # 总交易数: 742 ，胜率: 37.87%，均盈: 6.55% | 均亏: 3.01% | 均持: 2.24，盈亏比: 2.18:1，期望: 0.610
    if -0.14 < bias_ratio <= -0.07:
        score -= 10

    is_red_candle_m1 = close_m1 > open_m1
    # 计算T-1之前的10日平均成交量
    avg_volume_pre_10d = df.iloc[limit_up_day_idx - 11: limit_up_day_idx - 1]['volume'].mean()
    if avg_volume_pre_10d > 0 and is_red_candle_m1:
        # 总交易数: 168 ，胜率: 39.88%，均盈: 9.10% | 均亏: 3.03% | 均持: 1.89，盈亏比: 3.00:1，期望: 1.804
        # 总交易数: 573 ，胜率: 39.62%，均盈: 6.61% | 均亏: 2.95% | 均持: 1.91，盈亏比: 2.24:1，期望: 0.839
        if avg_volume_pre_10d * 0.4 < volume_m1 <= (avg_volume_pre_10d * 0.7):
            score -= 10

    full_range = high_p1 - low_p1
    if full_range > 1e-5:
        upper_body = max(open_p1, close_p1)
        upper_wick_length = high_p1 - upper_body
        # 定义长上影线为上影线长度占K线总长度的比例
        rate = (upper_wick_length / full_range)
        # 总交易数: 104 ，胜率: 34.62%，均盈: 8.40% | 均亏: 2.47% | 均持: 2.29，盈亏比: 3.40:1，期望: 1.292
        # 总交易数: 451 ，胜率: 39.02%，均盈: 7.53% | 均亏: 2.97% | 均持: 2.33，盈亏比: 2.54:1，期望: 1.130
        if 0.1 < rate <= 0.2:
            score -= 5
        # 总交易数: 117 ，胜率: 47.01%，均盈: 6.68% | 均亏: 2.83% | 均持: 1.91，盈亏比: 2.36:1，期望: 1.639
        # 总交易数: 494 ，胜率: 41.50%，均盈: 7.20% | 均亏: 3.10% | 均持: 2.10，盈亏比: 2.32:1，期望: 1.174
        if 0.4 < rate <= 0.45:
            score -= 5

    pre_window_30d = df.iloc[limit_up_day_idx - 30: limit_up_day_idx]
    daily_change_pct = (pre_window_30d['close'].pct_change(fill_method=None).abs())
    # 如果在过去30天里，有超过11天（50%）的日收盘价变动小于1%，则认为是极端沉寂
    stagnant_days_count = (daily_change_pct < 0.01).sum()
    # 总交易数: 213 ，胜率: 48.83%，均盈: 10.37% | 均亏: 2.80% | 均持: 2.28，盈亏比: 3.71:1，期望: 3.633
    # 总交易数: 822 ，胜率: 43.80%，均盈: 7.92% | 均亏: 3.33% | 均持: 2.23，盈亏比: 2.38:1，期望: 1.595
    if 4 < stagnant_days_count <= 6:
        score += 10

    # 涨停后第一日低开幅度
    if (limit_up_close * 1.00) < low_p1 <= (limit_up_close * 1.02):
        # 总交易数: 320 ，胜率: 45.62 %，均盈: 10.29 % | 均亏: 2.86 % | 均持: 2.04，盈亏比: 3.60:1，期望: 3.141
        # 总交易数: 1166，胜率: 43.83 %，均盈: 8.04 % | 均亏: 3.01 % | 均持: 2.30，盈亏比: 2.67:1，期望: 1.833
        score += 10
    elif (limit_up_close * 0.980) < low_p1 <= (limit_up_close * 0.985):
        # 总交易数: 125 ，胜率: 41.60 %，均盈: 9.02 % | 均亏: 2.69 % | 均持: 2.43，盈亏比: 3.35:1，期望: 2.178
        # 总交易数: 465 ，胜率: 37.85 %，均盈: 6.90 % | 均亏: 3.03 % | 均持: 2.32，盈亏比: 2.28:1，期望: 0.731
        score -= 10
    elif (limit_up_close * 0.96) < low_p1 <= (limit_up_close * 0.97):
        # 总交易数: 123 ，胜率: 40.65 %，均盈: 9.27 % | 均亏: 3.24 % | 均持: 2.03，盈亏比: 2.86:1，期望: 1.848
        # 总交易数: 442 ，胜率: 39.82 %，均盈: 6.39 % | 均亏: 3.55 % | 均持: 2.14，盈亏比: 1.80:1，期望: 0.407
        score -= 10

    # 涨停前30日内从最低点上涨幅度
    pre_limit_up_window = df.iloc[limit_up_day_idx - 30: limit_up_day_idx]
    lowest_low_30d = pre_limit_up_window['low'].min()
    price_before_limit_up = close_m1
    if lowest_low_30d > 0:
        gain_from_low = (price_before_limit_up - lowest_low_30d) / lowest_low_30d
        if 0 < gain_from_low <= 0.025:
            # 总交易数: 164 ，胜率: 51.83%，均盈: 8.79% | 均亏: 3.04% | 均持: 3.51，盈亏比: 2.90:1，期望: 3.095
            # 总交易数: 592 ，胜率: 47.97%，均盈: 7.79% | 均亏: 2.92% | 均持: 2.95，盈亏比: 2.66:1，期望: 2.215
            score += 5
        elif 0.10 < gain_from_low <= 0.125:
            # 总交易数: 124 ，胜率: 45.16%，均盈: 10.89% | 均亏: 2.57% | 均持: 2.27，盈亏比: 4.24:1，期望: 3.510
            # 总交易数: 378 ，胜率: 44.71%，均盈: 7.49% | 均亏: 2.68% | 均持: 2.20，盈亏比: 2.80:1，期望: 1.869
            score += 5
        elif 0.29 < gain_from_low:
            # 总交易数: 168 ，胜率: 51.19%，均盈: 12.76% | 均亏: 4.21% | 均持: 2.01，盈亏比: 3.03:1，期望: 4.475
            # 总交易数: 539 ，胜率: 43.78%，均盈: 9.82% | 均亏: 4.59% | 均持: 1.85，盈亏比: 2.14:1，期望: 1.722
            score += 10

    # --- 均线粘合支撑相关（可用性存疑，后续验证）---
    if checker.is_555_m1_abs_nian_he():
        # 总交易数: 83  ，胜率: 54.22 %，均盈: 8.03 % | 均亏: 1.95 % | 均持: 2.11，盈亏比: 4.11:1，期望: 3.460
        # 总交易数: 381 ，胜率: 43.31 %，均盈: 7.12 % | 均亏: 2.78 % | 均持: 2.28，盈亏比: 2.56:1，期望: 1.505
        score += 10
        reasons.append("(+10) 涨停前一日五日五十五日线粘合")
    if checker.is_120_p0m1m2_support():
        # 总交易数: 54  ，胜率: 53.70%，均盈: 8.75% | 均亏: 2.26% | 均持: 2.04，盈亏比: 3.88:1，期望: 3.654
        # 总交易数: 290 ，胜率: 40.00%，均盈: 7.49% | 均亏: 2.54% | 均持: 2.02，盈亏比: 2.95:1，期望: 1.473
        score += 10
        reasons.append("(+10) 涨停日以及前两日120日线支撑")

    # --- 涨停日最低价 ---
    # 检查涨停日最低价是否大幅高于前一日收盘价，这通常意味着非常强势的跳空高开且未回补缺口
    if (close_m1 * 1.02) < limit_up_day_low <= (close_m1 * 1.05):
        # 总交易数: 144 ，胜率: 46.53%，均盈: 7.16% | 均亏: 3.13% | 均持: 2.62，盈亏比: 2.28:1，期望: 1.655
        # 总交易数: 507 ，胜率: 40.63%，均盈: 6.83% | 均亏: 3.03% | 均持: 2.41，盈亏比: 2.25:1，期望: 0.973
        score -= 10
        reasons.append("(-10) 涨停日最低价>昨收1.020<昨收1.050")
    elif (close_m1 * 0.985) < limit_up_day_low <= (close_m1 * 0.990):
        # 总交易数: 113 ，胜率: 51.33 %，均盈: 9.51 % | 均亏: 2.83 % | 均持: 2.26，盈亏比: 3.36:1，期望: 3.501
        # 总交易数: 468 ，胜率: 43.38%，均盈: 7.18% | 均亏: 3.19% | 均持: 2.19，盈亏比: 2.25:1，期望: 1.311
        score += 10
        reasons.append("(+10) 涨停日最低价>昨收0.985<昨收0.990")

    # --- 均线趋势相关---
    ma5_p0, ma10_p0, ma20_p0, ma55_p0 = day_minus_1_data[['ma5', 'ma10', 'ma20', 'ma60']]
    if all(pd.notna([ma5_p0, ma10_p0, ma20_p0, ma55_p0])):
        if day_minus_1_data['close'] > ma5_p0 > ma10_p0 > ma20_p0 > ma55_p0:
            # 总交易数: 101 ，胜率: 40.59%，均盈: 9.76% | 均亏: 3.06% | 均持: 2.06，盈亏比: 3.19:1，期望: 2.147
            # 总交易数: 280 ，胜率: 40.71%，均盈: 6.90% | 均亏: 4.33% | 均持: 1.75，盈亏比: 1.59:1，期望: 0.244
            score -= 10
            reasons.append("(-10) 黄金趋势背景")

    # --- 前五日累计涨幅 ---
    # 前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
    pre5_start = df.index[limit_up_day_idx - 5]
    pre5_close = df.loc[pre5_start, 'close']
    if pre5_close != 0:
        total_change = (limit_up_day_price - pre5_close) / pre5_close * 100
        if total_change < 1:
            # 总交易数: 214 ，胜率: 48.13%，均盈: 10.08% | 均亏: 3.35% | 均持: 3.01，盈亏比: 3.01:1，期望: 3.115
            score += 10
            reasons.append("(+10) 四天累计只能涨1%")
        elif 3 <= total_change < 6:
            # 总交易数: 191 ，胜率: 52.36%，均盈: 10.26% | 均亏: 2.68% | 均持: 2.37，盈亏比: 3.82:1，期望: 4.095
            score += 10
            reasons.append("(+10) 四天累计涨幅在3-5%内")

    # --- 涨停后第一天和涨停日的量能相关(没有验证19年数据，需重新检验) ---
    if volume_p1 <= limit_up_day_volume * 1.3:
        # 总交易数: 206 ，胜率: 52.91%，均盈: 10.84% | 均亏: 2.78% | 均持: 2.67，盈亏比: 3.91:1，期望: 4.430
        score += 10
        reasons.append("(+10) 成交量占比小于1.3")
    elif limit_up_day_volume * 1.5 < volume_p1 <= limit_up_day_volume * 1.8:
        # 总交易数: 249 ，胜率: 46.18%，均盈: 10.65% | 均亏: 2.86% | 均持: 2.32，盈亏比: 3.73:1，期望: 3.382
        score += 10
        reasons.append("(+10) 成交量占比在1.5-1.8内")
    elif limit_up_day_volume * 3 < volume_p1 <= limit_up_day_volume * 4.5:
        # 总交易数: 159 ，胜率: 47.17 %，均盈: 7.09 % | 均亏: 3.24 % | 均持: 1.99，盈亏比: 2.19:1，期望: 1.632
        score -= 10
        reasons.append("(-10) 成交量占比在3-4.5内")
    elif limit_up_day_volume * 1.3 < volume_p1 <= limit_up_day_volume * 1.5:
        # 总交易数: 179 ，胜率: 44.69%，均盈: 6.27% | 均亏: 3.18% | 均持: 2.09，盈亏比: 1.97:1，期望: 1.044
        score -= 10
        reasons.append("(-10) 成交量占比在1.3-1.5内")

    if offset == 2:
        if limit_up_day_idx >= 35:
            # 获取不同周期的前期高点
            high_3d = df.iloc[limit_up_day_idx - 3: limit_up_day_idx]['high'].max()
            high_10d = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]['high'].max()
            high_35d = df.iloc[limit_up_day_idx - 35: limit_up_day_idx]['high'].max()
            price_value = high_p1
            if high_10d > price_value >= high_3d:
                # 总交易数: 101 ，胜率: 52.48%，均盈: 11.03% | 均亏: 3.07% | 均持: 2.45，盈亏比: 3.59:1，期望: 4.327
                # 总交易数: 479 ，胜率: 49.90%，均盈: 9.38% | 均亏: 3.68% | 均持: 2.47，盈亏比: 2.55:1，期望: 2.838
                score += 10
                reasons.append("(+10) T+1最高价价突破前3-10日高点")
            elif high_35d > price_value >= high_10d:
                # 总交易数: 174 ，胜率: 48.28 %，均盈: 6.37 % | 均亏: 3.11 % | 均持: 2.09，盈亏比: 2.05:1，期望: 1.468
                # 总交易数: 737 ，胜率: 40.84%，均盈: 5.86% | 均亏: 3.34% | 均持: 2.10，盈亏比: 1.75:1，期望: 0.416
                score -= 10
                reasons.append("(-10) T+1最高价突破前10-35日高点")

    if offset == 4:
        day_plus_2_data = df.iloc[limit_up_day_idx + 2]
        day_plus_3_data = df.iloc[limit_up_day_idx + 3]
        open_p2 = day_plus_2_data['open']
        high_p2 = day_plus_2_data['high']
        close_p2 = day_plus_2_data['close']
        open_p3 = day_plus_3_data['open']
        high_p3 = day_plus_3_data['high']
        close_p3 = day_plus_3_data['close']
        low_p3 = day_plus_3_data['low']
        volume_p3 = day_plus_3_data['volume']

        hugging_rate_0003 = 0.003
        hugging_rate_0005 = 0.005
        is_p1_hugging = (abs(open_p1 - limit_up_day_price) / limit_up_day_price < hugging_rate_0003) or \
                        (abs(close_p1 - limit_up_day_price) / limit_up_day_price < hugging_rate_0003)
        is_p2_hugging = (abs(open_p2 - limit_up_day_price) / limit_up_day_price < hugging_rate_0003) or \
                        (abs(close_p2 - limit_up_day_price) / limit_up_day_price < hugging_rate_0003)
        is_p3_hugging = (abs(open_p3 - limit_up_day_price) / limit_up_day_price < hugging_rate_0005) or \
                        (abs(close_p3 - limit_up_day_price) / limit_up_day_price < hugging_rate_0005)
        if is_p1_hugging or is_p2_hugging or is_p3_hugging:
            # 总交易数: 129，胜率: 52.71%，均盈: 12.86% | 均亏: 2.63% | 均持: 2.50，盈亏比: 4.88:1，期望: 5.531
            # 总交易数: 470，胜率: 40.64 %，均盈: 8.41 % | 均亏: 2.58 % | 均持: 2.37，盈亏比: 3.26:1，期望: 1.887
            score += 15
            reasons.append("(+15) 其开盘价或收盘价是否紧贴涨停价 (距离小于1%)")

        if limit_up_day_idx >= 55:
            # 获取不同周期的前期高点
            high_5d = df.iloc[limit_up_day_idx - 5: limit_up_day_idx]['high'].max()
            high_10d = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]['high'].max()
            high_17d = df.iloc[limit_up_day_idx - 17: limit_up_day_idx]['high'].max()
            high_33d = df.iloc[limit_up_day_idx - 33: limit_up_day_idx]['high'].max()
            high_55d = df.iloc[limit_up_day_idx - 55: limit_up_day_idx]['high'].max()
            price_value = max(high_p1, high_p2, high_p3)
            if high_55d > price_value >= high_33d:
                # 总交易数: 68  ，胜率: 38.24%，均盈: 6.58% | 均亏: 2.53% | 均持: 1.81，盈亏比: 2.60:1，期望: 0.955（四日总数480）
                # 总交易数: 266 ，胜率: 35.71%，均盈: 5.87% | 均亏: 2.37% | 均持: 2.11，盈亏比: 2.47:1，期望: 0.570（四日总数1967）
                score -= 10
                reasons.append("(-10) T+1到T+3最高价突破前33-55日高点")
            elif high_17d > price_value >= high_10d:
                # 总交易数: 40  ，胜率: 52.50%，均盈: 11.40% | 均亏: 1.47% | 均持: 2.67，盈亏比: 7.77:1，期望: 5.289
                # 总交易数: 189 ，胜率: 37.57%，均盈: 8.70% | 均亏: 2.18% | 均持: 2.66，盈亏比: 4.00:1，期望: 1.911
                score += 10
                reasons.append("(+10) T+1到T+3最高价突破前10-17日高点")
            elif high_5d > price_value:
                # 总交易数: 21  ，胜率: 52.38%，均盈: 12.22% | 均亏: 3.69% | 均持: 2.95，盈亏比: 3.31:1，期望: 4.643
                # 总交易数: 155 ，胜率: 57.42%，均盈: 7.97% | 均亏: 2.52% | 均持: 3.41，盈亏比: 3.16:1，期望: 3.505
                score += 10
                reasons.append("(+10) T+1到T+3最高价小于前5日高点")
    return score, reasons


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
    df['ma250'] = df['close'].rolling(250, min_periods=250).mean()

    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv

    # 2. 计算涨跌停价和是否触及
    limit_rate = config.MARKET_LIMIT_RATES[get_market_type(symbol)]
    df['prev_close'] = df['close'].shift(1)
    df['limit_price'] = (df['prev_close'] * (1 + limit_rate)).round(2)
    df['is_limit'] = df['close'] >= df['limit_price']
    df['down_limit_price'] = (df['prev_close'] * (1 - limit_rate)).round(2)
    df['is_limit_down'] = df['close'] <= df['down_limit_price']
    return df


def is_valid_first_limit_up_day(stock_info: StockInfo, df: pd.DataFrame, day: pd.Timestamp, config: StrategyConfig) -> \
        Optional[RuleEnum]:
    """
    检查给定的某一天是否是符合所有条件的首板涨停日。默认获得涨停后一日的数据
    :return: 如果通过所有检查, 返回True; 如果任何一个检查失败, 返回False。
    """
    code = stock_info.code
    name = stock_info.name
    theme = stock_info.theme
    market_value = stock_info.market_value
    day_idx = df.index.get_loc(day)

    if df.index.get_loc(day) + 1 >= len(df):
        return RuleEnum.NO_DATA_AFTER_LIMIT_UP  # 没有涨停后一天数据的直接排除
    if df.index.get_loc(day) < 20:
        return RuleEnum.INSUFFICIENT_DATA_BEFORE_LIMIT_UP  # 涨停前数据少于20的直接排除

    checker = ConditionChecker(df, limit_up_day_idx=day_idx, offset=0, stock_info=stock_info)

    limit_up_day_date = day  # 涨停日日期
    limit_up_day_idx = df.index.get_loc(limit_up_day_date)  # 涨停日行号
    limit_up_day_data = df.loc[limit_up_day_date]  # 涨停日数据
    limit_up_day_price = limit_up_day_data['close']  # 涨停日收盘价，重要支撑位
    open_p0 = limit_up_day_data['open']
    limit_up_day_high = limit_up_day_data['high']
    limit_up_day_low = limit_up_day_data['low']
    limit_up_day_volume = limit_up_day_data['volume']

    day_minus_1_idx = limit_up_day_idx - 1
    day_minus_1_day_date = df.index[day_minus_1_idx]
    day_minus_1_data = df.iloc[day_minus_1_idx]
    open_m1 = day_minus_1_data['open']
    high_m1 = day_minus_1_data['high']
    low_m1 = day_minus_1_data['low']
    close_m1 = day_minus_1_data['close']
    volume_m1 = day_minus_1_data['volume']
    day_minus_1_idx = limit_up_day_idx - 1
    day_minus_1_day_date = df.index[day_minus_1_idx]
    day_minus_1_data = df.iloc[limit_up_day_idx - 1]
    ma5_m1 = day_minus_1_data['ma5']
    ma10_m1 = day_minus_1_data['ma10']
    ma20_m1 = day_minus_1_data['ma20']
    ma30_m1 = day_minus_1_data['ma30']

    day_minus_2_idx = limit_up_day_idx - 2
    day_minus_2_day_date = df.index[day_minus_2_idx]
    day_minus_2_data = df.iloc[limit_up_day_idx - 2]
    open_m2 = day_minus_2_data['open']
    high_m2 = day_minus_2_data['high']
    low_m2 = day_minus_2_data['low']
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
    volume_p1 = day_plus_1_data['volume']
    is_red_candle_p1 = close_p1 >= open_p1

    # 条件0：排除市值大于250亿的股票
    if market_value > config.MAX_MARKET_CAP_BILLIONS:
        return RuleEnum.MARKET_CAP_TOO_HIGH

    # 条件1：排除特定题材
    if "证券" in name or "金融" in name or "证券" in theme or "金融" in theme or "外贸" in theme or "环境" in name or "环境" in theme:
        return RuleEnum.IS_FINANCE_OR_SECURITY

    # 条件2：排除前一日涨停和后一日涨停 (连板)
    if day_minus_1_data['is_limit'] or day_plus_1_data['is_limit']:
        return RuleEnum.IS_CONSECUTIVE_LIMIT_UP

    # 条件3：涨停后第一天量能过滤条件。副条件：并且第一天是阴线
    if (volume_p1 >= limit_up_day_volume * 3.6) and not is_red_candle_p1:
        return RuleEnum.P1_HUGE_VOLUME_GREEN_CANDLE

    # 条件4：排除10日内涨停超过4次的股票
    pre_window_10d = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]
    limit_up_count = (pre_window_10d['close'] >= pre_window_10d['limit_price']).sum()
    if limit_up_count >= 4:
        return RuleEnum.FREQUENT_LIMIT_UPS_RECENTLY

    # 条件5：前五日累计涨幅校验（相当于往前数五根k线，那天的收盘价到涨停当天收盘价的涨幅，也就是除涨停外，四天累计只能涨5%）
    pre5_start = df.index[df.index.get_loc(day) - 5]
    pre5_close = df.loc[pre5_start, 'close']
    if pre5_close != 0:
        total_change = (limit_up_day_price - pre5_close) / pre5_close * 100
        if total_change >= 15:
            return RuleEnum.LARGE_GAIN_BEFORE_LIMIT_UP

    # 条件6：涨停前的“高波动/长影线”混乱形态，K线实体长度占总振幅的比例极小（如小于30%），同时日内总振幅又比较大（如超过当天开盘价的5%）
    range_m1 = high_m1 - low_m1
    is_chaotic_m1 = False
    if range_m1 > 0 and (range_m1 / open_m1 > 0.05):
        if abs(open_m1 - close_m1) / range_m1 < 0.3:
            is_chaotic_m1 = True
    range_m2 = high_m2 - low_m2
    is_chaotic_m2 = False
    if range_m2 > 0 and (range_m2 / open_m2 > 0.05):
        if abs(open_m2 - close_m2) / range_m2 < 0.3:
            is_chaotic_m2 = True
    if (is_chaotic_m1 and is_chaotic_m2
            and not checker.is_10_m1m2m3_support_only_close()):
        return RuleEnum.PRE_LIMIT_UP_CHAOTIC_PATTERN

    # 条件9：排除4日内存在跌破一半的涨停
    pre_window_4d = df.iloc[limit_up_day_idx - 4: limit_up_day_idx]
    recent_limit_ups = pre_window_4d[pre_window_4d['close'] >= pre_window_4d['limit_price']]
    if not recent_limit_ups.empty:
        last_limit_up_day = recent_limit_ups.index[-1]
        price_floor = close_m1 * 1.035
        intermediate_days_loc = slice(df.index.get_loc(last_limit_up_day) + 1, limit_up_day_idx)
        intermediate_days = df.iloc[intermediate_days_loc]
        if not intermediate_days.empty:
            min_low_in_between = intermediate_days['low'].min()
            if min_low_in_between < price_floor:
                return RuleEnum.RECENT_LIMIT_UP_FAILURE

    # 条件12: 排除涨停前10日内，先出现至少2连板后又出现至少2连跌停的极端走势
    # 检查窗口内是否存在2连板,找出所有2连板结束的位置
    up_streaks = pre_window_10d['is_limit'].rolling(window=2).sum()
    up_streak_end_indices = up_streaks[up_streaks >= 2].index
    if not up_streak_end_indices.empty:
        for up_streak_end_date in up_streak_end_indices:
            up_streak_end_pos = pre_window_10d.index.get_loc(up_streak_end_date)
            search_for_down_streak_window = pre_window_10d.iloc[up_streak_end_pos + 1:]
            if len(search_for_down_streak_window) >= 2:
                down_streaks = search_for_down_streak_window['is_limit_down'].rolling(window=2).sum()
                if (down_streaks >= 2).any():
                    print(f"[{code}] 在 {day.date()} 排除：涨停前10日内出现先2连板后2连跌的极端走势。")
                    return RuleEnum.EXTREME_VOLATILITY_UP_DOWN

    # 条件13: 排除前期高位连板后，出现的缩量反抽形态，如果回调期最低点高于起涨的最低点，排除说明的确是在反抽，不过是缩量反抽，不需要参与
    limit_up_days = pre_window_10d[pre_window_10d['is_limit']]
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
                    return RuleEnum.LOW_VOLUME_REBOUND_AFTER_STREAK

    # 条件53：排除一字板，且排除最近20天内在20日线下超过一次的票，筛选出那些在涨停前趋势保持良好、没有经历深度或反复调整的股票。副条件：排除涨停前第二天低于五日线的票
    pre_window_20d = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
    breakdown_count = (pre_window_20d['close'] < pre_window_20d['ma20']).sum()
    is_m2_day_below_ma5 = close_m2 < day_minus_2_data['ma5']
    if limit_up_day_low == limit_up_day_price and breakdown_count > 1 and is_m2_day_below_ma5:
        print("排除首板是一字板的")
        return RuleEnum.ONE_WORD_LIMIT_UP_WITH_WEAK_TREND

    if limit_up_day_idx > 60:
        pre_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]

        # 条件3：前期密集峰/平台压力
        # 逻辑：如果前期存在一个由多个小高点组成的密集平台区，这个区域将构成一个强大的阻力带。我们用算法找出这些小高点，如果它们密集分布，就形成一个平台。
        peaks, _ = find_peaks(pre_window_60d['high'], prominence=pre_window_60d['high'].std() * 0.5, width=2)
        if len(peaks) >= 3:
            peak_highs = pre_window_60d.iloc[peaks]['high']
            price_spread = (peak_highs.max() - peak_highs.min()) / peak_highs.mean()
            # 如果这些高点的价格差异在5%以内，我们认为它们构成了一个密集平台
            if price_spread < 0.05:
                resistance_zone_top = peak_highs.max()
                # T+1的最高价进入了这个平台区域，但收盘价未能站稳在平台之上
                if (high_p1 >= resistance_zone_top * 0.98 and close_p1 < resistance_zone_top
                        and not checker.is_51020_m1_nian_he()):
                    # print(f"[{code}] T+2买入排除(条件3)：T+1受阻于前期平台 {resistance_zone_top:.2f}。")
                    return RuleEnum.REJECTED_BY_RESISTANCE_PLATFORM

        # 条件2：前期放量高点压力
        # 逻辑：前期成交量最大的那一天往往是多空分歧最激烈的地方，其最高价是重要心理关口。如果T+1日尝试突破这个高点但失败，且成交量未能超越当时的量，说明买方力量不足。
        peak_volume_day_idx_loc = pre_window_60d['volume'].argmax()
        initial_peak_day = pre_window_60d.iloc[peak_volume_day_idx_loc]
        effective_peak_high = initial_peak_day['high']
        effective_peak_volume = initial_peak_day['volume']
        start_loc = max(0, peak_volume_day_idx_loc - 2)
        end_loc = min(len(pre_window_60d), peak_volume_day_idx_loc + 3)
        search_window = pre_window_60d.iloc[start_loc:end_loc]
        for _, candidate_day in search_window.iterrows():
            is_high_volume = candidate_day['volume'] >= initial_peak_day['volume'] * 0.95
            is_higher_price = candidate_day['high'] > effective_peak_high * 1.02
            if is_high_volume and is_higher_price:
                effective_peak_high = candidate_day['high']
                effective_peak_volume = candidate_day['volume']
        if (high_p1 >= effective_peak_high and close_p1 < effective_peak_high and volume_p1 < effective_peak_volume
                and not checker.is_51020_m1_nian_he(0.02)
                and not checker.is_30_p0m1m2_support()):
            # print(f"[{code}] T+2买入排除(条件2-优化版)：T+1在放量区高点 {effective_peak_high:.2f} 处缩量回落。")
            return RuleEnum.REJECTED_BY_HIGH_VOLUME_PEAK

        # 条件13：价升量缩的“量价背离”压力
        if (limit_up_day_volume * 1.2 < volume_p1 <= limit_up_day_volume * 1.3
                and not checker.is_102030_m1_nian_he()
                and not checker.is_55120250_m1_nian_he()
                and not checker.is_51020_m1_nian_he(0.02)
                and not checker.is_60_m1m2m3_support()
                and not checker.is_stable_platform_to_new_low()):
            return RuleEnum.WEAK_FOLLOW_THROUGH_VOLUME

        # 条件11：“力不从心”——OBV顶背离
        peaks, _ = find_peaks(pre_window_60d['high'], prominence=pre_window_60d['high'].std())
        if len(peaks) > 0:
            # 取最近的前高点
            prev_peak_idx_in_window = peaks[-1]
            prev_peak_day = pre_window_60d.iloc[prev_peak_idx_in_window]
            prev_peak_price = prev_peak_day['high']
            obv_at_prev_peak = prev_peak_day['obv']
            is_close_abc = False
            price_a = limit_up_day_low
            if day_minus_1_data['close'] < day_minus_1_data['open']:
                price_b = day_minus_1_data['close']
            else:
                price_b = day_minus_1_data['low']
            if day_minus_2_data['close'] < day_minus_2_data['open']:
                price_c = day_minus_2_data['close']
            else:
                price_c = day_minus_2_data['low']
            price_list = [price_a, price_b, price_c]
            if all(p > 0 for p in price_list):
                max_price = max(price_list)
                min_price = min(price_list)
                avg_price = sum(price_list) / 3
                if avg_price > 0 and (max_price - min_price) / avg_price < 0.015:
                    is_close_abc = True
            # 检查T+1是否形成背离
            if (high_p1 > prev_peak_price and day_plus_1_data['obv'] < obv_at_prev_peak
                    and not is_close_abc
                    and not checker.is_55120250_m1_nian_he()
                    and not checker.is_10_m1_support()):
                return RuleEnum.OBV_BEARISH_DIVERGENCE

    if limit_up_day_idx >= 40:
        pre_window_40d = df.iloc[limit_up_day_idx - 40: limit_up_day_idx]
        pre_window_6d = df.iloc[limit_up_day_idx - 6: limit_up_day_idx]

        # 条件14: 排除过去40天里，振幅超过3.5%的天数少于3天的股票。副条件：第一天的量小于涨停日的2.5倍
        valid_prev_close = pre_window_40d['prev_close'].replace(0, np.nan).dropna()
        if not valid_prev_close.empty:
            daily_amplitude = (pre_window_40d['high'].loc[valid_prev_close.index] - pre_window_40d['low'].loc[
                valid_prev_close.index]) / valid_prev_close
            volatile_days_count = (daily_amplitude < 0.035).sum()
            if volatile_days_count > 37 and volume_p1 < limit_up_day_volume * 2.5:
                # print(f"[{code}] 在 {day.date()} 的涨停被排除：前{lookback_days_40}日波动过小({volatile_days_count}天振幅 > 3.5%)。")
                return RuleEnum.INSUFFICIENT_PRIOR_VOLATILITY

        # 条件15:如果涨停前6天的最高点也是涨停前40天里面的最高点。p1最高价高于最高点并且上影线超过4%，并且收盘价低于最高点。并且最高点的量大于涨停后第一天量能的1.2倍，排除
        peak_40d_price = pre_window_40d['high'].max()
        peak_6d_price = pre_window_6d['high'].max()
        # 条件A: 涨停前6天的最高点也是前40天的最高点
        if abs(peak_6d_price - peak_40d_price) < 1e-5:
            peak_day_loc = pre_window_40d['high'].idxmax()
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
                return RuleEnum.FALSE_BREAKOUT_OF_KEY_PEAK

    # 条件7：排除涨停和涨停后一天的最大量是前120天中最大量四倍以上的股票
    if limit_up_day_idx > 120:
        pre_window_120d = df.iloc[limit_up_day_idx - 120: limit_up_day_idx]
        prev_high_volume = df.loc[pre_window_120d['volume'].idxmax(), 'volume']
        max_volume = max(limit_up_day_volume, volume_p1)
        if max_volume > prev_high_volume * 4:
            # print("排除涨停和涨停后一天的交易量是前120天中最大交易量四倍以上的股票")
            return RuleEnum.ABNORMAL_VOLUME_SPIKE

    if close_m1 > 0 and limit_up_day_price > 0:
        # 规则A：T+0日跳空高开超过2%
        gap_up_on_limit_day = (open_p0 - close_m1) / close_m1 > 0.02
        # 规则B：T+1日低开超过2%
        gap_down_after_limit = (limit_up_day_price - open_p1) / limit_up_day_price > 0.02
        if (gap_up_on_limit_day and gap_down_after_limit
                and not checker.is_stable_platform_to_new_low()):
            return RuleEnum.ISLAND_REVERSAL_PATTERN
    return None  # 所有检查通过
    # return RuleEnum.ISLAND_REVERSAL_PATTERN


def is_valid_buy_opportunity(stock_info: StockInfo, df: pd.DataFrame, limit_up_day_idx: int, offset: int,
                             config: StrategyConfig) -> Optional[RuleEnum]:
    """
    检查从首板日到潜在买入日之间，是否满足所有的买入前置条件。
    """
    code = stock_info.code
    checker = ConditionChecker(df, limit_up_day_idx=limit_up_day_idx, offset=offset, stock_info=stock_info)

    limit_up_day_date = df.index[limit_up_day_idx]
    limit_up_day_data = df.loc[limit_up_day_date]  # 涨停日数据
    limit_up_day_price = limit_up_day_data['close']  # 涨停日收盘价，重要支撑位
    open_p0 = limit_up_day_data['open']
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
    low_m1 = day_minus_1_data['low']
    ma5_m1 = day_minus_1_data['ma5']
    ma10_m1 = day_minus_1_data['ma10']
    ma20_m1 = day_minus_1_data['ma20']
    ma30_m1 = day_minus_1_data['ma30']
    ma5_p0 = limit_up_day_data['ma5']
    ma10_p0 = limit_up_day_data['ma10']
    ma20_p0 = limit_up_day_data['ma20']
    ma30_p0 = limit_up_day_data['ma30']

    day_minus_2_idx = limit_up_day_idx - 2
    day_minus_2_day_date = df.index[day_minus_2_idx]
    day_minus_2_data = df.iloc[limit_up_day_idx - 2]
    open_m2 = day_minus_2_data['open']
    high_m2 = day_minus_2_data['high']
    close_m2 = day_minus_2_data['close']
    low_m2 = day_minus_2_data['low']
    volume_m2 = day_minus_2_data['volume']
    day_minus_3_idx = limit_up_day_idx - 3
    day_minus_3_day_date = df.index[day_minus_3_idx]
    day_minus_3_data = df.iloc[limit_up_day_idx - 3]
    low_m3 = day_minus_3_data['low']
    day_plus_1_idx = limit_up_day_idx + 1
    day_plus_1_day_date = df.index[day_plus_1_idx]
    day_plus_1_data = df.iloc[day_plus_1_idx]

    open_p1 = day_plus_1_data['open']
    high_p1 = day_plus_1_data['high']
    close_p1 = day_plus_1_data['close']
    low_p1 = day_plus_1_data['low']
    volume_p1 = day_plus_1_data['volume']
    ma10_p1 = day_plus_1_data['ma10']
    ma20_p1 = day_plus_1_data['ma20']
    ma30_p1 = day_plus_1_data['ma30']
    is_red_candle_p1 = close_p1 >= open_p1
    upper_body_p1 = max(open_p1, close_p1)
    upper_shadow_ratio_p1 = (high_p1 - upper_body_p1) / limit_up_day_price if limit_up_day_price > 0 else 0  # P1上影线比例

    potential_buy_day_idx = limit_up_day_idx + offset

    # 买前条件1: 排除买入前日收盘价>80的股票
    latest_close = df.iloc[potential_buy_day_idx - 1]['close']
    if latest_close > 80:
        return RuleEnum.PRICE_TOO_HIGH

    # 买前条件2: 排除在首板日和买入日之间，有新的涨停的和收盘价低于首板收盘价
    for i in range(1, offset):
        check_day_data = df.iloc[limit_up_day_idx + i]
        if check_day_data['is_limit']:
            return RuleEnum.NEW_LIMIT_UP_AFTER_FIRST
        if check_day_data['close'] < limit_up_day_price:
            if not config.USE_SINGLE_RULE:
                if i == 1:
                    # 提取T+1日所需数据
                    ma5_p1 = day_plus_1_data['ma5']
                    ma10_p1 = day_plus_1_data['ma10']

                    # 检查所需数据是否有效，避免计算错误
                    is_data_valid = all(pd.notna([low_p1, limit_up_day_low, ma10_p1, close_p1, ma5_p1])) and \
                                    limit_up_day_low > 0 and ma10_p1 > 0

                    if is_data_valid:
                        # 条件1: T+1最低价非常接近涨停日最低价 (说明是回踩，非破位下跌)
                        is_low_near_limit_low = abs(low_p1 - limit_up_day_low) / limit_up_day_low < 0.005
                        # 条件2: T+1最低价精准回踩10日线 (获得均线支撑)
                        is_low_on_ma10 = abs(low_p1 - ma10_p1) / ma10_p1 < 0.005
                        # 条件3: T+1收盘价依然站在5日线上方 (保持短期强势)
                        is_close_above_ma5 = close_p1 > ma5_p1
                        # 如果三个强支撑信号同时满足，则豁免此次“破位”，不立即排除，继续观察后续走势
                        if is_low_near_limit_low and is_low_on_ma10 and is_close_above_ma5:
                            return None  # 跳过本次循环的排除逻辑，进入下一次循环(检查T+2日等)
                return RuleEnum.PRICE_FELL_BELOW_LIMIT_UP_PRICE
            else:
                return RuleEnum.PRICE_FELL_BELOW_LIMIT_UP_PRICE

    # 第二日买入专有策略
    if offset == 2:
        # 买前条件0：低开比例
        gap_down_over_5pct = False
        limit_up_close = limit_up_day_data['close']
        if limit_up_close > 0:
            # 计算低开幅度
            if (limit_up_close - open_p1) / limit_up_close > 0.03:
                gap_down_over_5pct = True
        if gap_down_over_5pct and upper_shadow_ratio_p1 > 0.02:
            # print(f"[{code}] 在 {day.date()} 排除：涨停次日低开超5%且有长上影线。")
            cond_all_closes_above_ma120 = False
            window = df.iloc[limit_up_day_idx - 10: limit_up_day_idx]
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
            if (not cond_all_closes_above_ma120
                    and not checker.is_51020_m1_nian_he(0.02)
                    and not checker.is_20_p0m1m2_support()
                    and not checker.is_30_p0m1m2_support()):
                return RuleEnum.T1_GAP_DOWN_WEAK_REBOUND

        # 买前条件5：前高压制条件
        historical_high = df.iloc[limit_up_day_idx - 20:limit_up_day_idx]['high'].max()
        # 检查前3日最高价是否触及前高的99%
        recent_3day_high = df.iloc[limit_up_day_idx - 3:limit_up_day_idx]['high'].max()
        if (historical_high * 0.99 <= recent_3day_high < historical_high
                and not checker.is_51020_m1_nian_he(0.02)):
            return RuleEnum.STALLED_BELOW_PRIOR_HIGH

        # 买前条件11：涨停“撞线”前期关键高点
        if limit_up_day_idx > 90:
            hist_window_90d = df.iloc[limit_up_day_idx - 90: limit_up_day_idx]
            prev_major_high = hist_window_90d['high'].max()
            if (limit_up_day_price < prev_major_high and limit_up_day_price >= prev_major_high * 0.98
                    and not checker.is_51020_m1_nian_he()
                    and not checker.is_102030_m1_nian_he()
                    and not checker.is_55120250_m1_nian_he(0.012)):
                return RuleEnum.LIMIT_UP_HITS_MAJOR_RESISTANCE

        if limit_up_day_idx > 60:
            pre_window_60d = df.iloc[limit_up_day_idx - 60: limit_up_day_idx]

            # 买前条件26：中短期均线“趋势冲突” --- 好评,短期趋势向上，而中期趋势向下，重点
            if limit_up_day_idx > 65:
                ma10_p1 = day_plus_1_data['ma10']
                ma60_p1 = day_plus_1_data['ma60']
                # 获取5天前的均线值用于计算斜率方向
                ma10_m4 = df.iloc[limit_up_day_idx - 4]['ma10']
                ma60_m4 = df.iloc[limit_up_day_idx - 4]['ma60']
                if all(pd.notna([ma60_p1, ma10_m4, ma60_m4])):
                    if (ma10_p1 > ma10_m4 and ma60_p1 < ma60_m4
                            and not checker.is_51020_m1_nian_he(0.015)
                            and not checker.is_555_m1_abs_nian_he(0.005)
                            and not checker.is_55120_m1_nian_he()
                            and not checker.is_55250_m1_nian_he()
                            and not checker.is_120250_m1_nian_he()
                            and not checker.is_30_m1_support()
                            and not checker.is_30_m1m2m3_support_only_close()
                            and not checker.is_120_p0m1m2_support()):
                        return RuleEnum.TREND_CONFLICT

            # 买前条件68：“下降趋势线”的精准“狙击”
            peaks, _ = find_peaks(pre_window_60d['high'], prominence=pre_window_60d['high'].std() * 0.8,
                                  distance=5)
            if len(peaks) >= 2:
                # 取最近的两个波段高点
                peak_b_idx, peak_a_idx = peaks[-1], peaks[-2]
                peak_b, peak_a = pre_window_60d.iloc[peak_b_idx], pre_window_60d.iloc[peak_a_idx]
                # 确认是下降趋势
                if peak_b['high'] < peak_a['high']:
                    # 计算趋势线斜率和截距
                    x_coords = np.array([peak_a_idx, peak_b_idx])
                    y_coords = np.array([peak_a['high'], peak_b['high']])
                    slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
                    is_30_close_support = False
                    pre_30d_window = df.iloc[limit_up_day_idx - 30: limit_up_day_idx]
                    if not pre_30d_window.empty:
                        lowest_close_30d = pre_30d_window['close'].min()
                        close_m1 = day_minus_1_data['close']
                        if lowest_close_30d > 0:
                            distance_pct = (close_m1 - lowest_close_30d) / lowest_close_30d
                            if distance_pct < 0.01:
                                is_30_close_support = True
                    # 延伸趋势线到T+0当天
                    trendline_price_t0 = peak_b['high'] + slope * (len(pre_window_60d) - 1 - peak_b_idx)
                    if (abs(limit_up_day_high - trendline_price_t0) / trendline_price_t0 < 0.02
                            and not is_30_close_support
                            and not checker.is_51020_m1_nian_he(0.015)
                            and not checker.is_203055_m1_nian_he()
                            and not checker.is_120_p0m1m2_support()
                            and not checker.is_stable_platform_to_new_low()):
                        return RuleEnum.REJECTED_BY_DOWNTREND_LINE

        # 买入条件5: “高位滞涨”形态 — T+1日缩量/平量横盘收十字星
        is_doji_candle = abs(open_p1 - close_p1) / limit_up_day_price < 0.01
        is_not_breakthrough_volume = volume_p1 < limit_up_day_volume * 1.2
        is_stagnated = high_p1 < limit_up_day_high * 1.02
        if (is_doji_candle and is_not_breakthrough_volume and is_stagnated
                and not checker.is_51020_m1_nian_he()
                and not checker.is_102030_m1_nian_he(0.01)):
            return RuleEnum.STAGNATION_AT_PEAK

        # 买入条件6：“微型通道”加速后的力竭涨停
        is_climbing_m1 = close_m1 > open_m1
        is_climbing_m3 = day_minus_3_data['close'] > day_minus_3_data['open']
        if is_climbing_m1 and is_climbing_m3:
            # 检查最近两天是否明显脱离5日线，没有回踩
            ma5_m1 = day_minus_1_data['ma5']
            ma5_m2 = day_minus_2_data['ma5']
            is_detached_m1 = low_m1 > ma5_m1
            is_detached_m2 = low_m2 > ma5_m2
            if is_detached_m1 and is_detached_m2:
                return RuleEnum.PARABOLIC_CLIMB_EXHAUSTION

        # 买前条件10：“僵尸股”突兀涨停
        pre_window_20d = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
        daily_change_pct = (pre_window_20d['close'].pct_change(fill_method=None).abs())
        # 如果在过去20天里，有超过11天（50%）的日收盘价变动小于1%，则认为是极端沉寂
        stagnant_days_count = (daily_change_pct < 0.01).sum()
        if stagnant_days_count > 11 and not checker.is_10120_m1_nian_he() and not checker.is_stable_platform_to_new_low():
            return RuleEnum.BREAKOUT_FROM_EXTREME_STAGNATION

        # 买前条件11：涨停前20日内从最低点上涨超过35%
        pre_limit_up_window = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
        lowest_low_30d = pre_limit_up_window['close'].min()
        price_before_limit_up = close_m1
        if lowest_low_30d > 0:
            gain_from_low = (price_before_limit_up - lowest_low_30d) / lowest_low_30d
            if 0.4 > gain_from_low > 0.35:
                return RuleEnum.EXCESSIVE_GAIN_FROM_LOW_POINT

        # 新策略开头对齐位置---------------------------------------------------
    # 二日专有策略结束 --------------------------------------------------------

    # 第四日买入专有策略
    if offset == 4:
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

        # 在第四天时，排除已经在第二天买入的股票
        day_2_idx = limit_up_day_idx + 2
        if day_2_idx < len(df):
            day_2_timestamp = df.index[day_2_idx]
            day_2_data = df.iloc[day_2_idx]
            hypothetical_price_on_day_2 = simulate_ma5_order_prices(df, day_2_timestamp,
                                                                    config.PREDICT_PRICE_INCREASE_RATIO)
            if hypothetical_price_on_day_2 is not None:
                if day_2_data['low'] <= hypothetical_price_on_day_2:
                    return RuleEnum.PRIOR_BUY_OPPORTUNITY_TRIGGERED

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
            return RuleEnum.STRONG_T1_IMMEDIATE_FAILURE_T2

        if limit_up_day_idx >= 40:
            # 买前条件7: 排除涨停后三天的最高价低于40天内最高价且距离小于1%。副条件：p3冲高回落超过4%
            pre_window_40d_5d = df.iloc[limit_up_day_idx - 40: limit_up_day_idx - 5]
            prev_40d_high = pre_window_40d_5d['high'].max()
            max_high = max(high_p1, high_p2, high_p3)
            if max_high < prev_40d_high and (prev_40d_high - max_high) / prev_40d_high < 0.01:
                if upper_shadow_ratio_p3 > 0.04:
                    return RuleEnum.REJECTION_AT_KEY_RESISTANCE

            # 买前条件9： 排除p1,p2连续创40日新高但巨量回落的“双顶出货”形态
            pre_window_40d = df.iloc[limit_up_day_idx - 40: limit_up_day_idx]
            prev_40d_high = pre_window_40d['high'].max()
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
            rolling_vol_sum = pre_window_40d['volume'].rolling(window=2).sum()
            max_rolling_vol = rolling_vol_sum.max()
            condition_D_met = False
            if pd.notna(max_rolling_vol):
                current_vol_sum = volume_p1 + volume_p2
                condition_D_met = (current_vol_sum > max_rolling_vol * 0.95)
            if condition_A_met and condition_B_met and condition_C_met and condition_D_met:
                print(f"[{code}] 在 {df.index[limit_up_day_idx].date()} 后触发T+1,T+2双顶巨量回落形态，排除。")
                return RuleEnum.DOUBLE_TOP_DISTRIBUTION

        # 买前条件8：排除p1,p2,p3受到前期双头颈线压制
        for i in range(1, offset):
            check_day_idx = limit_up_day_idx + i
            if check_day_idx >= len(df):
                continue
            is_resisted, neckline = check_double_top_neckline_resistance(df, check_day_idx, config)
            if is_resisted:
                check_day_date = df.index[check_day_idx].date()
                print(f"[{code}] 在涨停后第{i}天({check_day_date})触及双头颈线 {neckline:.2f} 回落，放弃买入机会。")
                return RuleEnum.REJECTED_BY_DOUBLE_TOP_NECKLINE

        # 买前条件9：如果涨停后第一天最高价大于第三天大于第二天，并且都有长上影线，则排除
        is_t1_peak = day_plus_1_data['high'] > day_plus_3_data['high'] > day_plus_2_data['high']
        s_t1 = upper_shadow_ratio_p1 > 0.04
        s_t2 = upper_shadow_ratio_p2 > 0.03
        s_t3 = upper_shadow_ratio_p3 > 0.03
        both_have_long_shadows = s_t2 and s_t3 and s_t1
        if is_t1_peak and both_have_long_shadows:
            return RuleEnum.PERSISTENT_REJECTION_LOWERING_PEAKS

        # 买前条件10：排除整理期连续或单日大幅高开
        consecutive_gaps = open_p2 > (close_p1 * 1.02) and open_p3 > (close_p2 * 1.02)
        single_large_gap = open_p2 > (close_p1 * 1.03)
        if consecutive_gaps or single_large_gap:
            return RuleEnum.EXCESSIVE_GAP_UP_IN_CONSOLIDATION

        # 买前条件11：如果涨停后p2和p3都收阴线，并且p1大幅回落小于2.5%大于1%就排除
        if not is_red_candle_p2 and not is_red_candle_p3 and 0.025 > upper_shadow_ratio_p1 > 0.01:
            return RuleEnum.PERSISTENT_WEAKNESS_IN_CONSOLIDATION

        # 买前条件12：涨停后第一天被各条均线压制，并且第二天收盘价被均线压制，并且前三日不被30日线支撑,第二天是阳线的不排除
        long_term_mas = ['ma20', 'ma30', 'ma55', 'ma60']
        for ma_col in long_term_mas:
            ma_value = day_plus_1_data[ma_col]
            if pd.notna(ma_value) and ma_value > 0:
                tested_ma = high_p1 > ma_value
                is_rejected = (abs(close_p1 - ma_value) / ma_value) <= 0.02
                if (tested_ma and is_rejected and close_p2 < ma_value and (close_p2 < open_p2)
                        and not checker.is_30_p1p2p3_support()):
                    # print(f"[{code}] 在 {day_plus_1_data.name.date()} 触及 {ma_col} ({ma_value:.2f}) 后被压制，排除。")
                    return RuleEnum.REJECTED_BY_MAJOR_MA

        # 买前条件13：挑选出涨停后第一天跳空高开，尾盘不回落的票，排除第二天和第三天不跌第一天最低点的票，排除涨停前一天粘合的票
        strong_support_level = limit_up_day_price * 1.03
        t1_low = day_plus_1_data['low']
        # 条件A: T+1最低价 > 涨停价的1.05倍 (跳空高开且不回补)
        is_strong_gap_and_hold = t1_low >= strong_support_level
        # 条件B: T+2收盘价守住T+1的最低点
        t2_holds_support = day_plus_2_data['low'] >= t1_low
        # 条件C: T+3收盘价也守住T+1的最低点
        t3_holds_support = day_plus_3_data['low'] >= t1_low
        if (is_strong_gap_and_hold and not t2_holds_support and not t3_holds_support
                and not checker.is_51020_m1_nian_he(0.005)):
            return RuleEnum.GAP_UP_STRENGTH_INVALIDATED

        # 买前条件14：p1最高价大于55日线，p2和p3收盘价低于55日线。副条件：三天都不被20日线和30日线支撑
        ma55_t0 = limit_up_day_data['ma55']
        ma55_t1 = day_plus_1_data['ma55']
        ma55_t2 = day_plus_2_data['ma55']
        ma55_t3 = day_plus_3_data['ma55']
        # 确保所有需要用到的MA55值都存在且有效
        if pd.notna([ma55_t0, ma55_t1, ma55_t2, ma55_t3]).all():
            # T+1日收盘价 < T+1日ma55，T+2日收盘价 > T+2日ma55，T+3日收盘价 < T+3日ma55
            t1_above_ma55 = day_plus_1_data['high'] > ma55_t1
            t2_above_ma55 = day_plus_2_data['close'] < ma55_t2
            t3_below_ma55 = day_plus_3_data['close'] < ma55_t3
            if (t1_above_ma55 and t3_below_ma55 and t2_above_ma55
                    and not checker.is_20_p1p2p3_support()
                    and not checker.is_30_p1p2p3_support()):
                # print(f"[{code}] 触发MA55假突破回落形态，排除。")
                return RuleEnum.FAILED_MA55_BREAKOUT

        # 买前条件15：p1阴线，p2红柱，p3收盘价小于p2实体一半，且无强支撑时排除
        if not is_red_candle_p1 and is_red_candle_p2:
            midpoint_p1_body = (open_p1 + close_p1) / 2
            # 首先检查 T+2 是否未能强势收复 T+1 的一半，这是后续判断的前提
            if close_p2 <= midpoint_p1_body:
                # --- 强支撑豁免逻辑 ---
                # 1. 定义支撑区域和判断函数
                limit_up_close = limit_up_day_data['close']
                upper_bound = limit_up_close * 1.015

                def is_day_supported(day_data):
                    """辅助函数：检查单日是否被涨停价强力支撑"""
                    # 支撑条件1: 开盘价在支撑区内，并且当天收阳线
                    open_supported = limit_up_close < day_data['open'] <= upper_bound and day_data['close'] > day_data[
                        'open']
                    # 支撑条件2: 收盘价在支撑区内
                    close_supported = limit_up_close < day_data['close'] <= upper_bound
                    return open_supported or close_supported

                # 2. 计算T+1到T+3获得支撑的天数
                days_to_check = [day_plus_1_data, day_plus_2_data, day_plus_3_data]
                supported_days_count = sum(1 for day in days_to_check if is_day_supported(day))

                # 定义豁免条件：三天中至少有两天获得强支撑
                has_strong_support_override = supported_days_count >= 2

                # --- 最终排除逻辑 ---
                # 将多个弱势条件分别赋值给变量，提高可读性
                midpoint_p2_body = (open_p2 + close_p2) / 2
                is_weak_follow_through = close_p3 < midpoint_p2_body
                is_ma_diverged = not checker.is_51020_m1_nian_he()
                is_ma_unsupported = not checker.is_10_p1p2p3_support()
                # 最终决策：当所有“弱势条件”都满足，并且“没有”强支撑豁免时，才执行排除
                if is_weak_follow_through and is_ma_diverged and is_ma_unsupported and not has_strong_support_override:
                    # print(f"[{code}] 触发上涨乏力形态 (T+3跌破T+2阳线一半)，排除。")
                    # return None
                    return RuleEnum.FAILED_REBOUND_PATTERN

        # 买前条件16：涨停后三天的最高值差不多相等，p1p2存在上影线长度占K线总长度的20%以上
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
            low_support = low_p3 > low_p1 and close_p3 > close_p1
            is_bald_p3 = abs(close_p3 - high_p3) / high_p3 <= 0.01
            # 如果三天最高价相近，并且每天都有长上影线，则排除
            if (all_have_long_wicks and not low_support and not is_bald_p3
                    and not checker.is_51020_m1_nian_he(0.005)):
                return RuleEnum.FLAT_TOP_REJECTION

        # 买前条件17：如果是第四天买入，排除第三天在预计1.04时可能买入的股票，同时保留0.98时候可能买入的股票。结果再排除被20日线和30日线支撑的股票
        day_3_104_price = simulate_ma5_order_prices(df, day_plus_3_day_date, 1.04)
        day_3_098_price = simulate_ma5_order_prices(df, day_plus_3_day_date, 0.98)
        if day_3_104_price is not None and day_3_098_price is not None:
            if (day_3_098_price < low_p3 <= day_3_104_price
                    and not checker.is_555_m1_abs_nian_he()
                    and not checker.is_10120_m1_nian_he()
                    and not checker.is_20_p1p2p3_support_only_close()
                    and not checker.is_30_p1p2p3_support_only_close()):
                return RuleEnum.UNSUPPORTED_PULLBACK_ON_T3

        # 买前条件18：涨停后第一日高点高于涨停后第二日和第三日。涨停后第一日的低点高于涨停后第二日和第三日的低点。19年的数据还要优化，期望0.78
        if high_p2 < high_p1 and high_p3 < high_p1:
            if (low_p2 < low_p1 and low_p3 < low_p1
                    and not checker.is_51020_m1_nian_he()
                    and not checker.is_102030_m1_nian_he(0.012)
                    and not checker.is_stable_platform_to_new_low()):
                return RuleEnum.WEAK_PULLBACK_AFTER_T1_PEAK

        # 买前条件19：整理期持续阴跌且缺乏支撑
        is_yin_p1 = close_p1 < open_p1
        is_yin_p3 = close_p3 < open_p3
        limit_up_close = limit_up_day_data['close']
        upper_bound = limit_up_close * 1.015

        def is_day_supported(day_data):
            """辅助函数：检查单日是否被涨停价强力支撑"""
            # 支撑条件1: 开盘价在支撑区内，并且当天收阳线
            open_supported = limit_up_close < day_data['open'] <= upper_bound and day_data['close'] > day_data[
                'open']
            # 支撑条件2: 收盘价在支撑区内
            close_supported = limit_up_close < day_data['close'] <= upper_bound
            return open_supported or close_supported

        # 2. 计算T+1到T+3获得支撑的天数
        days_to_check = [day_plus_1_data, day_plus_2_data, day_plus_3_data]
        supported_days_count = sum(1 for day in days_to_check if is_day_supported(day))
        # 定义豁免条件：三天中至少有两天获得强支撑
        has_strong_support_override = supported_days_count >= 2
        if (is_yin_p1 and is_yin_p3
                and not has_strong_support_override
                and not checker.is_555_m1_abs_nian_he()
                and not checker.is_20_p0m1m2_support()
                and not checker.is_20_p1p2p3_support()
                and not checker.is_60_p0m1m2_support()
                and not checker.is_stable_platform()):
            # return None
            return RuleEnum.UNSUPPORTED_WEAK_CONSOLIDATION

        # 买前条件20 : 排除T+3日出现高波动长影十字星，显示多空激战/力竭
        p3_range = high_p3 - low_p3
        p3_body = abs(open_p3 - close_p3)
        if p3_range > 0:
            is_wide_range = p3_range / close_p2 > 0.05
            is_small_body = p3_body / p3_range < 0.2
            is_volume_not_shrinking = volume_p3 > (limit_up_day_volume * 0.8)
            if (is_wide_range and is_small_body and is_volume_not_shrinking
                    and not checker.is_51020_m1_nian_he()
                    and not checker.is_20_p0m1m2_support()
                    and not checker.is_30_p0m1m2_support()
                    and not checker.is_60_p0m1m2_support()
                    and not checker.is_stable_platform()):
                # print(f"[{code}] 在 {day_plus_3_day_date.date()} 出现高波动十字星，排除。")
                return RuleEnum.EXHAUSTION_CANDLE_ON_T3

        # 买前条件21：30日最低点在T+1日收阳线中
        if limit_up_day_idx >= 30:
            pre_limit_up_window = df.iloc[limit_up_day_idx - 30: limit_up_day_idx]
            lowest_low_30d = pre_limit_up_window['close'].min()
            price_before_limit_up = close_m1
            if lowest_low_30d > 0:
                gain_from_low = (price_before_limit_up - lowest_low_30d) / lowest_low_30d
                if (0.7 > gain_from_low > 0.3
                        and not checker.is_120_p0m1m2_support()
                        and not checker.is_10_p1p2p3_support()
                        and not checker.is_120250_m1_nian_he()
                        and not checker.is_55120_m1_nian_he()):
                    return RuleEnum.EXCESSIVE_GAIN_FROM_LOW_POINT

        upper_shadow_p2 = high_p2 - close_p2
        candle_body_p2 = abs(open_p2 - close_p2)
        p2_is_significantly_higher = (high_p2 - close_p2) / close_p1 > 0.02
        is_body_relatively_small = candle_body_p2 < 2 * upper_shadow_p2
        highs_are_similar = abs(high_p2 - high_p3) / high_p2 < 0.008
        # 条件B: T+2和T+3的收盘价非常接近 (<1%的差异)
        closes_are_similar = abs(close_p2 - close_p3) / close_p2 < 0.01
        # 条件C: T+2的最高价略高于T+3
        p2_high_is_slightly_higher = high_p2 > high_p3
        # 如果三个条件同时满足，则构成一个微弱的下降结构，可能是上涨乏力的信号，应予以排除
        if highs_are_similar and closes_are_similar and p2_high_is_slightly_higher and p2_is_significantly_higher and is_body_relatively_small:
            # print(f"[{code}] 在 {day_plus_3_day_date.date()} 触发T+2, T+3高位滞涨形态，排除。") # 这是一条可选的调试信息
            return RuleEnum.STAGNATION_WITH_SLIGHTLY_LOWER_HIGH
        # 新策略开头对齐位置---------------------------------------------------
    # 四日专有策略结束 --------------------------------------------------------

    pre_window_20d = df.iloc[limit_up_day_idx - 20: limit_up_day_idx]
    four_day_limit_up_streak = pre_window_20d['is_limit'].rolling(window=4).sum()
    streak_end_dates = four_day_limit_up_streak[four_day_limit_up_streak >= 4].index

    # 步骤2: 如果找到了连板行情，则进入统一的后续检查逻辑
    if not streak_end_dates.empty:
        last_streak_end_date = streak_end_dates.max()
        # --- 分支判断 1: “中位线压力”检查 ---
        # a. 定位连板期间的最高点
        last_streak_end_idx_loc = pre_window_20d.index.get_loc(last_streak_end_date)
        streak_start_idx_loc = last_streak_end_idx_loc
        while streak_start_idx_loc >= 0 and pre_window_20d.iloc[streak_start_idx_loc]['is_limit']:
            streak_start_idx_loc -= 1
        streak_start_idx_loc += 1
        streak_days_df = pre_window_20d.iloc[streak_start_idx_loc: last_streak_end_idx_loc + 1]
        streak_peak_high = streak_days_df['high'].max()
        streak_peak_date = streak_days_df['high'].idxmax()
        streak_peak_idx_global = df.index.get_loc(streak_peak_date)
        # b. 寻找后续调整的最低点并计算中位线
        search_window_for_low = df.iloc[streak_peak_idx_global: limit_up_day_idx]
        if not search_window_for_low.empty:
            subsequent_trough_low = search_window_for_low['low'].min()
            midpoint_price = (streak_peak_high + subsequent_trough_low) / 2
            # c. 判断T+1最高价是否受阻
            if midpoint_price > 0:
                if abs(high_p1 - midpoint_price) / midpoint_price < 0.02:
                    return RuleEnum.REJECTED_BY_STREAK_MIDPOINT_V2  # 排除：受阻于中位线
        # --- 分支判断 2: “假突破20日线”检查 ---
        # a. 定义连板结束后的窗口
        search_start_idx_global = df.index.get_loc(last_streak_end_date) + 1
        if search_start_idx_global < limit_up_day_idx:
            after_streak_window = df.iloc[search_start_idx_global: limit_up_day_idx]
            # b. 检查此窗口内是否“没有”3连跌停
            three_day_limit_down_streak = after_streak_window['is_limit_down'].rolling(window=3).sum()
            if not (
                    three_day_limit_down_streak >= 3).any() and high_p1 > ma20_p1 and close_p1 < ma20_p1 and is_red_candle_p1:
                return RuleEnum.POST_STREAK_FAKE_BREAKTHROUGH

    if limit_up_day_idx >= 25:
        window = df.iloc[limit_up_day_idx - 25: limit_up_day_idx]
        window_50 = df.iloc[limit_up_day_idx - 50: limit_up_day_idx]
        peak_idx = window['high'].idxmax()
        peak_high = window.loc[peak_idx, 'high']
        # 2. 在“最高点”到“涨停日”的区间内寻找最低点 (Trough)
        trough_search_window = window.loc[peak_idx:]
        trough_idx = trough_search_window['low'].idxmin()
        trough_low = trough_search_window.loc[trough_idx, 'low']
        peak_loc = window.index.get_loc(peak_idx)
        trough_loc = window.index.get_loc(trough_idx)
        trading_days_apart = trough_loc - peak_loc

        has_large_rebound = False
        # 定义下跌区间（从最高点之后一天到最低点当天）
        decline_start_loc = peak_loc + 1
        decline_end_loc = trough_loc + 1  # 切片不包含末尾，所以+1以包含最低点当天
        if decline_start_loc < decline_end_loc:
            decline_window = window.iloc[decline_start_loc:decline_end_loc]
            # 计算区间内每日的涨幅
            daily_change = (decline_window['close'] - decline_window['prev_close']) / decline_window['prev_close']
            # 检查是否有任何一天的涨幅超过5.5%
            if (daily_change > 0.055).any():
                has_large_rebound = True

        is_trough_significant = False
        pre_peak_start_loc = peak_loc - trading_days_apart
        # 确保历史数据足够长，可以进行对比
        pre_peak_window = window_50.iloc[pre_peak_start_loc + 25:peak_loc + 25]
        # 计算在“顶”之前，有多少天的价格比我们找到的“底”还低
        lower_low_count = (pre_peak_window['low'] < trough_low).sum()
        if lower_low_count > 3:
            is_trough_significant = True
        # 3. 验证形态: 确保从高点到低点有显著的跌幅
        if peak_high > 0 and (peak_high - trough_low) / peak_high > 0.05:
            midpoint = (peak_high + trough_low) / 2
            # 条件：T+0收盘价 < 中位线 < T+1最高价，表示反弹区间“包含”了中位线
            if (limit_up_day_price < midpoint and high_p1 > midpoint
                    and trading_days_apart > 10 and not has_large_rebound and is_trough_significant
                    and not checker.is_102030_m1_nian_he(0.012)
                    and not checker.is_120_p0m1m2_support()
            ):
                return RuleEnum.REBOUND_TO_MIDPOINT_RESISTANCE

    if config.USE_SINGLE_RULE:
        return RuleEnum.WEAK_PULLBACK_AFTER_T1_PEAK
    else:
        return None  # 所有检查通过


def second_chance_check(df: pd.DataFrame, limit_up_day_idx: int, offset: int, code: str,
                        config: StrategyConfig) -> bool:
    """
    二次机会检查的公有方法。
    如果常规检查失败，则进入此函数进行最终裁决。
    :return: 如果满足强力看涨条件，返回True，即可“复活”该股；否则返回False。
    """

    return False


def find_first_limit_up(stock_info: StockInfo, df, config: StrategyConfig, rejection_log):
    """识别首板涨停日并排除连板"""
    limit_days = df[df['is_limit']].index.tolist()
    code = stock_info.code
    name = stock_info.name
    valid_days = []
    for day in limit_days:
        # 日期过滤条件（方便回测）
        if day < pd.Timestamp('2024-03-01') and not config.USE_2019_DATA:
            continue

        rejection_rule = is_valid_first_limit_up_day(stock_info, df, day, config)

        if rejection_rule:
            rejection_log.append({
                '股票代码': code, '股票名称': name, '日期': day.strftime('%Y-%m-%d'),
                '阶段': '首板筛选', '规则名称': rejection_rule.name,
                '规则描述': rejection_rule.description, '规则类型': rejection_rule.rule_type,
                '排除原因': rejection_rule.message,
            })
        else:
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


def generate_signals(stock_info: StockInfo, df, first_limit_day, config: StrategyConfig, rejection_log,
                     use_optimized_sell_logic: bool = False):
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

    def _create_record_dict(buy_data, sell_day, sell_price, sell_reason_str, hold_days_val, actual_price, score,
                            rejection_rule=None):
        # 确保价格非空，避免计算错误
        actual_price = actual_price if actual_price is not None else 0
        sell_price = sell_price if sell_price is not None else 0
        # 计算收益率
        profit_rate = ((sell_price - actual_price) / actual_price * 100) if actual_price > 0 else 0
        record = {
            '股票代码': stock_info.code,
            '股票名称': stock_info.name,
            '首板日': first_limit_day.strftime('%Y-%m-%d'),
            '买入日': buy_data.name.strftime('%Y-%m-%d'),
            '卖出日': sell_day.strftime('%Y-%m-%d'),
            '评分': score,
            '涨停后天数': offset,
            '持有天数': hold_days_val,
            '买入价': round(actual_price, 2),
            '卖出价': round(sell_price, 2),
            '触碰类型': 'MA5支撑反弹' if buy_data['close'] > buy_data['ma5'] else 'MA5破位回升',
            '收益率(%)': round(profit_rate, 2),
            '涨停后第二日涨幅(%)': round(next_day_2_pct, 2) if next_day_2_pct is not None else None,
            '卖出原因': sell_reason_str,
        }
        # 如果是排除记录，则附加规则详情
        if rejection_rule:
            record.update({
                '阶段': f'T+{(buy_data.name - first_limit_timestamp).days}买入筛选',
                '规则名称': rejection_rule.name,
                '规则描述': rejection_rule.description,
                '规则类型': rejection_rule.rule_type,
            })
        return record

    # for offset in range(4, 5):  # 检查涨停后第2或4天
    # for offset in [2, 4]:  # 检查涨停后第2和4天
    for offset in config.BUY_OFFSETS:
        if start_idx + offset >= len(df):
            break

        current_day_idx = start_idx + offset
        current_day = df.index[start_idx + offset]
        current_data = df.iloc[start_idx + offset]

        price_ma5 = simulate_ma5_order_prices(df, current_day, config.PREDICT_PRICE_INCREASE_RATIO)
        if price_ma5 is None: continue

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

        rejection_rule = is_valid_buy_opportunity(stock_info, df, start_idx, offset, config)

        score, reasons = calculate_quality_score(stock_info, df, start_idx, offset, config)

        if score < 10:
            rejection_rule = RuleEnum.SCORE_IS_SMALL_TEN

        # 持有期卖出逻辑
        hold_days = 0
        final_sell_day = None
        final_sell_price = 0
        final_reason = '持有中'
        first_limit_up_price = df.loc[first_limit_day, 'close']

        for sell_offset in range(1, 20):
            sell_day_idx = start_idx + offset + sell_offset

            if sell_day_idx < len(df):
                sell_day = df.index[start_idx + offset + sell_offset]
                current_sell_data = df.loc[sell_day]
                prev_sell_data = df.iloc[sell_day_idx - 1]
                hold_days += 1

                position_info = {'hold_days': hold_days, 'first_limit_up_price': first_limit_up_price}
                market_data = MarketDataContext(
                    high=current_sell_data['high'],
                    low=current_sell_data['low'],
                    open=current_sell_data['open'],
                    close=current_sell_data['close'],
                    ma5=current_sell_data['ma5'],
                    up_limit_price=current_sell_data['limit_price'],
                    down_limit_price=current_sell_data['down_limit_price'],
                    prev_close=prev_sell_data['close'],
                    prev_up_limit_price=prev_sell_data['limit_price'],
                    prev_down_limit_price=prev_sell_data['down_limit_price'],
                    today_df=df.iloc[:sell_day_idx + 1],
                )
                should_sell, reason = get_sell_decision(stock_info, position_info, market_data,
                                                        use_optimized_logic=use_optimized_sell_logic)
                if should_sell:
                    final_sell_day = sell_day
                    final_sell_price = current_sell_data['close']
                    final_reason = reason
                    break
            else:
                # 持有到回测结束
                last_day_idx = len(df) - 1
                final_sell_day = df.index[last_day_idx]
                final_sell_price = df.iloc[last_day_idx]['close']
                if hold_days == sell_offset - 1:
                    hold_days += 1
                break

        # 如果循环结束都没有卖点，确保有最后的卖出信息
        if final_sell_day is None and hold_days > 0:
            last_day_idx = len(df) - 1
            final_sell_day = df.index[last_day_idx]
            final_sell_price = df.iloc[last_day_idx]['close']

        # --- 第四步：根据判断结果，将完整记录存入对应的列表 ---
        if rejection_rule:
            # 如果被排除，创建一条完整的排除记录
            rejection_record = _create_record_dict(
                current_data, final_sell_day, final_sell_price,
                final_reason, hold_days, actual_price, score,
                rejection_rule=rejection_rule
            )
            rejection_log.append(rejection_record)
        else:
            # 如果未被排除，创建一条成功的交易信号
            signal = _create_record_dict(
                current_data, final_sell_day, final_sell_price,
                final_reason, hold_days, actual_price, score
            )
            signals.append(signal)
            # 找到一个买点后就终止对该首板信号的后续检查
            break

    return signals


def create_daily_holdings(result_df):
    """
    创建每日持仓情况的DataFrame。
    新版格式：数量 | 日期 | 具体的股票（名称），并按日期降序排列。
    """
    # 确保有交易数据，否则返回一个空表
    if result_df.empty:
        return pd.DataFrame(columns=['数量', '日期', '具体的股票'])
    # 转换日期格式
    result_df['买入日'] = pd.to_datetime(result_df['买入日'])
    result_df['卖出日'] = pd.to_datetime(result_df['卖出日'])
    # 生成日期范围
    min_date = result_df['买入日'].min()
    max_date = result_df['卖出日'].max()
    date_range = pd.date_range(min_date, max_date)
    # 创建一个列表来存放每日的持仓记录
    holdings_records = []
    for date in date_range:
        # 筛选出在当前日期仍然持有的股票
        # 条件1：卖出日在今天之后，说明今天肯定持有
        sold_after_today = (result_df['卖出日'] > date)

        # 条件2：卖出日就是今天，但原因是“持有中”，说明今天收盘时也应算作持有
        sold_today_but_holding_to_end = (
                (result_df['卖出日'] == date) &
                (result_df['卖出原因'] == '持有中')
        )
        # 筛选出在当前日期仍然持有的股票（满足以上任一条件即可）
        held_stocks_df = result_df[
            (result_df['买入日'] <= date) &
            (sold_after_today | sold_today_but_holding_to_end)
            ]

        # 如果当天有持仓
        if not held_stocks_df.empty:
            # 获取持仓股票的【名称】列表
            held_stock_names = held_stocks_df['股票名称'].tolist()
            # 创建当天的持仓记录字典
            record = {
                '日期': date.strftime('%Y-%m-%d'),
                '数量': len(held_stock_names),
                '具体的股票': ', '.join(held_stock_names)  # 使用逗号和空格连接，更美观
            }
            holdings_records.append(record)
    # 如果整个回测期间都没有任何一天有持仓，则返回空表
    if not holdings_records:
        return pd.DataFrame(columns=['数量', '日期', '具体的股票'])
    # 将记录列表转换为DataFrame
    holdings_df = pd.DataFrame(holdings_records)
    # 按照要求的列顺序重新排列
    holdings_df = holdings_df[['数量', '日期', '具体的股票']]
    # 按照日期降序排序
    holdings_df = holdings_df.sort_values(by='日期', ascending=False)
    # 重置索引，让序号从0开始
    holdings_df = holdings_df.reset_index(drop=True)
    return holdings_df


def save_trades_excel(result_df, rejections_df):
    """专业级Excel导出函数"""
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    excel_name = f"首板交易记录_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        workbook = writer.book
        # --- 预先定义通用格式 ---
        format_green = workbook.add_format({'font_color': '#00B050', 'num_format': '0.00%'})
        format_red = workbook.add_format({'font_color': '#FF0000', 'num_format': '0.00%'})

        # --- 写入交易明细 ---
        if not result_df.empty:
            column_order = ['股票代码', '股票名称', '首板日', '买入日', '卖出日', '评分', '涨停后天数',
                            '持有天数', '买入价', '卖出价', '涨停后第二日涨幅(%)', '收益率(%)', '卖出原因']
            result_df = result_df[column_order]
            # 按买入日降序排序
            result_df = result_df.sort_values(by='买入日', ascending=False)
            result_df.to_excel(writer, sheet_name='交易明细', index=False)
            # 获取工作表对象
            workbook = writer.book
            worksheet = writer.sheets['交易明细']

        if not rejections_df.empty:
            # --- 核心修改：在处理前，先筛选掉“首板筛选”阶段的记录 ---
            rejections_df = rejections_df[rejections_df['阶段'] != '首板筛选'].copy()
            if not rejections_df.empty:
                rejection_columns = [
                    '股票代码', '股票名称', '首板日', '买入日', '卖出日', '评分',
                    '持有天数', '买入价', '卖出价', '收益率(%)', '卖出原因',
                    '阶段', '规则名称', '规则描述', '规则类型'
                ]
                rejections_df = rejections_df.sort_values(by='首板日', ascending=False)
                rejections_df_display = rejections_df.reindex(columns=rejection_columns)
                rejections_df_display.to_excel(writer, sheet_name='排除记录', index=False)

                rejection_sheet = writer.sheets['排除记录']
                for idx, col in enumerate(rejections_df_display.columns):

                    # 处理可能完全为空的列
                    if col in rejections_df_display:
                        max_len = max(rejections_df_display[col].astype(str).map(len).max(), len(col)) + 2
                        rejection_sheet.set_column(idx, idx, max_len)

                    # --- 核心新增：为排除记录表添加收益率红绿颜色格式 ---
                if '收益率(%)' in rejections_df_display.columns:
                    profit_col_idx_rej = rejections_df_display.columns.get_loc('收益率(%)')
                    for row_num, profit in enumerate(rejections_df_display['收益率(%)']):
                        if pd.notna(profit):
                            cell_value = profit / 100
                            if cell_value >= 0:
                                rejection_sheet.write(row_num + 1, profit_col_idx_rej, cell_value, format_green)
                            else:
                                rejection_sheet.write(row_num + 1, profit_col_idx_rej, cell_value, format_red)

                    # 格式化：冻结首行和筛选
                rejection_sheet.freeze_panes(1, 0)
                # 1. 列宽自适应
                for idx, col in enumerate(result_df.columns):
                    max_len = max(result_df[col].astype(str).map(len).max(), len(col)) + 2
                    if col == '股票名称':
                        max_len += 3
                    worksheet.set_column(idx, idx, max_len)
                rejection_sheet.autofilter(0, 0, len(rejections_df_display), len(rejections_df_display.columns) - 1)

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


def process_single_stock(stock_info_task):
    """
    处理单只股票的回测逻辑，用于并行计算。
    """
    code, name, query_tool, config, use_optimized_sell_logic = stock_info_task  # 接收包含所有必需信息的元组
    rejection_log = []

    df, _ = get_stock_data(code, config)
    if df is None or df.empty:
        return []  # 如果没有数据，返回空列表
    df = prepare_data(df, code, config)
    stock_info = StockInfo(
        code=code,
        name=name,
        market_value=query_tool.get_stock_market_value(code),
        theme=query_tool.get_theme_by_code(code),
    )
    first_limit_days = find_first_limit_up(stock_info, df, config, rejection_log)

    all_signals = []
    for day in first_limit_days:
        signals = generate_signals(stock_info, df, day, config, rejection_log, use_optimized_sell_logic)
        all_signals.extend(signals)

    return all_signals, rejection_log


def run_backtest(config, query_tool, stock_list, use_optimized_sell_logic=False):
    """
    封装的回测执行函数。
    :param use_optimized_sell_logic: 是否启用优化后的卖出逻辑。
    :return: 包含所有交易信号的DataFrame。
    """
    print(f"\n--- 开始回测 (优化逻辑: {'开启' if use_optimized_sell_logic else '关闭'}) ---")
    all_signals = []

    # 注意：这里我们忽略了 rejections，因为对比时我们只关心成功交易的差异
    all_rejections = []

    tasks = [(code, name, query_tool, config, use_optimized_sell_logic) for code, name in stock_list]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_stock, tasks))

    for signals, rejections in results:
        all_signals.extend(signals)
        all_rejections.extend(rejections)

    return pd.DataFrame(all_signals)


class ComparisonRunner:
    """
    一个用于运行和比较不同策略版本回测结果的类。
    """

    def __init__(self, config, query_tool, stock_list):
        self.config = config
        self.query_tool = query_tool
        self.stock_list = stock_list
        self.original_trades = None
        self.optimized_trades = None

    def run(self):
        """执行原始版本和优化版本的回测。"""
        self.original_trades = run_backtest(self.config, self.query_tool, self.stock_list,
                                            use_optimized_sell_logic=False)
        self.optimized_trades = run_backtest(self.config, self.query_tool, self.stock_list,
                                             use_optimized_sell_logic=True)

    def _create_trade_key(self, row):
        """为每一笔交易创建一个唯一的标识符。"""
        return f"{row['股票代码']}_{row['首板日']}_{row['买入日']}"

    def compare_and_display(self):
        """比较两次回测的结果，并只显示有差异的交易。"""
        if self.original_trades is None or self.optimized_trades is None:
            print("请先运行 .run() 方法进行回测。")
            return

        print("\n\033[1m\033[94m=== 卖出逻辑优化效果对比 ===\033[0m")

        # 使用唯一key创建字典以便快速查找
        orig_dict = {self._create_trade_key(row): row for _, row in self.original_trades.iterrows()}
        opt_dict = {self._create_trade_key(row): row for _, row in self.optimized_trades.iterrows()}

        affected_trades = []

        # 遍历优化后的结果，与原始结果进行对比
        for key, opt_trade in opt_dict.items():
            orig_trade = orig_dict.get(key)
            if orig_trade is None:
                # 这种情况理论上不应该发生，除非优化逻辑改变了买入行为
                continue

            # 核心对比逻辑：如果卖出日、卖出原因或收益率不同，则认为是受影响的交易
            if (orig_trade['卖出日'] != opt_trade['卖出日'] or
                    orig_trade['卖出原因'] != opt_trade['卖出原因'] or
                    orig_trade['收益率(%)'] != opt_trade['收益率(%)']):
                affected_trades.append({
                    '股票': f"{opt_trade['股票名称']}({opt_trade['股票代码']})",
                    '买入日': opt_trade['买入日'],
                    '原卖出': orig_trade['卖出日'],
                    '优卖出': opt_trade['卖出日'],
                    '原卖出因': orig_trade['卖出原因'],
                    '优卖出因': opt_trade['卖出原因'],
                    '原收益(%)': orig_trade['收益率(%)'],
                    '优收益(%)': opt_trade['收益率(%)'],
                })

        if not affected_trades:
            print("\033[92m本次逻辑修改未对任何交易的卖出时点产生影响。\033[0m")
            return

        # 使用pandas来美观地展示结果
        comparison_df = pd.DataFrame(affected_trades)

        comparison_df['买入日'] = pd.to_datetime(comparison_df['买入日'])
        # 按“买入日”降序排列
        comparison_df = comparison_df.sort_values(by='买入日', ascending=False)

        # 计算收益变化
        comparison_df['收益变化(%)'] = comparison_df['优收益(%)'] - comparison_df['原收益(%)']

        # 打印结果
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)
        print(f"\033[93m共找到 {len(comparison_df)} 笔受本次优化影响的交易：\033[0m")
        print(comparison_df.to_string(index=False))

        # 打印汇总统计
        total_change = comparison_df['收益变化(%)'].sum()
        avg_change = comparison_df['收益变化(%)'].mean()
        positive_impacts = (comparison_df['收益变化(%)'] > 0).sum()
        negative_impacts = (comparison_df['收益变化(%)'] < 0).sum()

        print("\n\033[1m--- 影响汇总 ---\033[0m")
        print(f"对受影响交易的平均收益影响: {avg_change:+.2f}%")
        print(f"正向:负向  {positive_impacts}:{negative_impacts}")
        print(f"优化对总收益率的净影响: {total_change:+.2f}%")


if __name__ == '__main__':
    total_start = time.perf_counter()  # 记录程序开始时间

    config = StrategyConfig()
    query_tool = getAllStockCsv.StockQuery()
    # 加载股票列表并过滤
    stock_list = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values

    if config.USE_SELL_LOGIC:
        # 新流程可以对比卖出条件的效果
        runner = ComparisonRunner(config, query_tool, stock_list)
        runner.run()
        runner.compare_and_display()
    else:
        # 下面是老流程，主要是回测买入时的排除条件
        all_signals = []
        all_rejections = []
        stock_process_start = time.perf_counter()
        tasks = [(code, name, query_tool, config, False) for code, name in stock_list]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # executor.map 会自动将tasks列表中的每一项分配给一个进程，并收集结果
            results = list(executor.map(process_single_stock, tasks))

        for signals, rejections in results:
            all_signals.extend(signals)
            all_rejections.extend(rejections)

        stock_process_duration = time.perf_counter() - stock_process_start

        # 生成统计报表
        result_df = pd.DataFrame(all_signals)
        rejections_df = pd.DataFrame(all_rejections)
        if not result_df.empty or not rejections_df.empty:
            if not result_df.empty:
                # 把原有的买入日从年月日类型转为年月日时分秒，确保买入日为日期类型
                result_df['买入日'] = pd.to_datetime(result_df['买入日'])
                # 获取最近三个月的截止日期（含当月）
                current_date = datetime.now()
                months_num = 27 if config.USE_2019_DATA else 17
                three_months_ago = current_date - pd.DateOffset(months=months_num)

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

            #     print(f"\n\033[1m--- 评分表现汇总 ---\033[0m")
            #
            #     score_groups = {
            #         "评分 >= 70": result_df[result_df['评分'] >= 70],
            #         "60 <= 评分 < 70": result_df[(result_df['评分'] >= 60) & (result_df['评分'] < 70)],
            #         "50 <= 评分 < 60": result_df[(result_df['评分'] >= 50) & (result_df['评分'] < 60)],
            #         "40 <= 评分 < 50": result_df[(result_df['评分'] >= 40) & (result_df['评分'] < 50)],
            #         "30 <= 评分 < 40": result_df[(result_df['评分'] >= 30) & (result_df['评分'] < 40)],
            #         "20 <= 评分 < 30": result_df[(result_df['评分'] >= 20) & (result_df['评分'] < 30)],
            #         "10 <= 评分 < 20": result_df[(result_df['评分'] >= 10) & (result_df['评分'] < 20)],
            #         "0 <= 评分 < 10": result_df[(result_df['评分'] >= 0) & (result_df['评分'] < 10)],
            #         "-10 <= 评分 < 0": result_df[(result_df['评分'] >= -10) & (result_df['评分'] < 0)],
            #         "评分 < -10": result_df[result_df['评分'] < -10]
            #     }
            #
            #     for label, df_group in score_groups.items():
            #         if len(df_group) == 0:
            #             print(f"{label:<18}: 无交易记录")
            #             continue
            #
            #         win_rate_group = len(df_group[df_group['收益率(%)'] > 0]) / len(df_group) * 100
            #         avg_win_group = df_group[df_group['收益率(%)'] > 0]['收益率(%)'].mean()
            #         avg_loss_group = abs(df_group[df_group['收益率(%)'] <= 0]['收益率(%)'].mean())
            #         # 处理可能不存在亏损交易的情况
            #         if np.isnan(avg_loss_group): avg_loss_group = 0
            #
            #         profit_ratio_group = avg_win_group / avg_loss_group if avg_loss_group != 0 else np.inf
            #         avg_hold_days_group = df_group['持有天数'].mean()
            #
            #         # 处理可能不存在盈利交易的情况
            #         if np.isnan(avg_win_group): avg_win_group = 0
            #
            #         expectancy_group = (win_rate_group / 100) * avg_win_group - (
            #                 1 - win_rate_group / 100) * avg_loss_group
            #
            #         print(
            #             f"{label:<18}: "
            #             f"总交易数: {len(df_group):<4}，"
            #             f"胜率: {win_rate_group:.2f}%，"
            #             f"均盈: {avg_win_group:.2f}% | "
            #             f"均亏: {avg_loss_group:.2f}% | "
            #             f"均持: {avg_hold_days_group:.2f}，"
            #             f"盈亏比: {profit_ratio_group:.2f}:1，"
            #             f"期望: {expectancy_group:.3f}"
            #         )
            # else:
            #     print("未产生有效交易信号")

            if not result_df.empty:
                result_df['买入日'] = result_df['买入日'].dt.strftime('%Y-%m-%d')
                save_start = time.perf_counter()  # 记录Excel保存开始时间
                save_trades_excel(result_df, rejections_df)
                save_duration = time.perf_counter() - save_start
                print(f"Excel保存耗时: {save_duration:.4f}秒")

            # 程序总耗时统计
            total_duration = time.perf_counter() - total_start
            print(f"\n\033[1m=== 性能统计 ===\033[0m")
            print(f"总运行时间: {total_duration:.2f}秒")
            print(f"股票数据处理时间: {stock_process_duration:.2f}秒")
            print(f"Excel保存时间: {save_duration:.4f}秒")
            print(f"平均每支股票处理时间: {stock_process_duration / len(stock_list) * 1000:.2f}毫秒")
