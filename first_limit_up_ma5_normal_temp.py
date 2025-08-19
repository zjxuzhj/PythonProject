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


def calculate_stock_personality_score(df: pd.DataFrame, current_day: pd.Timestamp):
    """
    在指定日期前回溯一年，计算V1.0股性评分。
    如果数据不足一年或期间无涨停，则不进行评分。
    """
    # V1.0 公式系数
    INTERCEPT = 39.18
    COEF_SUCCESS_RATE = -29.91
    COEF_LIMIT_UPS = 1.21
    COEF_OPEN_CHANGE = -25.20
    COEF_CLOSE_CHANGE = 72.07

    # 1. 数据周期校验
    one_year_ago = current_day - pd.DateOffset(years=1)
    # 获取从一年前到当前日之前的所有数据
    hist_data_full = df.loc[one_year_ago:current_day].iloc[:-1]

    if len(hist_data_full) < 230:  # 交易日不足一年，不计算
        return None

    # 2. 筛选涨停日
    limit_up_days = hist_data_full[hist_data_full['is_limit']]
    if limit_up_days.empty:
        return 0  # 一年内无涨停，股性评分为0

    # 3. 计算各项指标
    limit_up_count = len(limit_up_days)
    successful_seals = (limit_up_days['close'] == limit_up_days['limit_price']).sum()
    success_rate = successful_seals / limit_up_count if limit_up_count > 0 else 0

    next_day_open_changes = []
    next_day_close_changes = []

    for limit_day_date in limit_up_days.index:
        limit_day_idx = df.index.get_loc(limit_day_date)
        next_day_idx = limit_day_idx + 1

        if next_day_idx < len(df):
            prev_close = df.iloc[limit_day_idx]['close']  # 注意：这里用涨停日收盘价作为基准
            next_day_data = df.iloc[next_day_idx]

            if prev_close > 0:
                open_change = (next_day_data['open'] - prev_close) / prev_close
                close_change = (next_day_data['close'] - prev_close) / prev_close
                next_day_open_changes.append(open_change)
                next_day_close_changes.append(close_change)

    if not next_day_open_changes:  # 如果所有涨停日都没有次日数据
        return 0

    avg_next_day_open = np.mean(next_day_open_changes)
    avg_next_day_close = np.mean(next_day_close_changes)

    # 4. 套入V1.0公式计算
    score = (INTERCEPT +
             (COEF_SUCCESS_RATE * success_rate) +
             (COEF_LIMIT_UPS * limit_up_count) +
             (COEF_OPEN_CHANGE * avg_next_day_open) +
             (COEF_CLOSE_CHANGE * avg_next_day_close))

    return score

