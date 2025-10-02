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

# first_limit_up_ma5_normal的工具类，把逻辑封装成函数放在这
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

