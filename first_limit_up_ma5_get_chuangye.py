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


@dataclass
class StockDataContext:
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


def get_stock_data_for_analysis(symbol: str) -> pd.DataFrame | None:
    """
    一个简化的数据加载函数，专门用于本次分析。
    - 默认从2024年的数据缓存目录 'data_cache' 加载。
    - symbol: 股票代码, e.g., 'sz300001'
    """
    cache_dir = "data_cache"
    file_name = f"stock_{symbol}_20240201.parquet"
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        return df
    except Exception as e:
        print(f"读取缓存文件失败 {file_path}: {e}")
        return None


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    为数据计算分析所需的移动平均线(MA)。
    """
    # 计算半年线
    if 'ma120' not in df.columns:
        df['ma120'] = df['close'].rolling(120, min_periods=120).mean()
    # 计算年线
    if 'ma250' not in df.columns:
        df['ma250'] = df['close'].rolling(250, min_periods=250).mean()
    return df


def analyze_gem_stock_performance_with_filters(start_date_str: str, end_date_str: str):
    """
    分析创业板股票在指定时间段内的表现，并应用多重筛选条件。

    Args:
        start_date_str (str): 区间涨幅的计算起始日期, 格式 'YYYY-MM-DD'.
        end_date_str (str): 区间涨幅的计算结束日期, 格式 'YYYY-MM-DD'.
    """
    try:
        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)
    except ValueError:
        print("错误：日期格式不正确，请输入 'YYYY-MM-DD' 格式。")
        return

    print("=" * 80)
    print("开始执行创业板股票区间表现分析 (多条件筛选版)...")
    print(f"1. 区间涨幅计算周期: {start_date.date()} 收盘价 -> {end_date.date()} 收盘价")
    print(f"2. 筛选条件 (于 {end_date.date()}):")
    print("   - 当日涨幅 > 12%")
    print("   - 年线(MA250)或半年线(MA120)在股价下方10%以内形成支撑")
    print("3. 最终结果排除末位30%的股票")
    print("=" * 80)

    query_tool = getAllStockCsv.StockQuery()
    all_stocks = query_tool.get_all_filter_stocks()[['stock_code', 'stock_name']].values

    gem_stocks = [
        (code, name) for code, name in all_stocks
        if code.startswith('sz300') or code.startswith('sz301')
    ]
    print(f"已识别出 {len(gem_stocks)} 只创业板股票，开始遍历并应用筛选条件...")

    passed_initial_filters = []
    for code, name in gem_stocks:
        df = get_stock_data_for_analysis(code)
        if df is None or df.empty:
            continue

        theme = query_tool.get_theme_by_code(code)
        if "白酒" in theme or "光伏" in theme:
            continue
        if "旅游" in theme or "旅游" in name:
            continue
        if "酒店" in theme or "酒店" in name:
            continue
        if "教育" in theme or "教育" in name:
            continue
        if "水产" in theme or "水产" in name:
            continue
        if "酒" in theme or "酒" in name:
            continue

        # 计算所需的均线
        df = prepare_analysis_data(df)

        try:
            # --- 数据提取 ---
            # 获取结束日及其前一日的数据
            end_date_data = df.loc[:end_date].iloc[-1]
            ma120_ma250_date_data = df.iloc[-1]
            # 严格确保定位到的数据就是结束日当天的数据，避免因停牌等问题使用旧数据
            if end_date_data.name.date() != end_date.date():
                continue

            prev_day_data = df.loc[:end_date - pd.Timedelta(days=1)].iloc[-1]
            start_date_data = df.loc[:start_date].iloc[-1]

            # 提取价格和均线值
            end_price = end_date_data['close']
            prev_day_close = prev_day_data['close']
            start_price = start_date_data['close']
            end_price_ma120 = ma120_ma250_date_data['close']
            ma120 = ma120_ma250_date_data['ma120']
            ma250 = ma120_ma250_date_data['ma250']

            # --- 执行筛选 ---

            # 条件1: 结束日当天涨幅 > 12%
            if not (pd.notna(end_price) and pd.notna(prev_day_close) and prev_day_close > 0):
                continue
            daily_gain = (end_price - prev_day_close) / prev_day_close
            if daily_gain <= 0.149:
                continue

            # 条件2: 年线或半年线形成有效支撑
            # 检查半年线支撑
            is_ma120_support = pd.notna(ma120) and 0 < ma120 <= end_price_ma120 and ((end_price_ma120 - ma120) / ma120) < 0.03
            # 检查年线支撑
            is_ma250_support = pd.notna(ma250) and 0 < ma250 <= end_price_ma120 and ((end_price_ma120 - ma250) / ma250) < 0.03

            # if not (is_ma120_support or is_ma250_support):
            #     continue

            # --- 通过所有筛选，计算区间涨幅并记录 ---
            if pd.notna(start_price) and start_price > 0:
                weekly_change = (end_price - start_price) / start_price
                ma120_dist_pct = (end_price_ma120 - ma120) / ma120 if pd.notna(ma120) and ma120 > 0 else np.nan
                ma250_dist_pct = (end_price_ma120 - ma250) / ma250 if pd.notna(ma250) and ma250 > 0 else np.nan

                passed_initial_filters.append({
                    "股票代码": getAllStockCsv.convert_to_standard_format(code),
                    "股票名称": name,
                    "区间涨幅": weekly_change,
                    "9-30日涨幅": daily_gain,
                    "距MA120支撑": ma120_dist_pct,
                    "距MA250支撑": ma250_dist_pct,
                })

        except (KeyError, IndexError):
            continue

    if not passed_initial_filters:
        print("\n分析结束：经过筛选后，没有股票满足所有条件。")
        return

    print(f"\n{len(passed_initial_filters)} 只股票通过了个股筛选，现在进行排序和末位淘汰...")

    # 创建DataFrame并按区间涨幅降序排序
    performance_df = pd.DataFrame(passed_initial_filters)
    sorted_df = performance_df.sort_values(by="区间涨幅", ascending=False)

    # 条件3: 排除排名后30%的股票
    num_to_keep = int(len(sorted_df) * 0.3)
    if num_to_keep == 0 and len(sorted_df) > 0:
        num_to_keep = 1  # 如果数量太少，至少保留一个

    final_df = sorted_df.iloc[:num_to_keep].reset_index(drop=True)

    print(f"排除末位30%后，最终剩下 {len(final_df)} 只股票。")

    # --- 格式化并展示最终结果 ---
    final_df.insert(0, '排名', final_df.index + 1)
    final_df["区间涨幅"] = final_df["区间涨幅"].apply(lambda x: f"{x:.2%}")
    final_df["9-30日涨幅"] = final_df["9-30日涨幅"].apply(lambda x: f"{x:.2%}")
    final_df["距MA120支撑"] = final_df["距MA120支撑"].round(2)
    final_df["距MA250支撑"] = final_df["距MA250支撑"].round(2)

    print("\n分析完成！最终筛选结果如下：")
    print("-" * 80)
    print(final_df.to_string())
    print("-" * 80)


if __name__ == '__main__':
    # 设置区间涨幅的计算周期
    # “924-930的涨幅”通常指从9月23日的收盘价算起，到9月30日收盘价结束
    start_day = '2024-09-23'
    end_day = '2024-09-30'

    # 调用包含所有筛选条件的分析函数
    analyze_gem_stock_performance_with_filters(start_date_str=start_day, end_date_str=end_day)

