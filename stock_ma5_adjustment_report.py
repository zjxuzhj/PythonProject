import os
from datetime import datetime

import pandas as pd

import getAllStockCsv


def get_stock_data(symbol, start_date, force_update=False):
    """带本地缓存的数据获取"""
    # 生成唯一文件名（网页1）
    file_name = f"stock_{symbol}_{start_date}.parquet"
    cache_path = os.path.join("data_cache", file_name)

    # 非强制更新时尝试读取缓存
    if not force_update and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
            print(f"从缓存加载数据：{symbol}")
            return df, True  # 返回缓存标记
        except Exception as e:
            print(f"缓存读取失败：{e}（建议删除损坏文件：{cache_path}）")

    # 强制更新或缓存不存在时获取新数据（网页7）
    print(f"数据获取失败：{symbol}")
    return pd.DataFrame()


def modify_last_days_and_calc_ma5(df, tomorrow_mode=False):
    # 数据校验
    if df.empty or len(df) < (2 if tomorrow_mode else 1):
        raise ValueError(f"数据不足，至少需要{2 if tomorrow_mode else 1}个交易日数据")

    modified_df = df.copy().sort_index(ascending=True)
    modified_df['adjusted_close'] = modified_df['close']

    if tomorrow_mode:
        # === 步骤1：新增模拟数据行 ===
        # 复制最后一行并调整收盘价（网页1的行操作）
        new_row = modified_df.iloc[-1].copy()
        new_row['close'] *= 1.06
        new_row.name = new_row.name + pd.Timedelta(days=1)  # 日期顺延一日
        # 添加新行（网页3的DataFrame扩展）
        modified_df = pd.concat([modified_df, new_row.to_frame().T], axis=0)

    if not tomorrow_mode:
        # 倒序修改最后1日数据
        for i in range(1, 2):
            close_col = modified_df.columns.get_loc('close')
            modified_df.iloc[-i, close_col] = modified_df['close'].iloc[-i] * 1.06

    # 计算MA5（网页2的rolling方法[2](@ref)）
    modified_df['MA5'] = modified_df['close'].rolling(
        window=5,
        min_periods=1
    ).mean().round(2)

    return modified_df


if __name__ == '__main__':
    today = datetime.now()
    start_date = '20240201'

    query_tool = getAllStockCsv.StockQuery()
    filtered_stocks = query_tool.get_all_filter_stocks()
    stock_list = filtered_stocks[['stock_code', 'stock_name']].values

    target_stocks = ['sz000795', 'sz002437', 'sz002440', 'sh600800', 'sh603083']
    tomorrow_mode = True

    all_signals = []
    for code in target_stocks:
        # 从stock_list中匹配名称
        name = next((n for c, n in stock_list if c == code), '未知股票')

        df, _ = get_stock_data(code, start_date=start_date)
        if df.empty:
            continue

        modified_df = modify_last_days_and_calc_ma5(df, tomorrow_mode=tomorrow_mode)

        # 获取最新MA5值
        latest_data = modified_df.iloc[- 1:]

        # 记录信号
        all_signals.append({
            '代码': code,
            '名称': name,
            '最新收盘': latest_data['close'].values[-1],
            'MA5': modified_df['MA5'].iloc[-1],
            # 🌟 新增双日模式特有字段（网页1的条件判断[1](@ref)）
            **({'昨日收盘': modified_df.iloc[-2]['close']} if tomorrow_mode else {}),
            '题材': "| "+query_tool.get_theme_by_code(code),
        })

    if len(all_signals) > 0:
        result_df = pd.DataFrame(all_signals)
        print("\n目标股票五日线报告：")
        print(result_df.to_string(index=False))
    else:
        print("没有找到目标股票的有效数据，请检查：")
        print("1. 股票代码是否包含sh/sz前缀（网页3的编码规范）")
        print("2. 缓存文件是否存在完整收盘价数据（网页6的parquet格式要求）")
        print("3. 股票代码是否有效（如sh605198是华新水泥的B股代码）")
